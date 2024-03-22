import localintegrals
import dmet as dmet
import numpy as np
from mqc.system.fragment import Fragment

class DMET_base(object):
    def __init__(self,fragment: Fragment, method = "CC",SCmethod = "NONE", isTI = False):
        self.fragment = fragment
        self.mol = fragment.mol
        self.mf = fragment.mf
        self.basis = fragment.basis
        self.localint = None
        self.impurity_clusters = None
        self.isTI = isTI  #Transition Invariant or not.
        self.method = method
        self.SCmethod = SCmethod
        self.dmet_iter = None
        self.energy = None

    def build(self):
        self.get_localintegral()
        self.get_impurity_clusters()
        self.dmet_iter = dmet.dmet( self.localint, self.impurity_clusters, self.isTI, self.method, self.SCmethod)

    def get_localintegral(self, method = "meta_lowdin", molden_name = "tmp.molden"):
        self.localint = localintegrals.localintegrals( self.mf, range( self.mol.nao_nr() ), method )
        self.localint.molden( molden_name )
        self.localint.TI_OK = True
    
    def mulliken_pop_analyse(self, atom_list = None, thres = 0.01):
        mf = self.fragment.mf
        (pop, chg), dip = mf.analyze(verbose=0,with_meta_lowdin=True)
        mol = mf.mol
        if atom_list == None:
            atom_list = range(len(mol.atom))
        atom_list = [str(i) for i in atom_list]
        orbitals = mol.spheric_labels()
        assert len(orbitals)==len(pop)==mol.nao_nr()
        imp_orbitals = []
        for orb in orbitals:
            atom_idx, atom_name, atom_orb=orb.split()
            idx = orbitals.index(orb)
            if (atom_idx in atom_list) and (thres < pop[idx] < (2-thres)):
                imp_orbitals.append(idx)
        return imp_orbitals
    
    def get_impurity_clusters(self):
        self.impurity_clusters = []
        for frag in self.fragment.qm_fragment:
            imp_orbs = self.mulliken_pop_analyse(atom_list = frag,thres = 0.01)
            impurities = np.zeros( [ self.localint.Norbs ], dtype=int )
            for orb in imp_orbs:
                impurities[ orb ] = 1
            self.impurity_clusters.append( impurities )
            
    def run(self):
        self.energy=self.dmet_iter.doselfconsistent()
