import numpy as np
from mqc.tools.iotools import read_poscar, read_mol_structure
from pyscf import gto, scf, cc
from mqc.tools.tools import get_distance

class Structure(object):
    def __init__(self,geometry = None,file_name = None,file_format = None,basis = 'sto-3g',charge = 0,restricted =True,build = True,run_ccsd = False):
        
        self.input_geomrtry = geometry
        self.geometry =geometry
        self.file_name = file_name
        self.file_format = file_format
        
        if file_name is not None:
            if file_format == "POSCAR":
                self.input_geometry = read_poscar(fname=file_name)
                self.geometry = self.input_geometry
            elif file_format == "mol":
                self.input_geometry= read_mol_structure(fname = file_name)
                self.geometry = self.input_geometry
        else:
            raise ValueError("Either geometry or file path should be given")
        self.basis = basis
        self.charge = charge
        self.restricted = restricted
        self.mol = None
        self.mf = None
        self.run_ccsd = run_ccsd
        self.cc = None
    
    def run_pyscf_rhf(self):
        mol=gto.Mole()
        mol.atom=self.geometry
        mol.basis = self.basis
        mol.charge = self.charge
        assert( mol.nelectron%2 ==0)
        mol.spin = 0
        mol.build()
        self.mol = mol

        mf = scf.RHF(mol)
        mf.verbose = 3
        mf.max_cycle = 1000
        mf.scf(dm0=None)
        self.mf = mf
    
    def run_pyscf_uhf(self):
        mol=gto.Mole()
        mol.atom=self.geometry
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = mol.nelectron%2
        mol.build()
        self.mol = mol

        mf = scf.UHF(mol)
        mf.verbose = 3
        mf.max_cycle = 1000
        mf.scf(dm0=None)
        self.mf = mf
        
    def run_pyscf_rccsd(self):
        if self.mf == None:
            self.run_pyscf_rhf()
        mycc = cc.RCCSD(self.mf).run()
        self.cc = mycc
    
    def run_pyscf_uccsd(self):
        if self.mf == None:
            self.run_pyscf_uhf()
        mycc = cc.UCCSDCCSD(self.mf).run()
        self.cc = mycc
    
    def build(self):
        if self.restricted:
            self.run_pyscf_rhf()
            if self.run_ccsd:
                self.run_pyscf_rccsd()
        else:
            self.run_pyscf_uhf()
            if self.run_ccsd:
                self.run_pyscf_uccsd()
        
    
    def print_strucyure(self,file_format = None):
        '''TBD'''
        pass

    def print_structure_for_Gauss_View(self,file_name = "structure.com"):
        geometry = self.geometry
        GV_file = open(file_name,'w+')
        GV_file.write("# HF/3-21G** opt pop=full gfprint\n\nTitle: Created by Jmol version 14.31.60  2021-10-18 20:23\n\n0 1\n")
        for i in range(len(geometry)):
            GV_file.write(geometry[i][0]+'    '+str(geometry[i][1][0])+'   '+str(geometry[i][1][1])+'   '+str(geometry[i][1][2])+'   \n')
        GV_file.close()

    def structure_initialize(self):
        '''Initialize the given structure. No initialization is applied here'''
        pass
        return
    


class Structure_Al(Structure):
    def __init__(self,geometry = None,file_name = None,file_format = None,basis = 'sto-3g',charge = 0,restricted =True, build = True,run_ccsd = False,select_al_cluster = True, cluster_size = None):
        super().__init__(geometry=geometry,file_name=file_name,file_format=file_format,basis=basis,charge=charge,restricted=restricted, build=build,run_ccsd=run_ccsd)
        self._molecule = []
        self._molecule_bonding_atom = None
        self._substrate = []
        self._substrate_bonding_atom = None
        self._select_al_cluster = select_al_cluster
        if select_al_cluster == True:
            self._substrate_select = []
            self._cluster_size = cluster_size
        self.structure_initialize()
        if build == True:
            self.build()

    def structure_initialize(self):
        '''initialize structure for subsequence calculation.'''
        self._devide_molecule_substrate()
        self._find_bonding_atoms()
        if self._select_al_cluster==True:
            assert self._cluster_size is not None , "cluster_size should be given when select_al_cluster is True"
            self.select_al_cluster()
        else:
            self._substrate_select = self._substrate
        self.geometry = self._molecule+self._substrate_select
        self.get_bonding_atom_index()
        return
    
    def _devide_molecule_substrate(self):
        for atom in self.input_geometry:
            if atom[0] == "Al":
                self._substrate.append(atom)
            else:
                self._molecule.append(atom)

    def _find_bonding_atoms(self):
        '''assumes that the bonding atom in the molecule is the one that has the minimum value in Z direction'''
        self._molecule_bonding_atom = ("H",(0,0,50))
        for atom in self._molecule:
            if (atom[0] is not 'H') and (atom[1][2]< self._molecule_bonding_atom[1][2]):
                self._molecule_bonding_atom = atom
        dist_min = 5
        for atom in self._substrate:
            dist = get_distance(atom[1],self._molecule_bonding_atom[1])
            if dist < dist_min:
                dist_min = dist
                self._substrate_bonding_atom = atom
        return

    def select_al_cluster(self):
         
        for atom in self._substrate:
            if get_distance(atom[1],self._substrate_bonding_atom[1])<self._cluster_size:
                self._substrate_select.append(atom)

    def get_bonding_atom_index(self):
        self._bonding_atom_index = dict()
        for idx in range(len(self.geometry)):
            if np.isclose(self.geometry[idx][1]-self._molecule_bonding_atom[1],0).all():
                self._bonding_atom_index.update({"mol":idx})
            if np.isclose(self.geometry[idx][1]-self._substrate_bonding_atom[1],0).all():
                self._bonding_atom_index.update({"al":idx})               

            


class Structure_protein(Structure):
    '''structure class for proteins. To be finished'''
    def structure_initialize():
        pass
        return