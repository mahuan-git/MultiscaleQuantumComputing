import numpy as np
from mqc.tools.tools import get_distance
import copy
from .structure import Structure,Structure_protein
from pyscf import gto, scf, cc

class Fragment(object):
    ''' Define fragments 
    '''  
    def __init__(   self, 
                    structure: Structure,
                    basis = 'sto-3g',
                    charge = 0,
                    restricted =True,
                    run_ccsd = False
                    ):
        self.structure = structure
        self.qm_geometry = structure.qm_geometry

        self.basis = basis
        self.charge = charge
        self.restricted = restricted
        self.mol = None
        self.mf = None
        self.run_ccsd = run_ccsd
        self.cc = None
        self.cc_energy = None
        self.connection = []

        self.qm_fragment = None
        self.qm_atom_charge = None
        self.qm_fragment_charges = None

    def build(self):
        self.get_qm_fragment()
        if self.restricted:
            self.run_pyscf_rhf()
        else:
            self.run_pyscf_uhf()
        self.get_qm_fragment_charge()

        if self.run_ccsd:
            if self.restricted:
                self.run_pyscf_rccsd()
            else:
                self.run_pyscf_uccsd()
        self.get_connection()

    def run_pyscf_rhf(self):
        mol=gto.Mole()
        mol.atom=self.qm_geometry
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
        mol.atom=self.qm_geometry
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
        mycc = cc.UCCSD(self.mf).run()
        self.cc = mycc
    
    def get_qm_fragment(self,qm_fragment = None):
        """By default, all atoms are put in one fragment"""
        if qm_fragment is not None:
            self.qm_fragment = qm_fragment
        else:
            self.qm_fragment = [self.structure.qm_atom_list]

    def get_qm_fragment_charge(self):
        """Calculate charges every fragment should carry"""
        (pop, chg), dip = self.mf.analyze(verbose=0,with_meta_lowdin=True)
        self.qm_atom_charge = chg
        assert self.qm_fragment is not None
        self.qm_fragment_charges = []
        for frag in self.qm_fragment:
            charge_tmp = 0
            for idx in frag:
                charge_tmp+=chg[idx]
            self.qm_fragment_charges.append(charge_tmp)

    def get_connection(self):
        connection=[]
        natom = len(self.qm_geometry)
        for i in range(natom):
            connection_tmp = []
            for j in range(natom):
                if j==i:
                    dist = 2
                else:
                    dist = get_distance(self.qm_geometry[i][1],self.qm_geometry[j][1])
                if dist < 1.75:
                    connection_tmp.append(j)
            connection.append(connection_tmp)
        self.connection = connection


class SimpleFragment(Fragment):
    ''' fragments for simple systems such as hydrogen chain. Devide fragments based on number of atoms in per fragment.
    '''  
    def __init__(self, structure: Structure, basis='sto-3g', charge=0, restricted=True, run_ccsd=False,natom_per_fragment=2):
        super().__init__(structure, basis, charge, restricted, run_ccsd)
        self.natom_per_fragment = natom_per_fragment

    def get_qm_fragment(self):
        '''
        '''
        self.qm_fragment = []
        assert(len(self.qm_geometry)%self.natom_per_fragment==0)
        self.nfrag = len(self.qm_geometry)//self.natom_per_fragment
        for i in range(self.nfrag):
            frag = np.arange(self.natom_per_fragment*i,self.natom_per_fragment*i+self.natom_per_fragment)
            self.qm_fragment.append(list(frag))

class Fragment_Metal_Mol_MBE(Fragment):
    def get_qm_fragment(self, qm_fragment=None):
        if qm_fragment is not None:
            return super().get_qm_fragment(qm_fragment)
        else:
            """define default qm fragment"""
            self.qm_fragment = []
            substrate_list = []
            molecule_list = []
            for idx in range(len(self.qm_geometry)):
                if self.qm_geometry[idx][0] ==self.structure._metal_name:
                    substrate_list.append(idx)
                else:
                    molecule_list.append(idx)
            self.qm_fragment.append(substrate_list)
            self.qm_fragment.append(molecule_list)

class Fragment_Metal_Mol_DMET(Fragment):
    def __init__(self, structure: Structure, basis='sto-3g', charge=0, restricted=True, run_ccsd=False):
        super().__init__(structure, basis, charge, restricted, run_ccsd)

    def get_qm_fragment(self, qm_fragment=None):
        if qm_fragment is not None:
            return super().get_qm_fragment(qm_fragment)
        else:
            self.qm_fragment = []
            self.qm_fragment.append(list(self.structure._bonding_atom_index.values()))

class Fragment_protein(Fragment):
    """Fragment class for protein-ligend models"""
    def __init__(self, 
                 structure: Structure_protein, 
                 basis='sto-3g', 
                 charge=0, 
                 restricted=True, 
                 run_ccsd=False):
        super().__init__(structure, basis, charge, restricted, run_ccsd)
        self.structure = structure
        self._C_side = None
        self._N_side = None
        self._core_carbon = None
        self._R_group = None
        self._ligand_fragment_list = None

    def build(self):
        self.qm_atom_charge = self.structure._atom_charge
        self.divide_residues()
        self.get_fragment_for_residues()
        self.get_fragment_charge()
        self.add_fragment_for_ligand()
        self.get_fragment_charge()

    def divide_residues(self):
        N_side = []
        core_carbon = []
        C_side = []
        for residue in self.structure._res:
            N_idx = residue[0]
            if self.structure._atom_type[N_idx].startswith("O"):
                continue
            assert(self.structure._atom_type[N_idx] in ["N.am","N.3","N.4"])
            N_H_idx = []
            N_H_idx.append(N_idx)
            for atom_idx in residue:
                if self.structure._atom_label[atom_idx] =="CA":
                    core_carbon_idx = atom_idx
                if ((N_idx,atom_idx) in self.structure._bonds) or ((atom_idx,N_idx) in self.structure._bonds):
                    if self.structure._atom_type[atom_idx]=="H":
                        N_H_idx.append(atom_idx)
            N_side.append(N_H_idx)

            COO_idx = []
            for atom_idx in residue:
                if ((core_carbon_idx,atom_idx) in self.structure._bonds) or ((atom_idx,core_carbon_idx) in self.structure._bonds ):
                    if self.structure._atom_type[atom_idx]=="H":
                        core_C_H_idx = atom_idx
                    elif self.structure._atom_type[atom_idx] =="C.2":
                        COO_C_idx = atom_idx
                        COO_idx.append(COO_C_idx)
            core_carbon.append([core_carbon_idx,core_C_H_idx])

            for atom_idx in residue:
                if ((COO_C_idx,atom_idx) in self.structure._bonds) or ((atom_idx,COO_C_idx) in self.structure._bonds):
                    if self.structure._atom_type[atom_idx].startswith("O"):
                        COO_idx.append(atom_idx)
            C_side.append(COO_idx)

        assert(len(C_side)==len(N_side)==len(core_carbon)==len(self.structure._res))
        R_group = copy.deepcopy(self.structure._res)
        for i in range(len(C_side)):
            for idx in C_side[i]:
                R_group[i].remove(idx)
            for idx in core_carbon[i]:
                R_group[i].remove(idx)
            for idx in N_side[i]:
                R_group[i].remove(idx)
        self._C_side = C_side
        self._N_side = N_side
        self._core_carbon = core_carbon
        self._R_group = R_group
        return C_side, N_side, core_carbon, R_group    

    def get_fragment_for_residues(self) -> list:
        fragment = [[]for i in range(len(self._core_carbon))]
        for i in range(len(self._N_side)):
            N_idx = self._N_side[i][0]
            assert (self.structure._atom_type[N_idx] in ["N.4","N.3","N.am"])
            if self.structure._atom_type[N_idx] in ["N.4","N.3"]:  ##at the starting position
                fragment[i]=fragment[i]+self._N_side[i]
            elif self.structure._atom_type[N_idx] == "N.am":
                fragment[i-1] = fragment[i-1]+self._N_side[i]
            fragment[i] = fragment[i] + self._core_carbon[i]
            fragment[i] = fragment[i] + self._C_side[i]
        for i in range(len(self._R_group)):
            if len(self._R_group[i]) < 5:
                fragment[i]=fragment[i]+self._R_group[i]
            else:
                fragment.append(self._R_group[i])
        self.qm_fragment = fragment
        return fragment

    def add_fragment_for_ligand(self,atom_list = None) -> list:
        if atom_list ==None:
            ##give a default ligand fragment
            atom_list=[
                ['0', '1', '2', '3', '4', '5', '7', '8', '9', '14', '15'],
                ['10', '11', '12', '13'],
                ['16', '17', '18', '19', '20', '21', '22', '23', '24', '40'],
                ['6', '25', '26', '27', '28', '34', '41', '42', '37', '38', '39'],
                ['29', '30', '31', '32', '33', '35', '36']]
        for i in range(len(atom_list)):
            for j in range(len(atom_list[i])):
                atom_list[i][j] =int(atom_list[i][j]) +len(self.structure._atom_label)
        self._ligand_fragment_list = atom_list
        self.qm_fragment = self.qm_fragment + atom_list

    def get_fragment_charge(self):
        frag_charge = []
        for frag in self.qm_fragment:
            charge = 0
            for idx in frag:
                charge = charge+self.structure._atom_charge[idx]
            charge = round(charge)
            frag_charge.append(charge)
        assert(len(frag_charge)==len(self.qm_fragment))
        return frag_charge
    
