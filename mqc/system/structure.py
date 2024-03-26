import numpy as np
from mqc.tools.iotools import read_mol2_structure
from mqc.tools.tools import get_distance
from openbabel import openbabel
from openbabel import pybel


def hydrogen_ring(natom = 10,bondlength = 1.0):
    geometry = []
    r = 0.5 * bondlength / np.sin(np.pi/natom)
    for i in range(natom):
        theta = i * (2*np.pi/natom)
        geometry.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))
    return geometry


def hydrogen_chain(natom:int = 10,  
                     bondlength : float=1.0
                     ):
    geometry = []
    for i in range(natom):
        geometry.append(('H', (0, 0, i*bondlength)))
    return geometry


def Be_ring(natom:int = 30,  
                     bondlength : float=2.0
                     ):
    geometry = []
    r = 0.5 * bondlength / np.sin(np.pi/natom)
    for i in range(natom):
        theta = i * (2*np.pi/natom)
        geometry.append(('Be', (r*np.cos(theta), r*np.sin(theta), 0)))
    return geometry


def carbon_ring(shift=20.0):
    nat = 18
    shift =shift*2*np.pi/360
    R = 7.31/2
    geometry = []
    angle = 0.0
    for i in range( nat // 2 ):
        geometry.append(('C', (R * np.cos(angle        ), R * np.sin(angle        ), 0.0)))
        geometry.append(('C', (R * np.cos(angle + shift), R * np.sin(angle + shift), 0.0)))
        angle += 4.0 * np.pi / nat
    return geometry


class Structure(object):
    def __init__(self,geometry = None,file_name = None,file_format = None):
        
        self.input_geometry = geometry
        self.geometry =geometry
        self.file_name = file_name
        self.file_format = file_format

        self.qm_atom_list = None
        self.qm_geometry = None

        self.mm_atom_list = None
        self.mm_coords = None
        self.mm_charges = None
        
        self.obmol = None
        self.pymol = None
        self.obconv = None

    def build(self):
        self.read_geometry()
        self.structure_initialize()
        self.get_qm_atom_list()
        self.set_qm_geometry()
        self.get_mm_atom_list()
        self.get_mm_charge_mm_coords()

    def read_geometry(self) -> pybel.Molecule:
        if self.file_name is not None:
            self.input_geometry = []
            self.obmol = openbabel.OBMol()
            self.obconv = openbabel.OBConversion()
            self.obconv.SetInAndOutFormats(self.file_format,None)
            self.obconv.ReadFile(self.obmol, self.file_name)
            self.obmol.AddHydrogens()
            self.pymol = pybel.Molecule(self.obmol)
            for atom in self.pymol.atoms:
                from pyscf.data.elements import ELEMENTS
                self.input_geometry.append((ELEMENTS[atom.atomicnum],np.array(atom.coords)))
            self.geometry = self.input_geometry
        return self.obmol

    def write_geometry(self,out_put_file_name="out_file.mol2",out_put_file_format="mol2"):
        if self.obmol == None or self.obconv ==None:
            self.read_geometry()
        self.obconv.SetInAndOutFormats(self.file_format,out_put_file_format)
        self.obconv.WriteFile(self.obmol, out_put_file_name)
    
    def structure_initialize(self):
        '''Initialize the given structure. No initialization is applied here'''
        pass
        return
    
    def get_qm_atom_list(self,qm_atom_list = None):
        '''
        function to define qm atoms, by default all atoms are identified as qm atoms.
        '''
        if qm_atom_list is not None:
            self.qm_atom_list = qm_atom_list
        elif self.qm_atom_list is not None:
            return
        else:
            self.qm_atom_list = list(range(len(self.geometry)))

    def set_qm_geometry(self):        
        self.qm_geometry = []
        for idx in range(len(self.geometry)):
            if idx in self.qm_atom_list:
                self.qm_geometry.append(self.geometry[idx])

    def get_mm_atom_list(self):
        self.mm_atom_list = []
        for idx in range(len(self.geometry)):
            if idx not in self.qm_atom_list:
                self.mm_atom_list.append(idx)

    def get_mm_charge_mm_coords(self,basis = "sto-3g"):
        if (self.mm_atom_list is not None) and (len(self.mm_atom_list) > 0):
            from pyscf import gto,scf
            self.mm_charges = []
            self.mm_coords = []
            mol = gto.Mole()
            mol.atom = self.geometry
            mol.basis = basis
            hf = scf.HF(mol).run()
            (pop, chg), dip = hf.analyze(verbose=0,with_meta_lowdin=True) 
            for idx in self.mm_atom_list:
                self.mm_charges.append(chg[idx])
                self.mm_coords.append(self.geometry[idx][1])
        else:
            self.mm_charges = []
            self.mm_coords = []

    
class Structure_Metal_Mol(Structure):
    def __init__(self,geometry = None,file_name = None,file_format = None, metal_name = "Al",cluster_size = None):
        super().__init__(geometry=geometry, file_name=file_name, file_format=file_format)
        self._metal_name = metal_name
        self._molecule = []
        self._molecule_bonding_atom = None
        self._substrate = []
        self._substrate_bonding_atom = None
        self._substrate_select = []
        self._cluster_size = cluster_size
        self._bonding_atom_index = None

    def structure_initialize(self,cluster_size = None):

        '''initialize structure for subsequence calculation.'''
        self._devide_molecule_substrate()
        self._find_bonding_atoms()
        self._select_al_cluster(cluster_size)
        self.geometry = self._molecule+self._substrate_select
        self._get_bonding_atom_index()
        return
    
    def _devide_molecule_substrate(self):
        for atom in self.input_geometry:
            if atom[0] == self._metal_name:
                self._substrate.append(atom)
            else:
                self._molecule.append(atom)

    def _find_bonding_atoms(self):
        '''assumes that the bonding atom in the molecule is the one that has the minimum coordinate value in Z direction'''
        self._molecule_bonding_atom = ("H",(0,0,50))
        for atom in self._molecule:
            if (atom[0] != 'H') and (atom[1][2]< self._molecule_bonding_atom[1][2]):
                self._molecule_bonding_atom = atom
        dist_min = 5
        for atom in self._substrate:
            dist = get_distance(atom[1],self._molecule_bonding_atom[1])
            if dist < dist_min:
                dist_min = dist
                self._substrate_bonding_atom = atom
        return

    def _select_al_cluster(self,cluster_size = None):
        if cluster_size is not None:
            self._cluster_size = cluster_size
        for atom in self._substrate:
            if get_distance(atom[1],self._substrate_bonding_atom[1])<self._cluster_size:
                self._substrate_select.append(atom)

    def _get_bonding_atom_index(self):
        self._bonding_atom_index = dict()
        for idx in range(len(self.geometry)):
            if np.isclose(self.geometry[idx][1]-self._molecule_bonding_atom[1],0).all():
                self._bonding_atom_index.update({"mol":idx})
            if np.isclose(self.geometry[idx][1]-self._substrate_bonding_atom[1],0).all():
                self._bonding_atom_index.update({"al":idx})               


class Structure_Al(Structure_Metal_Mol):
    def __init__(self, geometry=None, file_name=None, file_format=None, cluster_size=None):
        super().__init__(geometry, file_name, file_format, "Al", cluster_size)


class Structure_protein(Structure):
    '''structure class for proteins. To be finished'''
    def __init__(self, geometry=None, file_name=None, file_format=None):
        super().__init__(geometry, file_name, file_format)
        self._res = None
        self._res_names = None
        self._protein_geometry = None
        self._bonds = None
        self._bond_types = None
        self._atom_type = None
        self._atom_label = None
        self._atom_charge = None
        self._res_charge = None
        self._ligand_geometry = None
        self._ligand_charge = None
        self._ligand_atom_idx = None
        self._mm_coords = None
        self._mm_charge = None

    def read_geometry(self):
        res,res_names, protein_geometry, bonds, bond_types,atom_type,atom_label,\
            atom_charge,res_charge,ligand_geometry,ligand_atom_charge,ligand_atom_idx,mm_coords,mm_charges=read_mol2_structure(self.file_name)
        self.input_geometry = None 
        self.geometry = protein_geometry + ligand_geometry
        self._protein_geometry = protein_geometry
        self._res = res
        self._res_names = res_names
        self._bonds = bonds
        self._bond_types = bond_types
        self._atom_type = atom_type
        self._atom_label = atom_label
        self._atom_charge = atom_charge
        for i in range(len(res_charge)):
            if res_charge[i]<0.1 and res_charge[i]>-0.1:
                res_charge[i]=0
            elif res_charge[i]>0.9 and res_charge[i]<1.1:
                res_charge[i]=1
            elif res_charge[i]<-0.9 and res_charge[i]>-1.1:
                res_charge[i]=-1
            else:
                pass
        self._res_charge = res_charge
        self._ligand_geometry = ligand_geometry
        self._ligand_charge = ligand_atom_charge
        self._ligand_atom_idx = ligand_atom_idx
        self._mm_coords = mm_coords
        self._mm_charges = mm_charges
    
    def structure_initialize(self):
        pass
        return
    
    def get_mm_atom_list(self):
        pass
        return
    
    def get_mm_charge_mm_coords(self):
        self.mm_coords = self._mm_coords
        self.mm_charges = self._mm_charges
    