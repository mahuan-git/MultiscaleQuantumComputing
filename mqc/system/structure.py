import numpy as np
from mqc.tools.iotools import read_poscar, read_mol_structure
from mqc.tools.tools import get_distance


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

    def build(self):
        self.read_geometry()
        self.structure_initialize()
        self.get_qm_atom_list()
        self.set_qm_geometry()
        self.get_mm_atom_list()

    def read_geometry(self):
        if self.file_name is not None:
            if self.file_format == "POSCAR":
                self.input_geometry = read_poscar(fname=self.file_name)
                self.geometry = self.input_geometry
            elif self.file_format == "mol":
                self.input_geometry= read_mol_structure(fname = self.file_name)
                self.geometry = self.input_geometry
        else:
            pass
        
        assert self.geometry is not None, "Either geometry or file path should be given"
    
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
    


class Structure_Al(Structure):
    def __init__(self,geometry = None,file_name = None,file_format = None, cluster_size = None):
        super().__init__(geometry=geometry, file_name=file_name, file_format=file_format)
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
        self._select_al_cluster()
        self.geometry = self._molecule+self._substrate_select
        self._get_bonding_atom_index()
        return
    
    def _devide_molecule_substrate(self):
        for atom in self.input_geometry:
            if atom[0] == "Al":
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

    def _select_al_cluster(self):
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

class Structure_protein(Structure):
    '''structure class for proteins. To be finished'''
    def structure_initialize():
        pass
        return