import numpy as np
from mqc.tools.tools import get_distance
import copy
from .structure import Structure, Structure_Al
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
        self.geometry = structure.qm_geometry

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

    def build(self):
        self.get_qm_fragment()

        if self.restricted:
            self.run_pyscf_rhf()
        else:
            self.run_pyscf_uhf()
        
        if self.run_ccsd:
            if self.restricted:
                self.run_pyscf_rccsd()
            else:
                self.run_pyscf_uccsd()
        self.get_connection()

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
        mycc = cc.UCCSD(self.mf).run()
        self.cc = mycc
    
    def get_qm_fragment(self,qm_fragment = None):
        """By default, all atoms are put in one fragment"""
        if qm_fragment is not None:
            self.qm_fragment = qm_fragment
        else:
            self.qm_fragment = [self.structure.qm_atom_list]

    def get_connection(self):
        connection=[]
        natom = len(self.geometry)
        for i in range(natom):
            connection_tmp = []
            for j in range(natom):
                if j==i:
                    dist = 2
                else:
                    dist = get_distance(self.geometry[i][1],self.geometry[j][1])
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
        assert(len(self.geometry)%self.natom_per_fragment==0)
        self.nfrag = len(self.geometry)//self.natom_per_fragment
        for i in range(self.nfrag):
            frag = np.arange(self.natom_per_fragment*i,self.natom_per_fragment*i+self.natom_per_fragment)
            self.qm_fragment.append(list(frag))

class Fragment_Al_MBE(Fragment):
    def get_qm_fragment(self, qm_fragment=None):
        if qm_fragment is not None:
            return super().get_qm_fragment(qm_fragment)
        else:
            """define default qm fragment"""
            self.qm_fragment = []
            substrate_list = []
            molecule_list = []
            for idx in range(len(self.geometry)):
                if self.geometry[idx][0] =="Al":
                    substrate_list.append(idx)
                else:
                    molecule_list.append(idx)
            self.qm_fragment.append(substrate_list)
            self.qm_fragment.append(molecule_list)

class Fragment_Al_DMET(Fragment):
    def __init__(self, structure: Structure, basis='sto-3g', charge=0, restricted=True, run_ccsd=False):
        super().__init__(structure, basis, charge, restricted, run_ccsd)

    def get_qm_fragment(self, qm_fragment=None):
        if qm_fragment is not None:
            return super().get_qm_fragment(qm_fragment)
        else:
            self.qm_fragment = []
            self.qm_fragment.append(list(self.structure._bonding_atom_index.values()))

    def mulliken_pop_analyse(self):
        pass

class Fragment_protein(Fragment):
    ''' Fragment class especially for proteins.
    '''
    def __init__(self, filename,natom_per_fragment = None,atom_list = None, mm_charges=None):
        '''
        '''
        self._filename = filename
        self._nfrag = 0
        self._qm_list = []
        res,res_names, geometry, bonds, bond_types,atom_type,atom_label,atom_charge,res_charge,ligand_geo,ligand_atom_charge,ligand_atom_idx,mm_coords,mm_charge=self.read_structure()
        self._geometry = geometry
        self._res = res
        self._res_names = res_names
        self._bonds = bonds
        self._bond_types = bond_types
        self._atom_type = atom_type
        self._atom_label = atom_label
        #print(len(self._atom_label))
        self._atom_charge = atom_charge
        #print(self._atom_charge)
        assert (len(res_charge) == len(res))
        assert(len(self._atom_charge)==len(self._geometry)+len(ligand_geo))
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
        #print(res_charge)
        self._ligand_geo = ligand_geo
        self._ligand_charge = ligand_atom_charge
        self._ligand_atom_idx = ligand_atom_idx
        #print(ligand_atom_charge)
        self._mm_coords = mm_coords
        self._mm_charge = mm_charge

        self.geo = self._geometry + self._ligand_geo

        C_side, N_side, core_carbon, R_group = self.divide_residues()
        self._C_side = C_side
        self._N_side = N_side
        self._core_carbon = core_carbon
        self._R_group = R_group
        self.fragment = self.get_fragment_for_residues()
        self.add_fragment_for_ligand()
        self.frag_charges=self.get_fragment_charge()
        self._qm_list = self.fragment
        self._mm_list = []
        self._mm_charges = mm_charges
        self.natom_per_fragment = natom_per_fragment
        self.atom_list = atom_list
        self.connection = None
        self.get_connection()
        print("number of fragments:", len(self.fragment))
        print("number of atoms:",len(self.geo))

    def read_structure(self):
        file = open(self._filename,"r+")
        lines = file.readlines()
        atom_idx = lines.index("@<TRIPOS>ATOM\n")
        bond_idx = lines.index("@<TRIPOS>BOND\n")
        try:
            struct_idx = lines.index("@<TRIPOS>SUBSTRUCTURE\n")
        except (ValueError):
            print("Substructure part not found")
            strcut_idx = None
        if struct_idx is not None:
            residue_num = 0
            for line in lines[struct_idx+1:]:
                if len(line)<5:
                    continue
                if line.startswith("#"):
                    continue
                #print(line)
                if line.split()[6] in ["GLY","ALA","PRO","VAL","LEU","ILE","SER","THR","GLN","ASN","MET","CYS","PHE","TYR","TRP","ASP","GLU","SEC","ARG","LYS","HIS"]:
                    residue_num += 1
        else:
            #print("no substrcuture part found")
            residue_num = int(lines[bond_idx-1].split()[6])
        print("Residue number:",residue_num)
        res=[[] for i in range(residue_num)]
        res_names = []
        geometry = []
        atom_type = []
        atom_label = []
        atom_charge = []
        mm_coords = []
        mm_charge = []
        ligand_geo = []
        ligand_atom_charge = []
        #lig_mark = False
        #other_atoms = 0
        res_atom_idx=[]
        ligand_atom_idx= []
        res_idx = 0
        for line in lines[atom_idx+1:bond_idx]:
            if len(line)<5:
                continue
            if line.startswith("#"):
                continue
            data = line.split()
            res_name = data[7]
            if res_name[0:3] in ["GLY","ALA","PRO","VAL","LEU","ILE","SER","THR","GLN","ASN","MET","CYS","PHE","TYR","TRP","ASP","GLU","SEC","ARG","LYS","HIS"]:
                res_atom_idx.append(data[0])
                atom_idx = int(data[0])-1 - len(mm_coords)
                geometry.append((data[1][0],(float(data[2]),float(data[3]),float(data[4]))))
                if res_name not in res_names:
                    res_names.append(res_name)
                    res_idx+=1
                res[res_idx-1].append(atom_idx)
                atom_type.append(data[5])
                atom_label.append(data[1])
                atom_charge.append(float(data[8]))
            elif res_name.startswith("Q6Z0"):
                atom_idx = int(data[0])-1-len(mm_coords)
                ligand_atom_idx.append(data[0])
                ligand_geo.append((data[1][0],(float(data[2]),float(data[3]),float(data[4]))))
                atom_charge.append(float(data[8]))
            elif res_name.startswith("HOH"):
                mm_coords.append((float(data[2]),float(data[3]),float(data[4])))
                mm_charge.append(float(data[8]))
            elif res_name.startswith("NA"):
                mm_coords.append((float(data[2]),float(data[3]),float(data[4])))
                mm_charge.append(float(data[8]))
            elif res_name.startswith("DMS"):
                mm_coords.append((float(data[2]),float(data[3]),float(data[4])))
                mm_charge.append(float(data[8]))
            else:
                raise ValueError("residue not regognized")
        res_charge = [0]*len(res)
        for i in range(len(res)):
            for idx in res[i]:
                res_charge[i]+=atom_charge[idx]
        bonds=[]
        bond_types=[]
        assert (res_atom_idx[0] == min(res_atom_idx))
        assert (ligand_atom_idx[0] == min(ligand_atom_idx))
        for line in lines[bond_idx+1:struct_idx]:
            data = line.split()
            if (data[1] not in res_atom_idx+ligand_atom_idx) or (data[2] not in res_atom_idx+ligand_atom_idx):
                continue
            if data[1] in res_atom_idx:
                bond_idx_1 = int(data[1]) - int(res_atom_idx[0])
            if data[1] in ligand_atom_idx:
                bond_idx_1 = int(data[1]) - int(ligand_atom_idx[0])+len(res_atom_idx)
            if data[2] in res_atom_idx:
                bond_idx_2 = int(data[2]) - int(res_atom_idx[0])
            if data[2] in ligand_atom_idx:
                bond_idx_2 = int(data[2]) - int(ligand_atom_idx[0])+len(res_atom_idx)
            bond = (bond_idx_1,bond_idx_2)
            bonds.append(bond)
            bond_types.append(data[3])
        length = len(res)
        idx= 0
        while idx < length:
            if res[idx] ==[]:
                del res[idx]
                idx-=1
                length-=1
            idx+=1
        assert(len(res_charge)==len(res))
        return res,res_names, geometry, bonds, bond_types,atom_type,atom_label,atom_charge,res_charge,ligand_geo,ligand_atom_charge,ligand_atom_idx,mm_coords,mm_charge

    def divide_residues(self):
        N_side = []
        core_carbon = []
        C_side = []
        for residue in self._res:
            N_idx = residue[0]
            if self._atom_type[N_idx].startswith("O"):
                continue
            assert(self._atom_type[N_idx] in ["N.am","N.3","N.4"])
            N_H_idx = []
            N_H_idx.append(N_idx)
            for atom_idx in residue:
                if self._atom_label[atom_idx] =="CA":
                    core_carbon_idx = atom_idx
                if ((N_idx,atom_idx) in self._bonds) or ((atom_idx,N_idx) in self._bonds):
                    if self._atom_type[atom_idx]=="H":
                        N_H_idx.append(atom_idx)
            N_side.append(N_H_idx)

            COO_idx = []
            for atom_idx in residue:
                if ((core_carbon_idx,atom_idx) in self._bonds) or ((atom_idx,core_carbon_idx) in self._bonds ):
                    if self._atom_type[atom_idx]=="H":
                        core_C_H_idx = atom_idx
                    elif self._atom_type[atom_idx] =="C.2":
                        COO_C_idx = atom_idx
                        COO_idx.append(COO_C_idx)
            core_carbon.append([core_carbon_idx,core_C_H_idx])

            for atom_idx in residue:
                if ((COO_C_idx,atom_idx) in self._bonds) or ((atom_idx,COO_C_idx) in self._bonds):
                    if self._atom_type[atom_idx].startswith("O"):
                        COO_idx.append(atom_idx)
            C_side.append(COO_idx)

        assert(len(C_side)==len(N_side)==len(core_carbon)==len(self._res))
        R_group = copy.deepcopy(self._res)
        for i in range(len(C_side)):
            for idx in C_side[i]:
                R_group[i].remove(idx)
            for idx in core_carbon[i]:
                R_group[i].remove(idx)
            for idx in N_side[i]:
                R_group[i].remove(idx)

        return C_side, N_side, core_carbon, R_group

    def get_fragment_for_residues(self):
        fragment = [[]for i in range(len(self._core_carbon))]
        for i in range(len(self._N_side)):
            N_idx = self._N_side[i][0]
            assert (self._atom_type[N_idx] in ["N.4","N.3","N.am"])
            if self._atom_type[N_idx] in ["N.4","N.3"]:  ##at the starting position
                fragment[i]=fragment[i]+self._N_side[i]
            elif self._atom_type[N_idx] == "N.am":
                fragment[i-1] = fragment[i-1]+self._N_side[i]
            fragment[i] = fragment[i] + self._core_carbon[i]
            fragment[i] = fragment[i] + self._C_side[i]
        for i in range(len(self._R_group)):
            if len(self._R_group[i]) < 5:
                fragment[i]=fragment[i]+self._R_group[i]
            else:
                fragment.append(self._R_group[i])
        return fragment

    def add_fragment_for_ligand(self):
        atom_list=[
                ['0', '1', '2', '3', '4', '5', '7', '8', '9', '14', '15'],
                ['10', '11', '12', '13'],
                ['16', '17', '18', '19', '20', '21', '22', '23', '24', '40'],
                ['6', '25', '26', '27', '28', '34', '41', '42', '37', '38', '39'],
                ['29', '30', '31', '32', '33', '35', '36']]
        for i in range(len(atom_list)):
            for j in range(len(atom_list[i])):
                atom_list[i][j] =int(atom_list[i][j]) +len(self._atom_label)
        #print(atom_list)
        self.fragment =self.fragment+atom_list

    def get_fragment_charge(self):
        frag_charge = []
        for frag in self.fragment:
            charge = 0
            for idx in frag:
                charge = charge+self._atom_charge[idx]
                #print(len(self._atom_charge))
                #print(idx)
            charge = round(charge)
            frag_charge.append(charge)
        assert(len(frag_charge)==len(self.fragment))
        return frag_charge
    def get_qm_atom(self):
        '''
        '''
        if not (self.natom_per_fragment == None):
            natom_per_fragment = self.natom_per_fragment
            self._nfrag = len(self._geometry)//natom_per_fragment
            for i in range(self._nfrag):
                frag = np.arange(natom_per_fragment*i,natom_per_fragment*i+natom_per_fragment)
                #print(get_distance(geometry[frag[0]][1],geometry[frag[1]][1]))
                #print(get_distance(geometry[frag[0]][1],geometry[frag[2]][1]))
                #assert(get_distance(self._geometry[frag[0]][1],self._geometry[frag[1]][1])<1.0)
                #assert(get_distance(self._geometry[frag[0]][1],self._geometry[frag[2]][1])<1.0)
                self._qm_list.append(list(frag))
        elif not (self.atom_list==None):
            self._qm_list = self.atom_list
        else:
            #print(self._qm_list)
            assert(self._qm_list ==None)
            print('atom list or natom_per_fragment not given')
            exit()
    def get_mm_atom(self):
        '''
        '''
        list_all = list(range(self._nfrag))
        for i in range(self._nfrag):
            self._mm_list.append([x for x in list_all if x not in self._qm_list[i]])

    def get_connection(self):
        natom = len(self.geo)
        connection = [[] for i in range(natom)]
        for bond in self._bonds:
            if (bond[0]>natom-1) or (bond[1]>natom-1):
                continue
            connection[bond[0]].append(bond[1])
            connection[bond[1]].append(bond[0])
        self.connection = connection
        return connection
    def output(self,output_file_name="out.frg"):
        output_file = open(output_file_name,"w+")
        for i in range(len(self.fragment)):
            string = str(i+1)+" 1 "
            frag_tmp = []
            for atom in self.fragment[i]:
                if self.geo[atom][0] != 'H':
                    frag_tmp.append(atom)
            string+=str(tuple(frag_tmp))
            charge = self.frag_charges[i]
            string+=" "
            if charge >0:
                string += "+"
            string += str(charge)
            string += "\n"
            output_file.write(string)
        output_file.close()
        return