import numpy as np
import copy



def get_circ_protein(geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None,
                save_directory = None,
                ncas_occ = 4,
                ncas_vir = 4):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if fragment_charge is not None:
        mol.charge = fragment_charge
    mol.basis = basis
    mol.build()
    assert (mol.nelectron%2 == 0)
    nocc = mol.nelectron//2
    ncas_occ = ncas_occ
    ncas_vir = ncas_vir
    ncas = ncas_occ+ncas_vir
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+ncas)
    mm_coords = None
    mm_charges = None
    if qmmm_charges is not None:
        mm_coords = []
        mm_charges = []
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':ncas,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'get_circ'},
               'file': {'save_directory':save_directory}
              }
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


class Fragment_protein_old():
    
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
        self._ligand_geo = ligand_geo
        self._ligand_charge = ligand_atom_charge
        self._ligand_atom_idx = ligand_atom_idx
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


class MBE(object):
    def __init__(   self,
                    geometry: list,
                    fragment,   # information about atom indices every fragment holds
                    solver  : str = 'rhf',
                    basis : str = 'sto_3g',
                    periodic: bool = False,
                    link_atom :bool = False,   #used when bond break and need to saturate bonds
                    qmmm_coords = None,
                    qmmm_charge_list = None,
                    save_prefix = None
                    ):
        self.geometry = geometry
        self.fragment = fragment._qm_list
        self.fragment_charge = fragment.frag_charges
        self.fragment_center = self.get_fragment_center()

        assert (len(self.fragment)==len(self.fragment_charge)==len(self.fragment_center))
        if link_atom  == True:
            self.connection = fragment.connection
            assert (self.connection is not None)
            assert (len(self.connection)>0 )
        else:
            self.connection = None
        self.qmmm_charges = fragment._mm_charges
        self.solver = solver.lower()
        self.mbe_series = []
        self.mbe_energy_1=None
        self.mbe_energy_2=None
        self.mbe_1=None
        self.mbe_2=None
        self.mbe_3=None
        self.mbe_4=None
        self.mbe_5=None
        self.mbe_1_qmmm = None
        self.basis = basis
        self.periodic = periodic
        self.link_atom=link_atom
        self.save_prefix = save_prefix
        self.mbe_2_fragment_idx = None
        #self.qmmm_option = qmmm_option
        if qmmm_coords is not None:
            self.qmmm_coords = qmmm_coords
        else:
            self.qmmm_coords = fragment._mm_coords
        if qmmm_charge_list is not None:
            self.qmmm_charge_list = qmmm_charge_list
        else:
            self.qmmm_charge_list = fragment._mm_charge
        mol=gto.Mole()
        mol.atom=self.geometry
        if mol.nelectron%2==0:
            mol.charge=0
        else:
            mol.charge=1
        print('charge of the molecule:', mol.charge)
        mol.spin = 0
        mol.basis = self.basis
        mol.build()
        if len(mol.atom)>100:
            print("too many atoms, skip scf process")
            self.mf = None
        else:
            mf = scf.RHF(mol)
            mf.verbose = 3
            mf.max_cycle = 1000
            mf.scf(dm0=None)
            self.mf = mf
        self.pool = None
        ##prepare folders
        if save_prefix is not None:
            if not os.path.exists(self.save_prefix):
                os.mkdir(self.save_prefix)
                print("directory made:",self.save_prefix)
            if not os.path.exists(self.save_prefix+"mbe_1/"):
                os.mkdir(self.save_prefix+"mbe_1/")
                print("directory made:",self.save_prefix+"mbe_1/")
            if not os.path.exists(self.save_prefix+"mbe_2/"):
                os.mkdir(self.save_prefix+"mbe_2/")
                print("directory made:",self.save_prefix+"mbe_2/")
            if not os.path.exists(self.save_prefix+"mbe_2_info/"):
                os.mkdir(self.save_prefix+"mbe_2_info/")
                print("directory made:",self.save_prefix+"mbe_2_info/")
            if (self.qmmm_coords is not None) and (len(self.qmmm_coords)>0):
                if not os.path.exists(self.save_prefix+"qmmm_mbe_1/"):
                    os.mkdir(self.save_prefix+"qmmm_mbe_1/")
                    print("directory made: ",self.save_prefix+"qmmm_mbe_1/")
    def get_fragment_center(self):
        frag_center = []
        for frag in self.fragment:
            natom = len(frag)
            coord_x = 0.0
            coord_y = 0.0
            coord_z = 0.0
            for atom_idx in frag:
                coord_x += self.geometry[atom_idx][1][0]
                coord_y += self.geometry[atom_idx][1][1]
                coord_z += self.geometry[atom_idx][1][2]
            coord_x = coord_x/natom
            coord_y = coord_y/natom
            coord_z = coord_z/natom
            frag_center.append((coord_x,coord_y,coord_z))
        return frag_center

    def get_mbe_energy( self,
                        order :int =2
                        ):
        self.mbe_series=[]
        if order >5:
            print('err: order larger than 5 not supported')
        elif order ==1:
            if self.mbe_1==None:
                self.get_mbe_1()
            #self.mbe_series.append(sum(self.mbe_1))
            mbe_energy=sum(self.mbe_1)
        elif order ==2:
            if self.mbe_1==None:
                self.get_mbe_1()
            if self.mbe_2==None:
                self.get_mbe_2()
            #self.mbe_series.append(sum(self.mbe_2)-(len(self.fragment)-2)*sum(self.mbe_1))
            mbe_energy=sum(self.mbe_2)-(len(self.fragment)-2)*sum(self.mbe_1)
        elif order ==3:
            if self.mbe_1==None:
                self.get_mbe_1()
            if self.mbe_2==None:
                self.get_mbe_2()
            if self.mbe_3==None:
                self.get_mbe_3()
            #self.mbe_series.append(sum(self.mbe_2)-(len(self.fragment)-2)*sum(self.mbe_1))
            mbe_energy=sum(self.mbe_3)-(len(self.fragment)-3)*sum(self.mbe_2)\
                        +0.5*(len(self.fragment)-2)*(len(self.fragment)-3)*sum(self.mbe_1)
        elif order ==4:
            if self.mbe_1==None:
                self.get_mbe_1()
            if self.mbe_2==None:
                self.get_mbe_2()
            if self.mbe_3==None:
                self.get_mbe_3()
            if self.mbe_4==None:
                self.get_mbe_4()
            #self.mbe_series.append(sum(self.mbe_2)-(len(self.fragment)-2)*sum(self.mbe_1))
            mbe_energy=sum(self.mbe_4)-(len(self.fragment)-4)*sum(self.mbe_3)\
                        +0.5*(len(self.fragment)-3)*(len(self.fragment)-4)*sum(self.mbe_2)\
                            -(1/6)*(len(self.fragment)-2)*(len(self.fragment)-3)*(len(self.fragment)-4)*sum(self.mbe_1)
        elif order ==5:
            if self.mbe_1==None:
                self.get_mbe_1()
            if self.mbe_2==None:
                self.get_mbe_2()
            if self.mbe_3==None:
                self.get_mbe_3()
            if self.mbe_4==None:
                self.get_mbe_4()
            if self.mbe_5==None:
                self.get_mbe_5()
            #self.mbe_series.append(sum(self.mbe_2)-(len(self.fragment)-2)*sum(self.mbe_1))
            mbe_energy=sum(self.mbe_5)-(len(self.fragment)-5)*sum(self.mbe_4)\
                        +0.5*(len(self.fragment)-4)*(len(self.fragment)-5)*sum(self.mbe_3)\
                            -(1/6)*(len(self.fragment)-3)*(len(self.fragment)-4)*(len(self.fragment)-5)*sum(self.mbe_2)\
                                +(1/24)*(len(self.fragment)-2)*(len(self.fragment)-3)*(len(self.fragment)-4)*(len(self.fragment)-5)*sum(self.mbe_1)
        self.mbe_energy = mbe_energy
        return(mbe_energy)

    
    def check_mbe_2_skip(self,thres = 1e-6):
        idx = 0
        skip_count = 0
        skip_dict = []
        for frag_idx in combinations(range(len(self.fragment)),2):
            idx = idx + 1
            ##pre-selection
            if True:
                rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
            else:
                rhf_binding_energy = 1.0
            if abs(rhf_binding_energy) < thres:
                skip_count += 1
                skip_dict.append(idx)
                print("%d dimer calculations skiped"%(skip_count))
                print("mbe_2 index: ",idx)
            else:
                if False:
                    print('  %d dimer calculation\n' %idx)
                    atom_list_re = self.fragment[frag_idx[0]]+self.fragment[frag_idx[1]]
                    fragment_charge = self.fragment_charge[frag_idx[0]]+self.fragment_charge[frag_idx[1]]
                    save_directory = self.save_prefix +"mbe_2/"+"params_"+str(idx)
                    self.mbe_2.append(self.get_energy(atom_list_re,fragment_charge,save_directory = save_directory))
                else:
                    print('%d dimer included in calculation '%idx)
        print('terms skipped: ',len(skip_dict))
        return


    def check_mbe_2_skip_by_fragment_center_distance(self,thres_dist = 10.0):
        idx = 0
        skip_count = 0
        skip_dict = []
        for frag_idx in combinations(range(len(self.fragment)),2):
            idx = idx + 1
            ##pre-selection
            if True:
                frag_center_dist = self.get_distance(self.fragment_center[frag_idx[0]],self.fragment_center[frag_idx[1]])
            else:
                frag_center_dist = 1.0
            if (frag_center_dist > thres_dist) and (self.fragment_charge[frag_idx[0]] == 0.0) and (self.fragment_charge[frag_idx[1]]==0.0):
                skip_count += 1
                skip_dict.append(idx)
                print("%d dimer calculations skiped"%(skip_count))
                #print("mbe_2 index: ",idx)
                #print("fragment index: ",frag_idx[0],"  ",frag_idx[1])
                if False:
                    rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
                    print("fragment distance = ", frag_center_dist)
                    print("charge of the fragments are: ",self.fragment_charge[frag_idx[0]]," " ,self.fragment_charge[frag_idx[1]])
                    print("RHF binding energy: ", rhf_binding_energy)
            elif frag_center_dist > thres_dist+2.0 :
                skip_count+=1
                skip_dict.append(idx)
                print("%d dimer calculations skiped"%(skip_count))
                if False:
                    rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
                    print("fragment distance = ", frag_center_dist)
                    print("charge of the fragments are: ",self.fragment_charge[frag_idx[0]]," " ,self.fragment_charge[frag_idx[1]])
                    print("RHF binding energy: ", rhf_binding_energy)
            else:
                if False:
                    print(' %d dimer calculation' %idx)
                    atom_list_re = self.fragment[frag_idx[0]]+self.fragment[frag_idx[1]]
                    fragment_charge = self.fragment_charge[frag_idx[0]]+self.fragment_charge[frag_idx[1]]
                    save_directory = self.save_prefix +"mbe_2/"+"params_"+str(idx)
                    self.mbe_2.append(self.get_energy(atom_list_re,fragment_charge,save_directory = save_directory))
                else:
                    print("%d dimer terms included in calculation" %idx)
        print("Number of dimer terms skipped , ",len(skip_dict))

    def get_mbe_1(self,ncas_occ = 4,ncas_vir = 4):
        self.mbe_1=[]
        print("start mbe calculation for monomers")
        if self.periodic == True:
            if self.save_prefix is not None:
                save_directory = self.save_prefix+'mbe_1/'+'params_'+str(i)
            else:
                save_directory = None
            energy_1 = self.get_energy(self.fragment[0],self.fragment_charge[0],save_directory=self.save_directory,ncas_occ = ncas_occ,ncas_vir = ncas_vir)
            for i in range(len(self.fragment)):
                self.mbe_1.append(energy_1)
        else:
            for i in range(len(self.fragment)):
                if self.save_prefix is not None:
                    save_directory = self.save_prefix+'mbe_1/'+'params_'+str(i)
                else:
                    save_directory = None
                self.mbe_1.append(self.get_energy(self.fragment[i],self.fragment_charge[i],save_directory = save_directory,ncas_occ = ncas_occ,ncas_vir = ncas_vir))
                print("%d/%d mbe_1 calculation have finished"%(i,len(self.fragment)))

    def get_mbe_2(self,thres_dist = 15.0,ncas_occ = 4,ncas_vir = 4):
        self.mbe_2=[]
        #if self.solver == 'water_hexamer':
        #    print(self.fragment[0],self.fragment[1])
        #    self.pool = get_ops_pool(self.geometry,self.fragment[0]+self.fragment[1])
        if self.periodic == True:
            mbe_2_tmp=[]
            for i in np.arange(1,len(self.fragment)):
                atom_list = []
                for atom_idx in self.fragment[0]:
                    atom_list.append(atom_idx)
                for atom_idx in self.fragment[i]:
                    atom_list.append(atom_idx)
                fragment_charge = self.fragment_charge[0]+self.fragment_charge[i]
                if self.save_prefix is not None:
                    save_directory = self.save_prefix + "mbe_2/"+"params_"+str(i)
                else:
                    save_directory = None
                mbe_2_tmp.append(self.get_energy(atom_list,fragment_charge,save_directory = save_directory,ncas_occ = ncas_occ,ncas_vir = ncas_vir))
            for i in range(len(self.fragment)-1):
                for j in np.arange(i,len(self.fragment)-1):
                    self.mbe_2.append(mbe_2_tmp[j])
        else:
            idx = 0
            skip_count = 0
            skip_dict = []
            for frag_idx in combinations(range(len(self.fragment)),2):
                idx = idx + 1
                ##pre-selection
                if True:
                    frag_center_dist = self.get_distance(self.fragment_center[frag_idx[0]],self.fragment_center[frag_idx[1]])
                else:
                    frag_center_dist = 1.0
                if (frag_center_dist) > thres_dist and (self.fragment_charge[frag_idx[0]] == 0.0) and (self.fragment_charge[frag_idx[1]]==0.0):
                    skip_count += 1
                    skip_dict.append(idx)
                    print("%d dimer calculations skiped"%(skip_count))
                    #print("mbe_2 index: ",idx)
                elif frag_center_dist > thres_dist :
                    skip_count+=1 
                    skip_dict.append(idx)
                    print("%d dimer calculations skiped"%(skip_count))
                    #rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
                    #print("fragment distance = ", frag_center_dist)
                    #print("charge of the fragments are: ",self.fragment_charge[frag_idx[0]]," " ,self.fragment_charge[frag_idx[1]])
                    #print("RHF binding energy: ", rhf_binding_energy)
                else:
                    if True:
                        print('  %d dimer calculation\n' %idx)
                        atom_list_re = self.fragment[frag_idx[0]]+self.fragment[frag_idx[1]]
                        fragment_charge = self.fragment_charge[frag_idx[0]]+self.fragment_charge[frag_idx[1]]
                        if self.save_prefix is not None:
                            save_directory = self.save_prefix +"mbe_2/"+"params_"+str(idx)
                        else:
                            save_directory = None
                        self.mbe_2.append(self.get_energy(atom_list_re,fragment_charge,save_directory = save_directory,ncas_occ = ncas_occ,ncas_vir = ncas_vir))
                    else:
                        pass
            print(skip_dict)
        
    def get_rhf_binding_energy(self,frag_idx_1,frag_idx_2):
        from mbe_solver import pyscf_rhf
        geometry = self.geometry
        basis = self.basis
        qmmm_charges = self.qmmm_charges
        link_atom = self.link_atom
        connection = self.connection
        def _get_rhf_energy(geometry, atom_list ,fragment_charge,qmmm_charges, basis , link_atom,connection):
            energy = pyscf_rhf( geometry=geometry,
                            atom_list=atom_list,
                            fragment_charge = fragment_charge,
                            qmmm_charges = qmmm_charges,
                            basis = basis,
                            link_atom=link_atom,
                            connection = connection)
            return energy
        frag1 = self.fragment[frag_idx_1]
        frag2 = self.fragment[frag_idx_2]
        frag_dimer = frag1+frag2
        frag_charge_1 = self.fragment_charge[frag_idx_1]
        frag_charge_2 = self.fragment_charge[frag_idx_2]
        frag_charge_dimer = frag_charge_1 + frag_charge_2

        energy_1 = _get_rhf_energy(geometry,frag1,frag_charge_1,qmmm_charges,basis,link_atom,connection)
        energy_2 = _get_rhf_energy(geometry,frag2,frag_charge_2,qmmm_charges,basis,link_atom,connection)
        energy_dimer = _get_rhf_energy(geometry,frag_dimer,frag_charge_dimer,qmmm_charges,basis,link_atom,connection)
        binding_energy = energy_dimer - energy_1 - energy_2
        print("RHF binding energy: ",binding_energy)

        return binding_energy
    
    def get_mbe_3(self):
        self.mbe_3=[]
        for atom_list in combinations(self.fragment,3):
            #print(atom_list)
            atom_list_re = []
            for frag in atom_list:
                for atom_idx in frag:
                    atom_list_re.append(atom_idx)
            #print(atom_list_re)
            fragment_charge = 0
            self.mbe_3.append(self.get_energy(atom_list_re,fragment_charge,save_directory = None,ncas_occ = None,ncas_vir = None))

    def get_mbe_4(self):
        self.mbe_4=[]
        for atom_list in combinations(self.fragment,4):
            #print(atom_list)
            atom_list_re = []
            for frag in atom_list:
                for atom_idx in frag:
                    atom_list_re.append(atom_idx)
            #print(atom_list_re)
            self.mbe_4.append(self.get_energy(atom_list_re))
    def get_mbe_5(self):
        self.mbe_5=[]
        for atom_list in combinations(self.fragment,5):
            #print(atom_list)
            atom_list_re = []
            for frag in atom_list:
                for atom_idx in frag:
                    atom_list_re.append(atom_idx)
            #print(atom_list_re)
            self.mbe_5.append(self.get_energy(atom_list_re))
    def get_qmmm_corr(self):
        if self.mbe_energy_1==None:
            self.mbe_energy_1 =self.get_mbe_energy(1)
        self.mbe_1_qmmm=[]
        #if self.solver == 'water_hexamer':
        #    self.pool = get_ops_pool(self.geometry,self.fragment[0])
        if self.periodic == True:
            save_directory = self.save_prefix + "qmmm_mbe_1/params_0"
            energy_1 = self.get_energy_qmmm(self.fragment[0],self.fragment_charge[0],save_directory=save_directory)
            for i in range(len(self.fragment)):
                self.mbe_1_qmmm.append(energy_1)
        else:
            for i in range(len(self.fragment)):
                save_directory = self.save_prefix + "qmmm_mbe_1/params_"+str(i)
                self.mbe_1_qmmm.append(self.get_energy_qmmm(self.fragment[i],self.fragment_charge[i],save_directory = save_directory))
        E_qmmm_coor = sum(self.mbe_1_qmmm)-self.mbe_energy_1
        return E_qmmm_coor

    def get_mbe_2_fragment_idx(self,thres_dist = 10.0):
        mbe_2_fragment_idx=[]
        if self.periodic == True:
            mbe_2_tmp=[]
            for i in np.arange(1,len(self.fragment)):
                atom_list = []
                for atom_idx in self.fragment[0]:
                    atom_list.append(atom_idx)
                for atom_idx in self.fragment[i]:
                    atom_list.append(atom_idx)
                fragment_charge = self.fragment_charge[0]+self.fragment_charge[i]
                save_directory = self.save_prefix + "mbe_2/"+"params_"+str(i)
                mbe_2_tmp.append(self.get_energy(atom_list,fragment_charge,save_directory = save_directory))
            for i in range(len(self.fragment)-1):
                for j in np.arange(i,len(self.fragment)-1):
                    self.mbe_2.append(mbe_2_tmp[j])
        else:
            idx = 0
            skip_count = 0
            skip_dict = []
            for frag_idx in combinations(range(len(self.fragment)),2):
                idx = idx + 1
                ##pre-selection
                if True:
                    frag_center_dist = self.get_distance(self.fragment_center[frag_idx[0]],self.fragment_center[frag_idx[1]])
                else:
                    frag_center_dist = 1.0
                if (frag_center_dist) > thres_dist and (self.fragment_charge[frag_idx[0]] == 0.0) and (self.fragment_charge[frag_idx[1]]==0.0):
                    skip_count += 1
                    skip_dict.append(idx)
                    #print("%d dimer calculations skiped"%(skip_count))
                    #print("mbe_2 index: ",idx)
                elif frag_center_dist > thres_dist+2.0 :
                    skip_count+=1
                    skip_dict.append(idx)
                    #print("%d dimer calculations skiped"%(skip_count))
                    #rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
                    #print("fragment distance = ", frag_center_dist)
                    #print("charge of the fragments are: ",self.fragment_charge[frag_idx[0]]," " ,self.fragment_charge[frag_idx[1]])
                    #print("RHF binding energy: ", rhf_binding_energy)
                else:
                    #print('  %d dimer calculation\n' %idx)
                    #atom_list_re = self.fragment[frag_idx[0]]+self.fragment[frag_idx[1]]
                    #fragment_charge = self.fragment_charge[frag_idx[0]]+self.fragment_charge[frag_idx[1]]
                    #save_directory = self.save_prefix +"mbe_2/"+"params_"+str(idx)
                    #self.mbe_2.append(self.get_energy(atom_list_re,fragment_charge,save_directory = save_directory))
                    mbe_2_fragment_idx.append(frag_idx)
        self.mbe_2_fragment_idx = mbe_2_fragment_idx
        return

    def get_mbe_2_params_parallel(self,thres_dist = 10.0,start_idx = 0, end_idx = 100000):
        from multiprocessing import Pool
        if self.mbe_2_fragment_idx ==None:
            self.get_mbe_2_fragment_idx(thres_dist = thres_dist)
        n_works = len(self.mbe_2_fragment_idx)
        if end_idx >n_works:
            end_idx = n_works
        args = []
        idx = start_idx
        for i in np.arange(start_idx,end_idx):
            frag_idx = self.mbe_2_fragment_idx[i]
            idx+=1
            atom_list = self.fragment[frag_idx[0]]+self.fragment[frag_idx[1]]
            fragment_charge = self.fragment_charge[frag_idx[0]]+self.fragment_charge[frag_idx[1]]
            save_directory = self.save_prefix +"mbe_2/"+"params_"+str(idx)
            info_file_name = self.save_prefix+"mbe_2_info/"+"info_"+str(idx)
            infofile = open(info_file_name,"w+")
            infofile.write(str(frag_idx))
            infofile.close()
            args.append((atom_list,fragment_charge,save_directory))
        n_proc = 6
        pool = Pool(n_proc)
        map_result=pool.map(self.get_circ,args)
        pool.close()
        pool.join()
        return

    def get_circ(self,args):
        atom_list = args[0]
        fragment_charge = args[1]
        save_directory = args[2]
        from mqc.solver.mbe_solver import get_circ_protein
        energy=get_circ_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                ncas_occ = 3,
                                ncas_vir = 3
                                )
        return energy
 
    def get_energy(self,atom_list,fragment_charge,save_directory = None,ncas_occ = 4,ncas_vir = 4):
        start_time = time.perf_counter()
        if self.solver=='uhf':
            from mqc.solver.mbe_solver import pyscf_uhf
            energy = pyscf_uhf( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='rhf':
            from mqc.solver.mbe_solver import pyscf_rhf
            energy = pyscf_rhf( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='dft':
            from mqc.solver.mbe_solver import pyscf_dft
            energy = pyscf_dft( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='vqe':
            from mqc.solver.mbe_solver import run_vqechem
            energy=run_vqechem( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'ccsd':
            from mqc.solver.mbe_solver import pyscf_ccsd
            energy = pyscf_ccsd( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'mp2':
            from mqc.solver.mbe_solver import pyscf_mp2
            energy = pyscf_mp2( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'dmrg' or self.solver == 'chemps2' :
            from mqc.solver.mbe_solver import chemps2
            energy = chemps2(geometry=self.geometry,
                            atom_list=atom_list,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'water_hexamer':
            from mqc.solver.mbe_solver import run_vqechem_water_hexamer
            energy = run_vqechem_water_hexamer(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'water_hexamer_sci':
            from mqc.solver.mbe_solver import run_sci_water_hexamer
            energy = run_sci_water_hexamer(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18':
            from mqc.solver.mbe_solver import run_vqechem_c18
            energy = run_vqechem_c18(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18_sci':
            from mqc.solver.mbe_solver import run_vqechem_c18_sci
            energy = run_vqechem_c18_sci(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18_no_mp2':
            from mqc.solver.mbe_solver import run_vqechem_c18_no_oo
            energy,dE1 = run_vqechem_c18_no_oo(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
            #print('solver complete, energy = ',energy,', delta E = ',dE1)
        elif self.solver=='bace_sci':
            from mqc.solver.mbe_solver import run_sci_bace
            energy=run_sci_bace( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='bace_vqe':
            from mqc.solver.mbe_solver import run_vqe_bace
            energy=run_vqe_bace( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='run_vqe_protein':
            from mqc.solver.mbe_solver import run_vqe_protein
            energy=run_vqe_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='get_circ_protein':
            from mqc.solver.mbe_solver import get_circ_protein
            energy=get_circ_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                ncas_occ = ncas_occ,
                                ncas_vir = ncas_vir
                                )
        elif self.solver=='run_sci_protein':
            from mqc.solver.mbe_solver import run_sci_protein
            energy=run_sci_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory
                                )
        elif self.solver == 'fno_ccsd_sci':
            from mqc.solver.mbe_solver import fno_ccsd_sci
            energy=fno_ccsd_sci( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory
                                )
        elif self.solver == 'fno_ccsd_circ':
            from mqc.solver.mbe_solver import fno_ccsd_circ
            energy=fno_ccsd_circ( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory
                                )
        elif self.solver == 'hf_ccsd_circ':
            from mqc.solver.mbe_solver import hf_ccsd_circ
            energy=hf_ccsd_circ( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                ncas_occ = ncas_occ,
                                ncas_vir = ncas_vir
                                )
        elif self.solver == 'hf_ccsd_sci':
            from mqc.solver.mbe_solver import hf_ccsd_sci
            energy=hf_ccsd_sci( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory
                                )
        elif self.solver=='run_vqechem_protein':
            from mqc.solver.mbe_solver import run_vqechem_protein
            energy=run_vqechem_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                ncas_occ = ncas_occ,
                                ncas_vir = ncas_vir
                                )
        elif self.solver=='run_q2chem':
            from mqc.solver.mbe_solver import run_q2chem
            energy=run_q2chem( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                ncas_occ = ncas_occ,
                                ncas_vir = ncas_vir
                                )
        elif self.solver=='get_ham':
            from mqc.solver.mbe_solver import get_hamiltonian
            ham=get_hamiltonian( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = None,
                                ncas_occ = ncas_occ,
                                ncas_vir = ncas_vir
                                )
            return ham
        else:
            try:
                exec('from mqc.solver.mbe_solver import '+self.solver)
            except ImportError :
                print('can not find solver, please check solver option')
                exit()
        end_time = time.perf_counter()
        time_used = end_time - start_time
        print("time used for calculation: ",time_used)
        print("final energy = ", energy)
        return energy

    def get_energy_qmmm(self,atom_list,fragment_charge,save_directory):
        start_time = time.perf_counter()
        if self.solver=='uhf':
            from mqc.solver.mbe_solver import pyscf_uhf_qmmm
            energy = pyscf_uhf_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='rhf':
            from mqc.solver.mbe_solver import pyscf_rhf_qmmm
            energy = pyscf_rhf_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                qmmm_coords = self.qmmm_coords,
                                qmmm_charge_list = self.qmmm_charge_list
                                )
        elif self.solver=='dft':
            from mqc.solver.mbe_solver import pyscf_dft_qmmm
            energy = pyscf_dft_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='vqe':
            from mqc.solver.mbe_solver import run_vqechem_qmmm
            energy=run_vqechem_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'ccsd':
            from mqc.solver.mbe_solver import pyscf_ccsd_qmmm
            energy = pyscf_ccsd_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                qmmm_coords = self.qmmm_coords,
                                qmmm_charge_list = self.qmmm_charge_list
                                )
        elif self.solver == 'mp2':
            from mqc.solver.mbe_solver import pyscf_mp2_qmmm
            energy = pyscf_mp2_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='bace_sci':
            from mqc.solver.mbe_solver import run_sci_bace_qmmm
            energy=run_sci_bace_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                qmmm_coords = self.qmmm_coords,
                                qmmm_charge_list = self.qmmm_charge_list
                                )
        elif self.solver=='bace_vqe':
            from mqc.solver.mbe_solver import run_vqe_bace_qmmm
            energy=run_vqe_bace_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                qmmm_coords = self.qmmm_coords,
                                qmmm_charge_list = self.qmmm_charge_list
                                )

        elif self.solver=='get_circ_protein':
            from mqc.solver.mbe_solver import get_circ_protein_qmmm
            energy=get_circ_protein_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection,
                                save_directory = save_directory,
                                qmmm_coords = self.qmmm_coords,
                                qmmm_charge_list = self.qmmm_charge_list
                                )
        end_time = time.perf_counter()
        time_used = end_time - start_time
        print('energy with QMMM charges:',energy)
        print("time used for calculation:", time_used)
        return energy
