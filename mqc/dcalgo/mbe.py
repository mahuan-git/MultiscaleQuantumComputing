'''
Many Body Expansion 
'''
import os
import time
import numpy as np
from itertools import combinations
from pyscf import gto, scf
from mqc.system.fragment import Fragment

class MBE_base(object):
    def __init__(   self, 
                    fragment:Fragment,
                    solver : str = "pyscf_rhf",
                    isTI: bool = False,
                    link_atom: str = "extend"
                    ):
        assert link_atom in ["extend","origin",None], "link atom mode not supported"
        assert solver in ["pyscf_uhf","pyscf_rhf"] , "solver not supported"
        self.fragment = fragment
        self.qm_fragment = fragment.qm_fragment
        self.num_frag = len(self.qm_fragment)
        self.solver = solver.lower()

        self.mbe_1 = None
        self.mbe_2 = None
        self.mbe_3 = None
        self.isTI = isTI
        self.link_atom = link_atom

        self.mbe_energy = None

    def get_mbe_1(self):
        self.mbe_1=[]
        if self.isTI == True:
            energy_1 = self.get_energy(self.qm_fragment[0])
            for i in range(self.num_frag):
                self.mbe_1.append(energy_1)
        else:
            for i in range(self.num_frag):
                self.mbe_1.append(self.get_energy(self.qm_fragment[i]))
    
    def get_mbe_2(self):
        self.mbe_2=[]
        if self.isTI == True:
            mbe_2_tmp=[]
            for i in np.arange(1,self.num_frag):
                atom_list = []
                for atom_idx in self.qm_fragment[0]:
                    atom_list.append(atom_idx)
                for atom_idx in self.qm_fragment[i]:
                    atom_list.append(atom_idx)
                mbe_2_tmp.append(self.get_energy(atom_list))
            for i in range(self.num_frag-1):
                for j in np.arange(i,self.num_frag-1):
                    self.mbe_2.append(mbe_2_tmp[j])
        else:
            for atom_list in combinations(self.qm_fragment,2):
                atom_list_re = []
                for frag in atom_list:
                    for atom_idx in frag:
                        atom_list_re.append(atom_idx)
                self.mbe_2.append(self.get_energy(atom_list_re))
    
    def get_mbe_3(self):
        self.mbe_3=[]
        for atom_list in combinations(self.qm_fragment,3):
            atom_list_re = []
            for frag in atom_list:
                for atom_idx in frag:
                    atom_list_re.append(atom_idx)
            self.mbe_3.append(self.get_energy(atom_list_re))

    def get_mbe_energy( self,  order :int =2 ):
        from scipy.special import comb
        if order >3:
            raise ValueError('mbe order larger than 3 not supported')
        for i in np.arange(order,0,-1):
            if getattr(self,"mbe_"+str(i)) is None:
                getattr(self,"get_mbe_"+str(i))()
        def _coeficient(n,o,x):
            return comb(n-x-1,o-x)*((-1)**(o-x))
        self.mbe_energy = 0
        for i in np.arange(order,0,-1):
            self.mbe_energy += sum(getattr(self,"mbe_"+str(i)))*_coeficient(self.num_frag,order,i)
        return(self.mbe_energy)        
        
    def get_energy(self,atom_list):
        from mqc.solver import mbe_solver
        energy = getattr(mbe_solver,self.solver)(fragment = self.fragment,
                                                 atom_list = atom_list, 
                                                 link_atom = self.link_atom, 
                                                 )
        return energy

class MBE_qmmm(MBE_base):
    def __init__(self, fragment: Fragment, 
                 solver: str = "pyscf_rhf", 
                 isTI: bool = False, 
                 link_atom: str = "extend"):
        super().__init__(fragment, solver, isTI, link_atom)
        self.structure = self.fragment.structure
        self.mm_charges = self.structure.mm_charges
        self.mm_coords = self.structure.mm_coords
        self.mbe_1_qmmm

    def get_qmmm_corr(self):
        if self.mbe_1==None or self.mbe_1 ==[]:
            self.get_mbe_1()
        self.mbe_1_qmmm=[]
        if self.isTI == True:
            energy_1 = self.get_energy_qmmm(self.fragment.qm_fragment[0])
            for i in range(len(self.fragment)):
                self.mbe_1_qmmm.append(energy_1)
        else:
            for i in range(len(self.fragment)):
                self.mbe_1_qmmm.append(self.get_energy_qmmm(self.fragment.qm_fragment[i]))
        E_qmmm_coor = sum(self.mbe_1_qmmm)-sum(self.mbe_1)
        return E_qmmm_coor

    def get_energy_qmmm(self,atom_list):
        from mqc.solver import mbe_solver_qmmm
        energy = getattr(mbe_solver_qmmm,self.solver+"_qmmm")(fragment = self.fragment,
                                                              atom_list = atom_list, 
                                                              link_atom = self.link_atom, 
                                                                )
        return energy

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

