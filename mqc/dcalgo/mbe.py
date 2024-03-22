'''
Many Body Expansion 
'''
import os
import numpy as np
from itertools import combinations
from mqc.system.fragment import Fragment
from mqc.tools.tools import get_distance
from .option import mbe_option

class MBE_base(object):
    def __init__(   self, 
                    fragment:Fragment,
                    mbe_option: mbe_option
                    ):
        self.fragment = fragment
        self.mbe_option =mbe_option
        self.structure = self.fragment.structure
        self.qm_fragment = fragment.qm_fragment
        self.num_frag = len(self.qm_fragment)
        self.solver = mbe_option.solver.lower()

        self.mbe_1 = None
        self.mbe_2 = None
        self.mbe_3 = None
        self.isTI = mbe_option.isTI
        self.link_atom = mbe_option.link_atom

        self.mbe_option.qmmm_charges = self.structure.mm_charges
        self.mbe_option.qmmm_coords = self.structure.mm_coords

        self.mbe_energy = None
        
        if self.mbe_option.diag == "get_ham":
            assert self.mbe_option.save_root is not None, "save root should be given when choose to save hamiltonians."
            if self.mbe_option.save_root[-1] != '/':
                self.mbe_option.save_root += '/'
            if not os.path.exists(self.mbe_option.save_root):
                os.mkdir(self.mbe_option.save_root)
                print("directory made:",self.mbe_option.save_root)
            if not os.path.exists(self.mbe_option.save_root+"mbe_1/"):
                os.mkdir(self.mbe_option.save_root+"mbe_1/")
                print("directory made:",self.mbe_option.save_root+"mbe_1/")
            if not os.path.exists(self.mbe_option.save_root+"mbe_2/"):
                os.mkdir(self.mbe_option.save_root+"mbe_2/")
                print("directory made:",self.mbe_option.save_root+"mbe_2/")
            if not os.path.exists(self.mbe_option.save_root+"mbe_2_info/"):
                os.mkdir(self.mbe_option.save_root+"mbe_2_info/")
                print("directory made:",self.mbe_option.save_root+"mbe_2_info/")
            if (self.mbe_option.qmmm_coords is not None) and (len(self.mbe_option.qmmm_coords)>0):
                if not os.path.exists(self.mbe_option.save_root+"qmmm_mbe_1/"):
                    os.mkdir(self.mbe_option.save_root+"qmmm_mbe_1/")
                    print("directory made: ",self.mbe_option.save_root+"qmmm_mbe_1/")

    def get_mbe_1(self):
        self.mbe_option.ncas = self.mbe_option.ncas1
        self.mbe_1=[]
        if self.isTI == True:
            energy_1 = self.get_energy(self.qm_fragment[0])
            for i in range(self.num_frag):
                self.mbe_1.append(energy_1)
        else:
            for i in range(self.num_frag):
                self.mbe_1.append(self.get_energy(self.qm_fragment[i]))
    
    def get_mbe_2(self):
        self.mbe_option.ncas = self.mbe_option.ncas2
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
        self.mbe_option.ncas = self.mbe_option.ncas3
        self.mbe_3=[]
        for atom_list in combinations(self.qm_fragment,3):
            atom_list_re = []
            for frag in atom_list:
                for atom_idx in frag:
                    atom_list_re.append(atom_idx)
            self.mbe_3.append(self.get_energy(atom_list_re))

    def get_mbe_energy( self,  order :int = 2 ):
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
        assert self.solver in ["pyscf_uhf","pyscf_rhf","pyscf_dft","pyscf_ccsd","pyscf_mp2","vqechem","vqe_oo"]
        energy = getattr(mbe_solver,self.solver)(fragment = self.fragment,
                                                 atom_list = atom_list,  
                                                 option = self.mbe_option 
                                                 )
        return energy
    
    def get_qmmm_corr(self):
        self.mbe_option.ncas = self.mbe_option.ncas1
        if self.mbe_1==None or self.mbe_1 ==[]:
            self.get_mbe_1()
        mbe_1_qmmm=[]
        if self.isTI == True:
            energy_1 = self.get_energy_qmmm(self.fragment.qm_fragment[0])
            for i in range(len(self.fragment)):
                mbe_1_qmmm.append(energy_1)
        else:
            for i in range(len(self.fragment)):
                mbe_1_qmmm.append(self.get_energy_qmmm(self.fragment.qm_fragment[i]))
        E_qmmm_coor = sum(mbe_1_qmmm)-sum(self.mbe_1)
        return E_qmmm_coor

    def get_energy_qmmm(self,atom_list):
        from mqc.solver import mbe_solver_qmmm
        energy = getattr(mbe_solver_qmmm,self.solver+"_qmmm")(fragment = self.fragment,
                                                              atom_list = atom_list, 
                                                              option = self.mbe_option 
                                                                )
        return energy


class MBE_protein(MBE_base):
    def __init__(self, fragment: Fragment, mbe_option: mbe_option):
        super().__init__(fragment, mbe_option)
        self.fragment_center = self.get_fragment_center()


    def get_fragment_center(self):
        frag_center = []
        for frag in self.qm_fragment:
            natom = len(frag)
            coord_x = 0.0
            coord_y = 0.0
            coord_z = 0.0
            for atom_idx in frag:
                coord_x += self.fragment.qm_geometry[atom_idx][1][0]
                coord_y += self.fragment.qm_geometry[atom_idx][1][1]
                coord_z += self.fragment.qm_geometry[atom_idx][1][2]
            coord_x = coord_x/natom
            coord_y = coord_y/natom
            coord_z = coord_z/natom
            frag_center.append((coord_x,coord_y,coord_z))
        return frag_center

    def get_rhf_binding_energy(self,frag_idx_1,frag_idx_2):
        from mqc.solver.mbe_solver import pyscf_rhf

        def _get_rhf_energy(atom_list):
            energy = pyscf_rhf( fragment= self.fragment,
                                atom_list=atom_list,
                                link_atom=self.ink_atom)
            return energy
        
        frag1 = self.qm_fragment[frag_idx_1]
        frag2 = self.qm_fragment[frag_idx_2]
        frag_dimer = frag1+frag2

        energy_1 = _get_rhf_energy(frag1)
        energy_2 = _get_rhf_energy(frag2)
        energy_dimer = _get_rhf_energy(frag_dimer)
        binding_energy = energy_dimer - energy_1 - energy_2

        return binding_energy

    def check_mbe_2_skip(self,thres = 1e-6):
        idx = 0
        skip_count = 0
        skip_dict = []
        for frag_idx in combinations(range(len(self.fragment)),2):
            idx = idx + 1
            rhf_binding_energy = self.get_rhf_binding_energy(frag_idx[0],frag_idx[1])
            if abs(rhf_binding_energy) < thres:
                skip_count += 1
                skip_dict.append(idx)
        return

    def check_mbe_2_skip_by_fragment_center_dist(self,thres_dist = 10.0):
        idx = 0
        skip_count = 0
        skip_dict = []
        for frag_idx in combinations(range(len(self.fragment)),2):
            idx = idx + 1
            frag_center_dist = get_distance(self.fragment_center[frag_idx[0]],self.fragment_center[frag_idx[1]])
            if (frag_center_dist > thres_dist):
                skip_count += 1
                skip_dict.append(idx)
                #print("mbe_2 index: ",idx)
                #print("fragment index: ",frag_idx[0],"  ",frag_idx[1])
        return
    
    def get_mbe_2(self,thres_dist = 15.0,ncas_occ = 4,ncas_vir = 4):
        self.mbe_2=[]
        if self.isTI == True:
            mbe_2_tmp=[]
            for i in np.arange(1,len(self.fragment)):
                atom_list = []
                for atom_idx in self.fragment[0]:
                    atom_list.append(atom_idx)
                for atom_idx in self.fragment[i]:
                    atom_list.append(atom_idx)
                fragment_charge = self.fragment_charge[0]+self.fragment_charge[i]
                if self.mbe_option.save_root is not None:
                    save_directory = self.mbe_option.save_root + "mbe_2/"+"params_"+str(i)
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
                        if self.mbe_option.save_root is not None:
                            save_directory = self.mbe_option.save_root +"mbe_2/"+"params_"+str(idx)
                        else:
                            save_directory = None
                        self.mbe_2.append(self.get_energy(atom_list_re,fragment_charge,save_directory = save_directory,ncas_occ = ncas_occ,ncas_vir = ncas_vir))
                    else:
                        pass
            print(skip_dict)
