'''
Many Body Expansion 
'''
import numpy as np
import os
from itertools import combinations
from pyscf import gto, scf
from scf_from_pyscf import pyscf_interface
from fermion_operator import FermionOps
from set_options import set_options
import time
#from mbe_solver import pyscf_uhf
#class combinations_new(combinations):

class MBE():
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
        #print(self.fragment_center)
        #exit()
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
    def get_distance(self,coord1,coord2):
        d2=(coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2+(coord2[2]-coord1[2])**2
        d= np.sqrt(d2)
        return d
    def fractorial(self,n : int):
        if n==1 or n==0:
            return 1
        else:
            return n*self.fractorial(n-1)

    def get_mbe_energy_approx( self,
                        n:int, # MBE order
                        ):
        calculated_mbe_order = len(self.mbe_series)
        for i in np.arange(calculated_mbe_order,n):
            print('Calculating the order %d mbe energy'%(i))
            self.get_n_th_order_mbe(n=i)

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


    def get_n_th_order_mbe( self,
                            n:int, #MBE order
                            ):
        if n==0:
            mbe_energy_0=0
            for i in range(len(self.fragment)):
                mbe_energy_0+=self.get_energy(self.fragment[i])
            self.mbe_series.append(mbe_energy_0)
        elif n>len(self.fragment):
            print('exceeds highest order')
            exit()
        elif n>len(self.mbe_series):
            print('lower order energy needed')
            exit()
        else:
            num_fragment = len(self.fragment)
            mbe_energy=0
            for atom_list in combinations(self.fragment,n+1):
                print(atom_list)
                atom_list_re = []
                for frag in atom_list:
                    for atom_idx in frag:
                        atom_list_re.append(atom_idx)
                print(atom_list_re)
                mbe_energy += self.get_energy(atom_list_re)
            for i in range(n):
                print(self.fractorial(num_fragment-i-1))
                mbe_energy -= self.mbe_series[i]*self.fractorial(num_fragment-i-1)/(self.fractorial(num_fragment-n-1)*self.fractorial(n-i))
                print(mbe_energy)
            self.mbe_series.append(mbe_energy)
    
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
        #if self.solver == 'water_hexamer':
        #    self.pool = get_ops_pool(self.geometry,self.fragment[0])
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

    def get_circ(self,args):
        atom_list = args[0]
        fragment_charge = args[1]
        save_directory = args[2]
        from mbe_solver import get_circ_protein
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
            from mbe_solver import pyscf_uhf
            energy = pyscf_uhf( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='rhf':
            from mbe_solver import pyscf_rhf
            energy = pyscf_rhf( geometry=self.geometry,
                                atom_list=atom_list,
                                fragment_charge = fragment_charge,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='dft':
            from mbe_solver import pyscf_dft
            energy = pyscf_dft( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='vqe':
            from mbe_solver import run_vqechem
            energy=run_vqechem( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'ccsd':
            from mbe_solver import pyscf_ccsd
            energy = pyscf_ccsd( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'mp2':
            from mbe_solver import pyscf_mp2
            energy = pyscf_mp2( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'dmrg' or self.solver == 'chemps2' :
            from mbe_solver import chemps2
            energy = chemps2(geometry=self.geometry,
                            atom_list=atom_list,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'water_hexamer':
            from mbe_solver import run_vqechem_water_hexamer
            energy = run_vqechem_water_hexamer(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'water_hexamer_sci':
            from mbe_solver import run_sci_water_hexamer
            energy = run_sci_water_hexamer(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18':
            from mbe_solver import run_vqechem_c18
            energy = run_vqechem_c18(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18_sci':
            from mbe_solver import run_vqechem_c18_sci
            energy = run_vqechem_c18_sci(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
        elif self.solver == 'vqe_c18_no_mp2':
            from mbe_solver import run_vqechem_c18_no_oo
            energy,dE1 = run_vqechem_c18_no_oo(geometry=self.geometry,
                            atom_list=atom_list,
                            qmmm_charges = self.qmmm_charges,
                            pool = self.pool,
                            basis = self.basis,
                            link_atom=self.link_atom)
            #print('solver complete, energy = ',energy,', delta E = ',dE1)
        elif self.solver=='bace_sci':
            from mbe_solver import run_sci_bace
            energy=run_sci_bace( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='bace_vqe':
            from mbe_solver import run_vqe_bace
            energy=run_vqe_bace( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='run_vqe_protein':
            from mbe_solver import run_vqe_protein
            energy=run_vqe_protein( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='get_circ_protein':
            from mbe_solver import get_circ_protein
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
            from mbe_solver import run_sci_protein
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
            from mbe_solver import fno_ccsd_sci
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
            from mbe_solver import fno_ccsd_circ
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
            from mbe_solver import hf_ccsd_circ
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
            from mbe_solver import hf_ccsd_sci
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
            from mbe_solver import run_vqechem_protein
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
            from mbe_solver import run_q2chem
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
            from mbe_solver import get_hamiltonian
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
                exec('from mbe_solver import '+self.solver)
            except ImportError :
                print('can not find solver, please check solver option')
                exit()
        end_time = time.perf_counter()
        time_used = end_time - start_time
        print("time used for calculation: ",time_used)
        print("final energy = ", energy)
        return energy

    def get_energy_qmmm(self,atom_list,fragment_charge):
        start_time = time.perf_counter()
        if self.solver=='uhf':
            from mbe_solver import pyscf_uhf_qmmm
            energy = pyscf_uhf_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='rhf':
            from mbe_solver import pyscf_rhf_qmmm
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
            from mbe_solver import pyscf_dft_qmmm
            energy = pyscf_dft_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='vqe':
            from mbe_solver import run_vqechem_qmmm
            energy=run_vqechem_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver == 'ccsd':
            from mbe_solver import pyscf_ccsd_qmmm
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
            from mbe_solver import pyscf_mp2_qmmm
            energy = pyscf_mp2_qmmm( geometry=self.geometry,
                                atom_list=atom_list,
                                qmmm_charges = self.qmmm_charges,
                                basis = self.basis,
                                link_atom=self.link_atom,
                                connection = self.connection)
        elif self.solver=='bace_sci':
            from mbe_solver import run_sci_bace_qmmm
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
            from mbe_solver import run_vqe_bace_qmmm
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
            from mbe_solver import get_circ_protein_qmmm
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

class MBE_substract(MBE):
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
        assert (len(self.fragment)==len(self.fragment_charge))
        if link_atom  == True:
            self.connection = fragment.connection
            assert (self.connection is not None)
            assert (len(self.connection)>0 )
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
        if len(mol.atom)>1000:
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
            if (self.qmmm_coords is not None) and (len(self.qmmm_coords)>0):
                if not os.path.exists(self.save_prefix+"qmmm_mbe_1/"):
                    os.mkdir(self.save_prefix+"qmmm_mbe_1/")
                    print("directory made: ",self.save_prefix+"qmmm_mbe_1/")



def get_ops_pool(geometry, atom_list):
    '''
    ''' 
    print('Generate operator pool first\n')
    print(atom_list)
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    mol.basis = 'cc-pvdz'
    mol.build()    

    if len(atom_list)==6:
        nacs = 8
        ncore = 4
        nvir = 2
        mo_list = range(ncore,ncore+nacs)
    else:
        nacs = 4
        ncore = 2
        nvir = 1
        mo_list = range(ncore,ncore+nacs)

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list},
               'ops' : {'class':'fermionic','spin_sym':'sa'},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    options = set_options(options)
    hf = pyscf_interface(mol, options)
    pool = FermionOps(hf, options['ops'])
    return pool

def geometry_h_ring(nat:int = 10,  #number of hydrogen atoms in the ring
                    bondlength : float=1.0
                    ):
    geometry = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        geometry.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))
    return geometry

def geometry_h_chain(nat:int = 10,  #number of hydrogen atoms in the ring
                    bondlength : float=1.0
                    ):
    geometry = []
    for i in range(nat):
        geometry.append(('H', (0, 0, i*bondlength)))
    return geometry

def geometry_Be_ring(nat:int = 30,  #number of atoms in the ring
                    bondlength : float=2.0
                    ):
    geometry = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        geometry.append(('Be', (r*np.cos(theta), r*np.sin(theta), 0)))
    return geometry

def geometry_carbon_ring(shift=20.0):
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

def test():
    natom = 10
    geometry = geometry_h_ring(natom)
    natom_per_fragment=2
    n_fragment = int(np.ceil(natom/natom_per_fragment))
    fragment = []
    for i in range(n_fragment):
        if (i+1)*natom_per_fragment <=natom:
            fragment.append(np.arange(i*natom_per_fragment,(i+1)*natom_per_fragment))
        else:
            fragment.append(np.arange(i*natom_per_fragment,natom))
    mbe = MBE(geometry,fragment,solver = 'uhf')
    return mbe
    mbe.get_n_th_order_mbe(0)
    print(mbe.mbe_series)
    
#if __name__=="__main__":
#    test()
