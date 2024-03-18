'''
solvers for mbe
'''
from pyscf import gto, scf, qmmm
from pyscf.cc import ccsd
from pyscf.mp.mp2 import MP2
from itertools import combinations
import numpy as np
import os
import sys

def get_distance(coordinate_1,coordinate_2):
    assert(len(coordinate_1)==len(coordinate_2)==3)
    distance = 0
    for i in range(len(coordinate_1)):
        distance +=(coordinate_1[i]-coordinate_2[i])**2
    distance = np.sqrt(distance)
    return distance

def get_connection(geometry):
    connection=[]
    natom = len(geometry)
    for i in range(natom):
        connection_tmp = []
        for j in range(natom):
            if i==j:
                dist = 2
            else:
                dist = get_distance(geometry[i][1],geometry[j][1])
            if dist < 1.75:
                connection_tmp.append(j)
        connection.append(connection_tmp)
    return connection

def get_link_atom_coordinate(   geometry : list,  
                                end_atoms : list,
                                add_atom_idx : int,
                                mode : str = 'extend',
                                ):
    '''
    get the coordinate of the added link atom
    geometry : the geometry of the original molecule
    end_atoms: the index of the two atom at the end of a fragment
    add_atom_idx : the atom index of the added atom.
    mode : two mode to add a link atom
    '''   
    natoms = len(geometry)    
    end_atoms = list(end_atoms)
    end_atoms.sort()     
    bondlength = 1.0
    link_atom_coordinate = [0,0,0]
    if mode == 'extend':
        coordinate_1 = geometry[end_atoms[0]%natoms][1]
        coordinate_2 = geometry[end_atoms[1]%natoms][1]
        distance = get_distance(coordinate_1,coordinate_2)
        delta_x = (coordinate_2[0]-coordinate_1[0])*bondlength/distance
        delta_y = (coordinate_2[1]-coordinate_1[1])*bondlength/distance
        delta_z = (coordinate_2[2]-coordinate_1[2])*bondlength/distance
        #print([delta_x,delta_y,delta_z])
        if add_atom_idx == (end_atoms[1]+1):
            link_atom_coordinate[0] = coordinate_2[0] + delta_x
            link_atom_coordinate[1] = coordinate_2[1] + delta_y
            link_atom_coordinate[2] = coordinate_2[2] + delta_z
        elif add_atom_idx == (end_atoms[0]-1):
            link_atom_coordinate[0] = coordinate_1[0] - delta_x
            link_atom_coordinate[1] = coordinate_1[1] - delta_y
            link_atom_coordinate[2] = coordinate_1[2] - delta_z
        else:
            print('add atom index does not match, check please')
            exit()

    elif mode =='origin':
        '''to be finished later
        '''
        exit()
        pass
    else:
        raise ValueError('mode to add the link atom not recognized, please check')
    
    return link_atom_coordinate


def add_link_atoms_extend( geometry,
                    atom_list,
                    mode='extend',
                    ):
    '''
    Do not use link_atom if there is only one atom in the fragment
    mode can either be extend or origin
    extend 为附加原子位于两原子的延长线上
    origin 为附加原子位于分子原有原子的位置附近
    '''
    natoms = len(geometry)
    link_atom_coordinate=[]
    for atoms in combinations(atom_list,2):
        if abs(atoms[1]-atoms[0])==1:   ## two selected atoms are neighbours
            atom_idx_1 = max(atoms)+1
            atom_idx_2 = min(atoms)-1
            if atom_idx_1%natoms not in atom_list:
                coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_1,mode = mode)
                link_atom_coordinate.append(coordinate)
            if atom_idx_2%natoms not in atom_list:
                coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_2,mode = mode)
                link_atom_coordinate.append(coordinate)

        elif abs(atoms[1]-atoms[0])==natoms-1:   ##the two selected atoms are at the head and the tail of the molecule respectively
            distance = get_distance(geometry[atoms[1]][1],geometry[atoms[0]][1])
            if distance >3.0:  ## indicate that the head and the tail are not connected
                pass
            else:          ## the head and the tail are connected. 
                atom_idx_1 = 1
                atom_idx_2 = -2
                atoms = [-1,0]
                if (atom_idx_1%natoms) not in atom_list:
                    coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_1,mode = mode)
                    link_atom_coordinate.append(coordinate)
                if (atom_idx_2%natoms) not in atom_list:
                    coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_2,mode = mode)
                    link_atom_coordinate.append(coordinate)
        else: ## the two atoms chosen are not connected
            pass
    return link_atom_coordinate

def add_link_atoms_origin(  geometry,
                            atom_list,
                            connection = None):
    bondlength = 1.0
    if connection == None:
        connection = get_connection(geometry)
    bondbreak = []
    for i in atom_list:
        for j in connection[i]:
            if not (j in atom_list):
                bondbreak.append((i,j))
    #print(bondbreak)
    link_atom_coordinates = []
    for bond in bondbreak :
        coordinate_1 = np.array(geometry[bond[0]][1])
        coordinate_2 = np.array(geometry[bond[1]][1])
        dist = get_distance(coordinate_1,coordinate_2)
        vec = (coordinate_2-coordinate_1)*bondlength/dist
        coordinate_new = coordinate_1+vec
        link_atom_coordinates.append(coordinate_new)
        #print(coordinate_1,coordinate_2,coordinate_new)
    return link_atom_coordinates

def add_link_atoms( geometry,
                    atom_list,
                    mode='origin',
                    connection = None
                    ):
    if mode == 'extend':
        link_atom_coordinate = add_link_atoms_extend(geometry,atom_list,mode = 'extend')
    elif mode =='origin':
        link_atom_coordinate = add_link_atoms_origin(geometry,atom_list,connection = connection)
    else:
        print('mode error')
        exit()
    return link_atom_coordinate

def pyscf_uhf(  geometry:list,
                atom_list:list, qmmm_charges:list=None,
                basis:str='sto-3g', 
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    mf = scf.UHF(mol)
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
    mf = qmmm.mm_charge(mf, mm_coords, mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    return mf.e_tot

def pyscf_rhf(  geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list=None,
                basis:str='sto-3g',
                link_atom :bool= False,
                connection = None):
    assert (connection is not None)
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
    #print(mol.atom)
    #print("charge for fragment:",fragment_charge)
    mol.basis = basis
    mol.build()
    assert (mol.nelectron%2==0)
    mf = scf.RHF(mol)
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    if mf.converged == False:
        print(atom_list)
        print(mol.atom)
    return mf.e_tot

def pyscf_rhf_qmmm( geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list = None,
                basis:str='sto-3g',
                link_atom : bool = False,
                connection = None,
                qmmm_coords = None,
                qmmm_charge_list = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection = connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if fragment_charge is not None:
        mol.charge = fragment_charge
    mol.spin = 0
    mol.basis = basis
    mol.build()

    EE_MB = (qmmm_charges is not None)
    if EE_MB:
        mm_coords = []
        mm_charges = []
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        #print(mol.atom)
        #print(mm_coords)
        #print(mm_charges)

    mf = scf.RHF(mol)
    mf = qmmm.mm_charge(mf, qmmm_coords, qmmm_charge_list)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    #ccsolver = ccsd.CCSD( mf )
    #ccsolver.verbose = 5
    #ECORR, t1, t2 = ccsolver.ccsd()
    #ERHF = mf.e_tot
    #ECCSD = ERHF + ECORR
    return mf.e_tot


def pyscf_dft(  geometry:list,
                atom_list:list, qmmm_charges:list=None,
                basis:str='cc-pvdz',
                link_atom :bool= False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    from pyscf import dft
    mf = dft.KS(mol,xc='HYB_GGA_XC_B3LYP')
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges)
    mf.verbose = 0
    mf.max_cycle = 1000
    mf.run()
    return mf.e_tot



def run_vqechem(geometry:list,
                atom_list:list, qmmm_charges:list,
                basis: str ='sto-3g',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    #if mol.nelectron%2==0:
    #    mol.spin=0
    #else:
    #    mol.spin=1
    print(mol.atom)
    assert(mol.nelectron%2 ==0)
    mol.basis = basis
    mol.build()
    import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    #from scf_from_pyscf import 
    import scipy
    import math 

    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':None,'ncore':None,'shift':0.5,
                        'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa'},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':1},
               'opt' : {'maxiter':300}
              }
    ansatz = run_vqe(mol,options)
    return ansatz._energy

def run_vqechem_bace(geometry:list,
                atom_list:list, qmmm_charges:list,
                basis: str ='sto-3g',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from VQEChem.algorithms import run_vqe
    #from scf_from_pyscf import 
    import scipy
    import math 

    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':None,'ncore':None,'shift':0.5,
                        'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa'},
               'ansatz' : {'method':'adapt','form':'taylor','Nt':10},
               'opt' : {'maxiter':300}
              }
    ansatz = run_vqe(mol,options)
    return ansatz._energy


def pyscf_ccsd( geometry:list,
                atom_list:list,
                qmmm_charges:list = None,
                basis:str='sto-3g',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection = connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if mol.nelectron%2==0:
        mol.charge=0
    elif mol.nelectron%2 ==1:
        mol.charge=1
    mol.spin = 0
    mol.basis = basis
    mol.build()

    EE_MB = (qmmm_charges is not None)
    if EE_MB:
        mm_coords = []
        mm_charges = []
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        #print(mol.atom)
        #print(mm_coords)
        #print(mm_charges)

    mf = scf.RHF(mol)
    if EE_MB:
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ERHF = mf.e_tot
    ECCSD = ERHF + ECORR
    print("CCSD energy: ", ECCSD)
    return ECCSD

def pyscf_ccsd_qmmm( geometry:list,
                atom_list:list,
                qmmm_charges:list = None,
                basis:str='sto-3g',
                link_atom : bool = False,
                connection = None,
                qmmm_coords = None,
                qmmm_charge_list = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection = connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if mol.nelectron%2==0:
        mol.charge=0
    elif mol.nelectron%2 ==1:
        mol.charge=1
    mol.spin = 0
    mol.basis = basis
    mol.build()

    EE_MB = (qmmm_charges is not None)
    if EE_MB:
        mm_coords = []
        mm_charges = []
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        #print(mol.atom)
        #print(mm_coords)
        #print(mm_charges)

    mf = scf.RHF(mol)
    mf = qmmm.mm_charge(mf, qmmm_coords, qmmm_charge_list)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ERHF = mf.e_tot
    ECCSD = ERHF + ECORR
    return ECCSD


def pyscf_mp2( geometry:list,
                atom_list:list,
                qmmm_charges:list = None,
                basis:str='sto-3g',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection = connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()

    EE_MB = (qmmm_charges is not None)
    if EE_MB:
        mm_coords = []
        mm_charges = []
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])
        #print(mol.atom)
        #print(mm_coords)
        #print(mm_charges)

    mf = scf.RHF(mol)
    if EE_MB:
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    mp2 = MP2( mf )
    mp2.verbose = 0
    #ECORR, t1, t2 = ccsolver.ccsd()
    mp2.run()
    #ERHF = mf.e_tot
    #ECCSD = ERHF + ECORR
    return mp2.e_tot

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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_sci_protein(geometry:list,
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'sci'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_vqechem_protein(geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None,
                save_directory = None,
                ncas_occ = 3,
                ncas_vir = 3):
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def hf_ccsd_sci(geometry:list,
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'ccsd','type':'hf','diag':'sci'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def hf_ccsd_circ(geometry:list,
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'ccsd','type':'hf','diag':'get_circ'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def fno_ccsd_sci(geometry:list,
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'ccsd','type':'fno','diag':'sci'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def fno_ccsd_circ(geometry:list,
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'ccsd','type':'fno','diag':'get_circ'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def get_circ_protein_qmmm(geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None,
                save_directory = None,
                qmmm_coords = None,
                qmmm_charge_list = None
                ):
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 4
    ncas_vir = 4
    ncas = ncas_occ+ncas_vir
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+ncas)
    #mm_coords = None
    #mm_charges = None
    #if qmmm_charges is not None:
    #    mm_coords = []
    #    mm_charges = []
    #    mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
    #    for i in mm_list:
    #        mm_coords.append(geometry[i][1])
    #        mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':ncas,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':qmmm_coords,
                        'qmmm_charges':qmmm_charge_list},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'get_circ'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1



def run_vqechem_water_hexamer(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math 
    nocc = mol.nelectron//2
    nacs = 8
    ncore = 6
    if len(atom_list)==6:
        if basis == 'cc-pvdz':
                nacs = nacs
                ncore = ncore
                #mo_list = range(6,14)
        else:
            print('basis not supported')
            exit()
    elif len(atom_list) == 3 :
        if basis == 'cc-pvdz':
            nacs = 4
            ncore = 3
            #mo_list = range(3,7)
        else:
            print('err')
            exit()
    nvir = ncore + nacs - nocc
    mo_list = np.arange(ncore,ncore+nacs)
    print(ncore,nacs,nvir,mo_list)
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_sci_water_hexamer(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math 
    nacs = 8
    ncore = 6
    if len(atom_list)==6:
        if basis == 'cc-pvdz':
                nacs = nacs
                ncore = ncore
                #mo_list = range(6,14)
        else:
            print('basis not supported')
            exit()
    elif len(atom_list) == 3 :
        if basis == 'cc-pvdz':
            nacs = 4
            ncore = 3
            #mo_list = range(3,7)
        else:
            print('err')
            exit()
    nvir = ncore + nacs - nocc
    mo_list = range(ncore,ncore+nacs)
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'sci'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E +dE1

def run_sci_bace(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.charge=0
    else:
        mol.charge=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 3
    ncas_vir = 3
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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'sci'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_sci_bace_qmmm(geometry:list,
                atom_list:list,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None,
                qmmm_coords = None,
                qmmm_charge_list = None):
    print('sci calculation with QMMM')
    #print(qmmm_coords)
    #print(qmmm_charges)
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.charge=0
    elif mol.nelectron%2 == 1:
        mol.charge=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 4
    ncas_vir = 4
    ncas = ncas_occ+ncas_vir
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+ncas)
    #mm_coords = []
    #mm_charges = []
    #if qmmm_charges is not None:
    #    mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
    #    for i in mm_list:
    #        mm_coords.append(geometry[i][1])
    #        mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':ncas,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':qmmm_coords,
                        'qmmm_charges':qmmm_charge_list},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'sci'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_vqe_bace(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.charge=0
    else:
        mol.charge=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 4
    ncas_vir = 4
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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def run_vqe_bace_qmmm(geometry:list,
                atom_list:list,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None,
                qmmm_coords = None,
                qmmm_charge_list = None):
    print('sci calculation with QMMM')
    #print(qmmm_coords)
    #print(qmmm_charges)
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.charge=0
    elif mol.nelectron%2 == 1:
        mol.charge=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 4
    ncas_vir = 4
    ncas = ncas_occ+ncas_vir
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+ncas)
    #mm_coords = []
    #mm_charges = []
    #if qmmm_charges is not None:
    #    mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
    #    for i in mm_list:
    #        mm_coords.append(geometry[i][1])
    #        mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':ncas,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':qmmm_coords,
                        'qmmm_charges':qmmm_charge_list},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1

def run_vqe_protein(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.charge=0
    else:
        mol.charge=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo 
    import scipy
    import math

    ncas_occ = 3
    ncas_vir = 3
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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


'''
def run_vqe_bace(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False,
                connection = None):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'origin',connection =connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))

    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    nocc = mol.nelectron//2
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

    ncas_occ = 4
    ncas_vir = 4
    ncas = ncas_occ+ncas_vir
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+ncas)
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1
'''


def run_vqechem_c18(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    ncarbon = len(atom_list)
    nhydrogen = len(H_atom_coordinates)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math 
    nelec = mol.nelectron
    nocc = nelec//2
    print('lenth of atom list is:', len(atom_list))
    ncore = 2*ncarbon+nhydrogen//2
    print('ncore = ',ncore)
    if basis =='cc-pvdz':
        if ncarbon ==2:
            ncore = ncore
            nacs = 5
        elif ncarbon ==4:
            ncore = ncore+1
            nacs = 9
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
    elif basis =='sto-3g':
        if ncarbon ==2:
            nacs = 4
        elif ncarbon ==4:
            nacs = 8
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
 
    '''
    if len(atom_list)==2:
        if basis == 'sto-3g':
            nacs =8
            ncore = 6
            nvir = 4
            mo_list = range(6,14)
        elif basis == 'cc-pvdz':
            nacs = 6
            ncore = 3
            nvir = 2
            mo_list = range(3,9)
            if (True):
                nacs = 6
                ncore = 3
                nvir = 2
                mo_list = range(3,9)
        else:
            print('basis not supported')
            exit()
    elif len(atom_list) == 4:
        if basis =='sto-3g':
            nacs = 4
            ncore = 3
            nvir = 2
            mo_list = range(3,7)
        elif basis == 'cc-pvdz':
            nacs = 8
            ncore = 8
            nvir = 4
            mo_list = range(8,16)
        else:
            print('err')
            exit()
    '''
    nvir = ncore +nacs-nocc
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1


def run_vqechem_c18_sci(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    ncarbon = len(atom_list)
    nhydrogen = len(H_atom_coordinates)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math 
    nelec = mol.nelectron
    nocc = nelec//2
    print('lenth of atom list is:', len(atom_list))
    ncore = 2*ncarbon+nhydrogen//2
    print('ncore = ',ncore)
    if basis =='cc-pvdz':
        if ncarbon ==2:
            ncore = ncore
            nacs = 5
        elif ncarbon ==4:
            ncore = ncore
            nacs = 9+1
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
    elif basis =='sto-3g':
        if ncarbon ==2:
            nacs = 4
        elif ncarbon ==4:
            nacs = 8
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
 
    '''
    if len(atom_list)==2:
        if basis == 'sto-3g':
            nacs =8
            ncore = 6
            nvir = 4
            mo_list = range(6,14)
        elif basis == 'cc-pvdz':
            nacs = 6
            ncore = 3
            nvir = 2
            mo_list = range(3,9)
            if (True):
                nacs = 6
                ncore = 3
                nvir = 2
                mo_list = range(3,9)
        else:
            print('basis not supported')
            exit()
    elif len(atom_list) == 4:
        if basis =='sto-3g':
            nacs = 4
            ncore = 3
            nvir = 2
            mo_list = range(3,7)
        elif basis == 'cc-pvdz':
            nacs = 8
            ncore = 8
            nvir = 4
            mo_list = range(8,16)
        else:
            print('err')
            exit()
    '''
    nvir = ncore +nacs-nocc
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'sci'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E+dE1



def run_vqechem_c18_no_oo(geometry:list,
                atom_list:list, qmmm_charges:list = None,
                pool = None,
                basis: str ='cc-pvdz',
                link_atom : bool = False):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    ncarbon = len(atom_list)
    nhydrogen = len(H_atom_coordinates)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math 
    nelec = mol.nelectron
    nocc = nelec//2
    print('lenth of atom list is:', len(atom_list))
    ncore = 2*ncarbon+nhydrogen//2
    print('ncore = ',ncore)
    if basis =='cc-pvdz':
        if ncarbon ==2:
            ncore = ncore
            nacs = 5
        elif ncarbon ==4:
            ncore = ncore
            nacs = 9+1
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
    elif basis =='sto-3g':
        if ncarbon ==2:
            nacs = 4
        elif ncarbon ==4:
            nacs = 8
        elif ncarbon == 3:
            nacs = 1
        elif ncarbon == 6:
            nacs = 1
        print('ncas = ',nacs)
        mo_list = np.arange(ncore,ncore+nacs)
 
    '''
    if len(atom_list)==2:
        if basis == 'sto-3g':
            nacs =8
            ncore = 6
            nvir = 4
            mo_list = range(6,14)
        elif basis == 'cc-pvdz':
            nacs = 6
            ncore = 3
            nvir = 2
            mo_list = range(3,9)
            if (True):
                nacs = 6
                ncore = 3
                nvir = 2
                mo_list = range(3,9)
        else:
            print('basis not supported')
            exit()
    elif len(atom_list) == 4:
        if basis =='sto-3g':
            nacs = 4
            ncore = 3
            nvir = 2
            mo_list = range(3,7)
        elif basis == 'cc-pvdz':
            nacs = 8
            ncore = 8
            nvir = 4
            mo_list = range(8,16)
        else:
            print('err')
            exit()
    '''
    nvir = ncore +nacs-nocc
    mm_coords = []
    mm_charges = []
    if qmmm_charges is not None:
        mm_list = [x for x in np.arange(len(geometry)) if x not in atom_list]
        for i in mm_list:
            mm_coords.append(geometry[i][1])
            mm_charges.append(qmmm_charges[i])

    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':nacs,'ncore':ncore,'mo_list':mo_list,
                        'shift':0.5,'qmmm_coords':mm_coords,
                        'qmmm_charges':mm_charges},
               'ops' : {'class':'fermionic','spin_sym':'sa','ops_pool':pool},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300,'tol':0.01},
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'vqe'}
              }
    #ansatz = run_vqe(mol,options)
    E, dE1, dE2 = vqe_oo(mol, options, nvir)

    return E,dE1

def get_hamiltonian(geometry:list,
                atom_list:list,
                fragment_charge = None,
                qmmm_charges:list = None,
                pool = None,
                basis: str ='sto-3g',
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
    #import sys
    #sys.path.append('/es01/home/shanghhui/mahuan/program/vqechem/src')
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo
    #from scf_from_pyscf import 
    import scipy
    import math

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
               'oo' : {'basis':'minao','low_level':'mp2','type':'hf','diag':'get_ham'},
               'file': {'save_directory':save_directory}
              }
    #ansatz = run_vqe(mol,options)
    ham = vqe_oo(mol, options, nvir)

    return ham



def chemps2(  geometry:list,
                atom_list:list,
                basis:str='sto-3g',
                link_atom : bool = False):
    import sys
    sys.path.append('/public/home/jlyang/quantum/program/vqechem/src')
    from scf_from_pyscf import PySCF, pyscf_interface , get_1e_integral, get_2e_integral
    import ctypes
    import PyCheMPS2
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(geometry,atom_list,mode = 'extend')
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    #print('molecule after link H atoms are added')
    #print(mol.atom)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = basis
    mol.build()
    
    options =   {
                         'mapping'        : 'JW',
                         'ncas'           : None,
                         'ncore'          : None,
                         'mo_list'        : None,
                         'shift'          : 0
                }

    hf = pyscf_interface(mol,options)
    FOCK = hf._h1
    CONST = hf._Enuc
    TEI = get_2e_integral(hf)
    Norb = hf._nmo
    Nel = sum(hf._mol.nelec)

    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    orbirreps = np.zeros([ Norb ], dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
    HamCheMPS2.setEconst( CONST )
    for cnt1 in range(Norb):
        for cnt2 in range(Norb):
            HamCheMPS2.setTmat(cnt1, cnt2, FOCK[cnt1, cnt2])
            for cnt3 in range(Norb):
                for cnt4 in range(Norb):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, TEI[cnt1, cnt3, cnt2, cnt4]) #From chemist to physics notation
    '''HamCheMPS2.save()
    exit(123)'''
    # Killing output if necessary
    # to be fixed later
    if ( False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)
    #if ( Norb <= 10 ):
    if (Norb <10):
    #if (False):
        # FCI ground state calculation
        #this part will raise Segmentation fault, the fault is in: /public/home/jlyang/quantum/anaconda3/envs/vqechem/lib/python3.7/site-packages/PyCheMPS2.cpython-37m-x86_64-linux-gnu.so ,this part is going to be fixed later, or skip this part
        assert( Nel % 2 == 0 )
        Nel_up       = Nel / 2
        Nel_down     = Nel / 2
        Irrep        = 0
        maxMemWorkMB = 100.0
        FCIverbose   = 2
        
        theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose )
        GSvector = np.zeros( [ theFCI.getVecLength() ], dtype=ctypes.c_double )
        theFCI.FillRandom( theFCI.getVecLength() , GSvector ) # Random numbers in [-1,1[
        GSvector[ theFCI.LowestEnergyDeterminant() ] = 12.345 # Large component for quantum chemistry
        print('start FCI.GSDavidson')
        
        EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
        
        print('end FCI.GCDavidson')
        #SpinSquared = theFCI.CalcSpinSquared( GSvector )
        TwoRDM = np.zeros( [ Norb**4 ], dtype=ctypes.c_double )
        
        ############################
        theFCI.Fill2RDM( GSvector, TwoRDM )# the source of segmentation fault
        ############################
        
        TwoRDM = TwoRDM.reshape( [Norb, Norb, Norb, Norb], order='F' )
        TwoRDM = np.swapaxes( TwoRDM, 1, 2 ) #From physics to chemistry notation
        del theFCI
    else:
    
        # DMRG ground state calculation
        assert( Nel % 2 == 0 )
        TwoS  = 0
        Irrep = 0
        Prob  = PyCheMPS2.PyProblem( HamCheMPS2, TwoS, Nel, Irrep )

        OptScheme = PyCheMPS2.PyConvergenceScheme(3) # 3 instructions
        #OptScheme.setInstruction(instruction, D, Econst, maxSweeps, noisePrefactor)
        OptScheme.setInstruction(0,  500, 1e-10,  3, 0.05)
        OptScheme.setInstruction(1, 1000, 1e-10,  3, 0.05)
        OptScheme.setInstruction(2, 1000, 1e-10, 10, 0.00) # Last instruction a few iterations without noise
        theDMRG = PyCheMPS2.PyDMRG( Prob, OptScheme )
        EnergyCheMPS2 = theDMRG.Solve()

    return EnergyCheMPS2
