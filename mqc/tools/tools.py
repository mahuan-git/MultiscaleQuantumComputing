import numpy as np
import scipy.linalg as la

def get_distance(coordinate_1,coordinate_2)-> float:
    assert(len(coordinate_1)==len(coordinate_2)==3)
    distance = 0
    for i in range(len(coordinate_1)):
        distance +=(coordinate_1[i]-coordinate_2[i])**2
    distance = np.sqrt(distance)
    return distance

def int_charge(charge,thres)-> float:
    """Transfer float charge into integer charge"""
    int_charge = int(np.around(charge,decimals=0))
    assert abs(int_charge-charge) < thres, "Bad charge input"
    return int_charge

def get_ops_pool(geometry, atom_list,):
    from pyscf import gto
    from vqechem.scf_from_pyscf import pyscf_interface
    from vqechem.fermion_operator import FermionOps
    from vqechem.set_options import set_options
    from mqc.dcalgo.option import mbe_option
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