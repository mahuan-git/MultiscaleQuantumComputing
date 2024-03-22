'''
solvers for mbe
args for a mbe solver should include:
    fragment : Fragment class
    atom list : list, atom index contained in the fragment
    link_atom : bool 
'''
import numpy as np
import os

from pyscf import gto, scf
from pyscf.cc import ccsd
from pyscf.mp.mp2 import MP2

from mqc.tools.tools import int_charge
from mqc.system.fragment import Fragment
from mqc.tools.link_atom_tool import add_link_atoms

def pyscf_uhf(  fragment :Fragment,
                atom_list:list,  
                link_atom : str = "extend",
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()

    mf = scf.UHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    return mf.e_tot

def pyscf_rhf(  fragment :Fragment,
                atom_list:list,  
                link_atom : str = "extend",
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    return mf.e_tot

def pyscf_dft(  fragment :Fragment,
                atom_list:list,  
                link_atom : str = "extend",
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom == True:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()
    from pyscf import dft
    mf = dft.KS(mol,xc='HYB_GGA_XC_B3LYP')
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    return mf.e_tot

def pyscf_ccsd(  fragment :Fragment,
                atom_list:list,  
                link_atom : str = "extend",
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ERHF = mf.e_tot
    ECCSD = ERHF + ECORR
    return ECCSD

def pyscf_mp2(  fragment :Fragment,
                atom_list:list,  
                link_atom : str = "extend",
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    mp2 = MP2( mf )
    mp2.verbose = 0
    mp2.run()

    return mp2.e_tot

def run_vqechem(  fragment :Fragment,
                  atom_list:list,  
                  link_atom : str = "extend",
                  ):

    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = link_atom ,connection=fragment.connection)
        for coordinate in H_atom_coordinates:
            mol.atom.append(('H',coordinate))
    charge = 0
    for idx in atom_list:
        charge += fragment.qm_atom_charge[idx]
    mol.charge = int_charge(charge=charge, thres = 0.1)
    if mol.nelectron%2==0:
        mol.spin=0
    else:
        mol.spin=1
    mol.basis = fragment.basis
    mol.build()

    from algorithms import run_vqe 
 
    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'ncas':None,'ncore':None,'shift':0.5,
                        'qmmm_coords':fragment.structure.mm_coords,
                        'qmmm_charges':fragment.structure.mm_charges},
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
    from VQEChem.algorithms import run_vqe
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
    from algorithms import run_vqe
    from VQEChem.orbital_optimize import vqe_oo 
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
