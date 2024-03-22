'''
solvers for mbe
args for a mbe solver should include:
    fragment : Fragment class
    atom list : list, atom index contained in the fragment
    link_atom : bool 
'''
import numpy as np


from pyscf import gto, scf
from pyscf.cc import ccsd
from pyscf.mp.mp2 import MP2

from mqc.tools.tools import int_charge
from mqc.system.fragment import Fragment
from mqc.tools.link_atom_tool import add_link_atoms
from mqc.dcalgo.option import mbe_option
from vqechem.algorithms import run_vqe
from vqechem.orbital_optimize import vqe_oo
import copy

def pyscf_uhf(  fragment :Fragment,
                atom_list:list, 
                option: mbe_option
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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
                option: mbe_option
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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
                option: mbe_option
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom == True:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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

def pyscf_ccsd( fragment :Fragment,
                atom_list:list, 
                option: mbe_option
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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
    E_corr, t1, t2 = ccsolver.ccsd()
    E_hf = mf.e_tot
    E_ccsd = E_hf + E_corr
    return E_ccsd

def pyscf_mp2(  fragment :Fragment,
                atom_list:list, 
                option: mbe_option
                ):
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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

def vqechem(    fragment :Fragment,
                atom_list:list, 
                option: mbe_option
                ):
    option_cp = copy.deepcopy(option)
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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
    nocc = mol.nelectron//2
    ncas_occ = int(np.floor(option.ncas/2))
    ncas_vir = int(np.ceil(option.ncas/2))
    assert (ncas_occ + ncas_vir) == option.ncas
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+option.ncas)    
    option_cp.update(ncore = ncore,mo_list=mo_list,qmmm_coords = None,qmmm_charges = None)
    vqe_options = option_cp.make_vqe_options()
    ansatz = run_vqe(mol,vqe_options)
    return ansatz._energy

def vqe_oo(    fragment :Fragment,
                atom_list:list, 
                option: mbe_option
                ):
    option_cp = copy.deepcopy(option)
    mol=gto.Mole()
    mol.atom=[]
    for i in atom_list:
        mol.atom.append(fragment.qm_geometry[i])
    if option.link_atom is not None:
        H_atom_coordinates = add_link_atoms(fragment.qm_geometry,atom_list,mode = option.link_atom ,connection=fragment.connection)
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
    nocc = mol.nelectron//2
    ncas_occ = int(np.floor(option.ncas/2))
    ncas_vir = int(np.ceil(option.ncas/2))
    assert (ncas_occ + ncas_vir) == option.ncas
    ncore = nocc - ncas_occ
    nvir = ncas_vir
    mo_list = range(ncore,ncore+option.ncas)    
    option_cp.update(ncore = ncore,mo_list=mo_list,qmmm_coords = None,qmmm_charges = None)
    vqe_options = option_cp.make_vqe_options()
    E, dE1, dE2 = vqe_oo(mol, vqe_options, nvir)
    return E+dE1

