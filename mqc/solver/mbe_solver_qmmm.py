import numpy as np
import os

from pyscf import gto, scf, qmmm
from pyscf.cc import ccsd
from pyscf.mp.mp2 import MP2
from itertools import combinations

from mqc.tools.tools import int_charge
from mqc.system.fragment import Fragment
from mqc.tools.link_atom_tool import add_link_atoms

def pyscf_rhf_qmmm(  fragment :Fragment,
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

    mf = scf.RHF(mol)
    mf = qmmm.mm_charge(mf, fragment.structure.mm_coords, fragment.structure.mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    return mf.e_tot

def pyscf_uhf_qmmm(  fragment :Fragment,
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

    mf = scf.UHF(mol)
    mf = qmmm.mm_charge(mf, fragment.structure.mm_coords, fragment.structure.mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    return mf.e_tot

def pyscf_dft_qmmm(  fragment :Fragment,
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
    mf = qmmm.mm_charge(mf, fragment.structure.mm_coords, fragment.structure.mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    return mf.e_tot

def pyscf_ccsd_qmmm(  fragment :Fragment,
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
    mf = qmmm.mm_charge(mf, fragment.structure.mm_coords, fragment.structure.mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)
    
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ERHF = mf.e_tot
    ECCSD = ERHF + ECORR
    return ECCSD

def pyscf_mp2_qmmm(  fragment :Fragment,
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
    mf = qmmm.mm_charge(mf, fragment.structure.mm_coords, fragment.structure.mm_charges)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=None)

    mp2 = MP2( mf )
    mp2.verbose = 0
    mp2.run()

    return mp2.e_tot

