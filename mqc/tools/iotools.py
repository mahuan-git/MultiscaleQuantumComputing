import re
import numpy as np
from mqc.tools.tools import Frac2Real
from openbabel import pybel

def read_poscar(fname="POSCAR"):
    """
    Read cell structure from a VASP POSCAR file.
    
    Args:
        fname: file name.

    Returns:
        cell: cell, without build, unit in A.
    """

    with open(fname, 'r') as f:
        lines = f.readlines()

        # 1 line scale factor
        line = lines[1].split()
        assert len(line) == 1
        factor = float(line[0])

        # 2-4 line, lattice vector 
        a = np.array([np.fromstring(lines[i], dtype=np.double, sep=' ') \
                for i in range(2, 5)]) * factor

        # 5, 6 line, species names and numbers
        sp_names = lines[5].split()
        if all([name.isdigit() for name in sp_names]):
            # 5th line can be number of atoms not names.
            sp_nums = np.fromstring(lines[5], dtype=int, sep=' ')
            sp_names = ["X" for i in range(len(sp_nums))]
            line_no = 6
        else:
            sp_nums = np.fromstring(lines[6], dtype=int, sep=' ')
            line_no = 7

        # 7, cartisian or fraction or direct
        line = lines[line_no].split()
        assert len(line) == 1
        use_cart = line[0].startswith(('C', 'K', 'c', 'k'))
        line_no += 1

        # 8-end, coords
        atom_col = []
        for sp_name, sp_num in zip(sp_names, sp_nums):
            for i in range(sp_num):
        # there may be 4th element for comments or fixation.
                coord = np.array(list(map(float, \
                        lines[line_no].split()[:3])))
                if use_cart:
                    coord *= factor
                else:
                    coord = Frac2Real(a, coord)
                atom_col.append((sp_name, coord))
                line_no += 1

        return atom_col

def read_mol_structure(fname = 'structure.mol'):
    geometry = []
    mol_file = open(fname,'r+')
    frags=[]
    line = mol_file.readline()
    while (line!=''):
        if (len(line)<4):
            pass
        elif (line[0]=='#'):
            pass
        else:
            atom = re.findall(r"[A-Z][a-z]*",line)
            coordinates = re.findall('-?\d+\.\d+',line)
            if not len(coordinates)==3:
                pass
            else:
                geometry.append([atom[0],[float(coordinates[0]),float(coordinates[1]),float(coordinates[2])]])
        line = mol_file.readline()
    return geometry

def read_mol2_structure(filename):
    """read mol2 file for protein-lgand models"""
    file = open(filename,"r+")
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
            if line.split()[6] in ["GLY","ALA","PRO","VAL","LEU","ILE","SER","THR","GLN","ASN","MET","CYS","PHE","TYR","TRP","ASP","GLU","SEC","ARG","LYS","HIS"]:
                residue_num += 1
    else:
        residue_num = int(lines[bond_idx-1].split()[6])
    res=[[] for i in range(residue_num)]
    res_names = []
    protein_geometry = []
    atom_type = []
    atom_label = []
    atom_charge = []
    mm_coords = []
    mm_charge = []
    ligand_geo = []
    ligand_atom_charge = []
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
            protein_geometry.append((data[1][0],(float(data[2]),float(data[3]),float(data[4]))))
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
    return res,res_names, protein_geometry, bonds, bond_types,atom_type,atom_label,atom_charge,res_charge,ligand_geo,ligand_atom_charge,ligand_atom_idx,mm_coords,mm_charge


def print_structure_for_Gauss_View(geometry,file_name = "structure.com"):
        GV_file = open(file_name,'w+')
        GV_file.write("# HF/3-21G** opt pop=full gfprint\n\nTitle: Created by Jmol version 14.31.60  2021-10-18 20:23\n\n0 1\n")
        for i in range(len(geometry)):
            GV_file.write(geometry[i][0]+'    '+str(geometry[i][1][0])+'   '+str(geometry[i][1][1])+'   '+str(geometry[i][1][2])+'   \n')
        GV_file.close()
