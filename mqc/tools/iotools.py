import re
import numpy as np

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

