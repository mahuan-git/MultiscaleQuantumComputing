from mqc.tools.tools import get_distance
from itertools import combinations
import numpy as np


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
                                add_atom_idx : int
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
        raise ValueError('add atom index does not match, check please')   
    return link_atom_coordinate


def add_link_atoms_extend( geometry,
                            atom_list,
                            ):
    ''' 
    '''
    natoms = len(geometry)
    link_atom_coordinate=[]
    for atoms in combinations(atom_list,2):
        if abs(atoms[1]-atoms[0])==1:   ## two selected atoms are neighbours
            atom_idx_1 = max(atoms)+1
            atom_idx_2 = min(atoms)-1
            if atom_idx_1%natoms not in atom_list:
                coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_1)
                link_atom_coordinate.append(coordinate)
            if atom_idx_2%natoms not in atom_list:
                coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_2)
                link_atom_coordinate.append(coordinate)

        elif abs(atoms[1]-atoms[0])==natoms-1:   ##the two selected atoms are at the head and the tail of the molecule respectively
            distance = get_distance(geometry[atoms[1]][1],geometry[atoms[0]][1])
            if distance >3.0:  ## indicate that the head and the tail are not connected
                pass
            else:        ## the head and the tail are connected. 
                atom_idx_1 = 1
                atom_idx_2 = -2
                atoms = [-1,0]
                if (atom_idx_1%natoms) not in atom_list:
                    coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_1)
                    link_atom_coordinate.append(coordinate)
                if (atom_idx_2%natoms) not in atom_list:
                    coordinate = get_link_atom_coordinate(geometry,atoms,atom_idx_2)
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
    link_atom_coordinates = []
    for bond in bondbreak :
        coordinate_1 = np.array(geometry[bond[0]][1])
        coordinate_2 = np.array(geometry[bond[1]][1])
        dist = get_distance(coordinate_1,coordinate_2)
        vec = (coordinate_2-coordinate_1)*bondlength/dist
        coordinate_new = coordinate_1+vec
        link_atom_coordinates.append(coordinate_new)
    return link_atom_coordinates

def add_link_atoms( geometry,
                    atom_list,
                    mode='origin',
                    connection = None
                    ):
    if mode == 'extend':
        link_atom_coordinate = add_link_atoms_extend(geometry,atom_list)
    elif mode =='origin':
        link_atom_coordinate = add_link_atoms_origin(geometry,atom_list,connection = connection)
    else:
        raise ValueError(" mode for adding link atom not supported")
    return link_atom_coordinate
