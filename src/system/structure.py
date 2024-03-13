import numpy as np
import scipy.linalg as la
from pyscf.pbc import gto

def Frac2Real(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, cellsize)

def Real2Frac(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, la.inv(cellsize))


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

        cell = gto.Cell()
        cell.a = a
        cell.atom = atom_col
        cell.unit = 'A'
        return cell

class Base_structure():
    def __init__(self):
        pass