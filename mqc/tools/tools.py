import numpy as np
import scipy.linalg as la
import pyscf.scf.hf

def Frac2Real(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, cellsize)

def Real2Frac(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, la.inv(cellsize))

def get_distance(coordinate_1,coordinate_2):
    assert(len(coordinate_1)==len(coordinate_2)==3)
    distance = 0
    for i in range(len(coordinate_1)):
        distance +=(coordinate_1[i]-coordinate_2[i])**2
    distance = np.sqrt(distance)
    return distance

