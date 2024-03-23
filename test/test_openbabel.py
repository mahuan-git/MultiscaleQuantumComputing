from openbabel import openbabel
from openbabel import pybel

def read(filename = "../structure/POSCAR_acid"):
    obmol = openbabel.OBMol()
    obconv=openbabel.OBConversion()
    obconv.SetInAndOutFormats("POSCAR",None)
    obconv.ReadFile(obmol, filename)
    obmol.AddHydrogens()
    pymol = pybel.Molecule(obmol)
    return pymol
