from openbabel import openbabel
from openbabel import pybel

def read(filename = "../structure/3zmg.mol2"):
    obmol = openbabel.OBMol()
    obconv=openbabel.OBConversion()
    obconv.SetInAndOutFormats("mol2",None)
    obconv.ReadFile(obmol, filename)
    obmol.AddHydrogens()
    pymol = pybel.Molecule(obmol)
    return pymol
