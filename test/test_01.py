from mqc.system.structure import Structure_Metal_Mol
from mqc.system.fragment import Fragment_Metal_Mol_DMET
from mqc.dcalgo.dmet import DMET_Base
from mqc.dcalgo.option import mbe_option
def test():
    structure_file_name = "../structure/POSCAR_acid"

    struct = Structure_Metal_Mol(file_name = structure_file_name, file_format= "POSCAR", cluster_size = 4)
    struct.build()

    frag = Fragment_Metal_Mol_DMET(structure = struct, basis='sto-3g', charge=0, restricted=True, run_ccsd=False)
    frag.build()

    dmet = DMET_Base(frag,method = "CC",)
    dmet.build()
    dmet.run()
    
    print("energy = ",dmet.energy)

    return dmet


if __name__ == "__main__":
    test()
