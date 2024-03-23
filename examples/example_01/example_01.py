from mqc.system.structure import Structure_Metal_Mol
from mqc.system.fragment import Fragment_Metal_Mol_DMET
from mqc.dcalgo.dmet import DMET_Base

def main(structure_file_name = "./POSCAR_acid"):
    struct = Structure_Metal_Mol(file_name = structure_file_name, file_format= "POSCAR", metal_name ="Al", cluster_size = 4)
    struct.build()

    frag = Fragment_Metal_Mol_DMET(structure = struct, basis='sto-3g', charge=0, restricted=True, run_ccsd=False)
    frag.build()

    dmet = DMET_Base(frag,method = "VQECHEM",)
    dmet.build()
    dmet.run()
    
    print("energy = ",dmet.energy)
    return dmet


if __name__ == "__main__":
    main()
