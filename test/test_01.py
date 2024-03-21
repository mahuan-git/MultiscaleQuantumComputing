from mqc.system.structure import Structure_Al
from mqc.system.fragment import Fragment_Al_DMET
from mqc.dcalgo.dmet import DMETAlgo
def test():
    structure_file_name = "../structure/POSCAR_acid"

    struct = Structure_Al(file_name = structure_file_name, file_format= "POSCAR", cluster_size = 4)
    struct.build()

    frag = Fragment_Al_DMET(structure = struct, basis='sto-3g', charge=0, restricted=True, run_ccsd=False)
    frag.build()

    dmet = DMETAlgo(frag,method = "CC",)
    dmet.build()
    dmet.run()
    
    print("energy = ",dmet.energy)

    return dmet


if __name__ == "__main__":
    test()
