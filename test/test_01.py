from mqc.system.structure import Structure_Al
from mqc.system.fragment import Fragment
from mqc.dcalgo.mbe import MBE
from mqc.tools.tools import get_distance
def test():
    structure_file_name = "../structure/POSCAR_acid"

    struct = Structure_Al(file_name = structure_file_name, file_format= "POSCAR",build = False,select_al_cluster = True, cluster_size = 4)

    return struct


if __name__ == "__main__":
    test()
