from mqc.system.structure import Structure_protein
from mqc.system.fragment import Fragment_protein
from mqc.dcalgo.mbe import MBE_protein
from mqc.dcalgo.option import mbe_option

def test():
    protein_file_name = "../structure/3zmg.mol2"
    struct = Structure_protein(file_name = protein_file_name)
    struct.build()

    frag = Fragment_protein(structure = struct, basis = "sto-3g",charge = sum(struct._atom_charge))
    frag.build()

    option = mbe_option(solver = "vqe_oo",ncas =4)

    #mbe = MBE_protein(fragment = frag, mbe_option=option)
    #mbe.get_mbe_energy()
    #print("mbe energy: ",mbe.mbe_energy)
    return frag

if __name__ == "__main__":
    test()
