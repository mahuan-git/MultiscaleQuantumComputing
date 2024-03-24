from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.mbe import MBE_Base
from mqc.dcalgo.option import mbe_option
def test():
    geometry = hydrogen_ring()

    struct = Structure(geometry = geometry)
    struct.build()

    frag = SimpleFragment(structure = struct, natom_per_fragment=2)
    frag.build()

    option = mbe_option(solver = "vqechem",link_atom =None)
    mbe = MBE_Base(fragment = frag,mbe_option=option)
    energy = mbe.get_mbe_energy()
    print("dmet energy: ",energy)
    return mbe


if __name__ == "__main__":
    test()
