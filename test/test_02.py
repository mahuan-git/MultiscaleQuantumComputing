from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.mbe import MBE_base

def test():
    geometry = hydrogen_chain()

    struct = Structure(geometry = geometry)
    struct.build()

    frag = SimpleFragment(structure = struct, natom_per_fragment=2)
    frag.build()

    mbe = MBE_base(fragment = frag,isTI=False,link_atom="extend")
    mbe.get_mbe_energy()
    print("mbe energy: ",mbe.mbe_energy)
    return mbe


if __name__ == "__main__":
    test()
