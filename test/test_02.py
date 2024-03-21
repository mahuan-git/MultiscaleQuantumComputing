from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.mbe import MBEAlgo
def test():
    geometry = hydrogen_chain()

    struct = Structure(geometry = geometry)
    struct.build()

    frag = SimpleFragment(structure = struct, natom_per_fragment=2)
    frag.build()

    mbe = MBEAlgo(fragment = frag,isTI=False,link_atom=False)
    mbe.get_mbe_energy()
    
    return mbe


if __name__ == "__main__":
    test()
