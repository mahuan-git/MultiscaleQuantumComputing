from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.dmet import DMET_Base

def test():
    geometry = hydrogen_chain()

    struct = Structure(geometry = geometry)
    struct.build()

    frag = SimpleFragment(structure = struct, natom_per_fragment=2)
    frag.build()

    dmet = DMET_Base(fragment = frag,method = "FCI",isTI = False)
    dmet.build()
    energy = dmet.run()
    print("dmet energy: ",energy)
    return dmet


if __name__ == "__main__":
    test()
