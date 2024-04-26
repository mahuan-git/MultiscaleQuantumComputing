from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.dmet import DMET_Base
import numpy as np
def test():
    dmet_energy = []
    for dist in np.arange(0.6,2.1,0.1):
        geometry = hydrogen_ring(bondlength = dist)

        struct = Structure(geometry = geometry)
        struct.build()

        frag = SimpleFragment(structure = struct, natom_per_fragment=2)
        frag.build()

        dmet = DMET_Base(fragment = frag,method = "VQECHEM",isTI = True)
        dmet.build()
        energy = dmet.run()
        #print("dmet energy: ",energy)
        dmet_energy.append(energy)
        print("DMET energy: ",dmet_energy)
    return


if __name__ == "__main__":
    test()
