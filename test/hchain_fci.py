from mqc.system.structure import Structure , hydrogen_ring, hydrogen_chain
from mqc.system.fragment import SimpleFragment
from mqc.dcalgo.dmet import DMET_Base
import numpy as np
from pyscf import gto,scf,fci

def test():
    hf_energy_list = []
    ci_energy_list = []
    for bl in np.arange(0.6,3.1,0.1):
        geometry = hydrogen_chain(natom = 10,bondlength = bl)
        mol = gto.Mole()
        mol.atom = geometry
        mol.basis = "sto-3g"
        mol.build()
        mf = scf.RHF(mol)
        mf.run()
        hf_energy_list.append(mf.e_tot)
        ci = fci.FCI(mf)
        ci.run()
        ci_energy_list.append(ci.e_tot)
        print("HF energy: ",hf_energy_list)
        print("FCI energy: ",ci_energy_list)
    return


if __name__ == "__main__":
    test()
