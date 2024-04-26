from mqc.system.structure import Structure_Metal_Mol, make_unbonded_structure
from mqc.system.fragment import Fragment_Metal_Mol_DMET
from mqc.dcalgo.dmet import DMET_Base
from pyscf import gto, scf
from pyscf.cc import ccsd
import time

def run_ccsd(structure_file_name = "./POSCAR_acid",basis = "sto-3g",charge = 0, spin = 0,unbond = False):
    struct = Structure_Metal_Mol(file_name = structure_file_name, file_format= "POSCAR", metal_name ="Al", cluster_size = 4)
    struct.build()
    print(structure_file_name)
    print(basis)
    if unbond == True:
        print("Unbonded Structure")
        struct = make_unbonded_structure(struct,dist = 10) ## unbonded structure
    else:
        print("Bonded Structure")
    print(struct.geometry)
    print(f"Number of aluminum atoms = {len(struct._substrate_select)}")
    mol = gto.Mole()
    mol.atom = struct.geometry
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    mol.build()

    print("Start running RHF")
    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.run()
    print(f"Finish running RHF, E_RHF = {mf.e_tot}, start runnning CCSD")

    start = time.perf_counter()
    cc=ccsd.CCSD(mf)
    cc.max_cycle = 100
    cc.run()
    end = time.perf_counter()
    print(f"Finished running CCSD, time used = {end-start}")

    print("CCSD energy: ", cc.e_tot)


if __name__ == "__main__":
    import sys
    argv = sys.argv
    if len(argv)>1:
        struct_file_name = argv[1]
    else:
        struct_file_name = "POSCAR_acid"
    run_ccsd(structure_file_name =struct_file_name,basis ="6-31G",unbond = True)
    run_ccsd(structure_file_name =struct_file_name,basis = "6-31G",unbond = False)

