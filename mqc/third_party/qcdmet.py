import sys
sys.path.append('/public/home/jlyang/quantum/program/QC-DMET-py3/src')
import localintegrals#, qcdmet_paths
import dmet as dmet
from pyscf import gto, scf , mp
from pyscf.cc import ccsd
import numpy as np


def test(shift = 20.0):
    nat = 18
    shift_deg = shift
    shift =shift*2*np.pi/360
    R = 7.31/2
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = []

    angle = 0.0
    for i in range( nat // 2 ):
        mol.atom.append(('C', (R * np.cos(angle        ), R * np.sin(angle        ), 0.0)))
        mol.atom.append(('C', (R * np.cos(angle + shift), R * np.sin(angle + shift), 0.0)))
        angle += 4.0 * np.pi / nat
    mol.basis = 'cc-pvdz'
    #mol.basis = 'sto-3g'
    mol.build()

    mf = scf.RHF( mol )
    mf.verbose = 3
    mf.scf()
    if ( False ):   
        ccsolver = ccsd.CCSD( mf )
        ccsolver.verbose = 0
        ECORR, t1, t2 = ccsolver.ccsd()
        ECCSD = mf.hf_energy + ECORR
        print("ECCSD for alpha %f =%f"%(alpha,ECCSD))
        exit()
    if ( False ):
        mp2solver = mp.MP2( mf )
        ECORR, t_mp2 = mp2solver.kernel()
        EMP2 = mf.hf_energy + ECORR
        print("EMP2 for alpha %f = %f "%(alpha,EMP2))
    
    myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
    myInts.molden( 'polyyne-loc.molden' )
    myInts.TI_OK = True
    atoms_per_imp = 1# Impurity size counted in number of atoms
    assert ( nat % atoms_per_imp == 0 )
    orbs_per_imp = myInts.Norbs * atoms_per_imp // nat
    impurityClusters = []
    for cluster in range( nat // atoms_per_imp ):
        impurities = np.zeros( [ myInts.Norbs ], dtype=int )
        for orb in [1,3,4,5]:
            impurities[ orbs_per_imp*cluster + orb ] = 1
        impurityClusters.append( impurities )
    isTranslationInvariant = True # Both in meta_lowdin (due to px, py) and Boys TI is not OK
    method = 'CC'
    SCmethod = 'NONE' #Don't do it self-consistently
    theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod)
    energy=theDMET.doselfconsistent()
    print('energy = %f when shift = %f'%(energy,shift_deg))
    #theDMET.dump_bath_orbs( 'polyyne-bath.molden' )
if __name__=="__main__":
    import sys
    shift = float(sys.argv[1])
    test(shift)
