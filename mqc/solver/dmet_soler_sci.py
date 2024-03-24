import os
import sys
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from pyscf import gto, scf,ao2mo , fci
import rhf_newtonraphson

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, energytype='LAMBDA', chempot_imp=0.0, printoutput=True ):

    assert (( energytype == 'LAMBDA' ) or ( energytype == 'LAMBDA_AMP' ) or ( energytype == 'LAMBDA_ZERO' ) or ( energytype == 'CASCI' ))
    raise NotImplementedError
    # Killing output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( DMguessRHF )

    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc )
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    # Check the RHF solution
    assert( Nel % 2 == 0 )
    numPairs = Nel // 2
    FOCKloc = FOCKcopy + np.einsum('ijkl,ij->kl', TEI, DMloc) - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc)
    eigvals, eigvecs = np.linalg.eigh( FOCKloc )
    idx = eigvals.argsort()
    eigvals = eigvals[ idx ]
    eigvecs = eigvecs[ :, idx ]
    print("psi4cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1])
    DMloc2  = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
    print("Two-norm difference of 1-RDM(RHF) and 1-RDM(FOCK(RHF)) =", np.linalg.norm(DMloc - DMloc2))
    
    # Get the CC solution from pyscf
    cisolver = fci.SCI( mf )
    cisolver.verbose = 5
    e ,vec = cisolver.kernel(h1e=FOCKcopy, eri = TEI, norb = Norb, nelec = Nel)
    ERHF = mf.e_tot
    e_fci = e

    
        # The 2-RDM is not required
        # Active space energy is computed with the Fock operator of the core (not rescaled)
    print("E FCI =", e_fci)
    pyscfRDM1 = cisolver.make_rdm1(vec, Norb, Nel)                                  # MO space
    pyscfRDM1 = 0.5 * (pyscfRDM1 + pyscfRDM1.T)                       # Symmetrize
    pyscfRDM1 = np.dot(mf.mo_coeff, np.dot(pyscfRDM1, mf.mo_coeff.T)) # From MO to localized space
    ImpurityEnergy = e_fci
    if ( chempot_imp != 0.0 ):
        # [FOCK - FOCKcopy]_{ij} = chempot_imp * delta(i,j) * delta(i \in imp)
        ImpurityEnergy += np.einsum('ij,ij->', FOCK - FOCKcopy, pyscfRDM1)

    # else:
    
    #     # Compute the DMET impurity energy based on the lambda equations
    #     if ( energytype == 'LAMBDA' ):
    #         ccsolver.solve_lambda()
    #         pyscfRDM1 = ccsolver.make_rdm1() # MO space
    #         pyscfRDM2 = ccsolver.make_rdm2() # MO space
    #     if ( energytype == 'LAMBDA_AMP' ):
    #         # Overwrite lambda tensors with t-amplitudes
    #         pyscfRDM1 = ccsolver.make_rdm1(t1, t2, t1, t2) # MO space
    #         pyscfRDM2 = ccsolver.make_rdm2(t1, t2, t1, t2) # MO space
    #     if ( energytype == 'LAMBDA_ZERO' ):
    #         # Overwrite lambda tensors with 0.0
    #         fake_l1 = np.zeros( t1.shape, dtype=float )
    #         fake_l2 = np.zeros( t2.shape, dtype=float )
    #         pyscfRDM1 = ccsolver.make_rdm1(t1, t2, fake_l1, fake_l2) # MO space
    #         pyscfRDM2 = ccsolver.make_rdm2(t1, t2, fake_l1, fake_l2) # MO space
    #     pyscfRDM1 = 0.5 * ( pyscfRDM1 + pyscfRDM1.T ) # Symmetrize
        
        
        # # Change the pyscfRDM1/2 from MO space to localized space
        # pyscfRDM1 = np.dot(mf.mo_coeff, np.dot(pyscfRDM1, mf.mo_coeff.T ))
        # pyscfRDM2 = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, pyscfRDM2)
        # pyscfRDM2 = np.einsum('bj,ajkl->abkl', mf.mo_coeff, pyscfRDM2)
        # pyscfRDM2 = np.einsum('ck,abkl->abcl', mf.mo_coeff, pyscfRDM2)
        # pyscfRDM2 = np.einsum('dl,abcl->abcd', mf.mo_coeff, pyscfRDM2)
        # ECCSDbis = CONST + np.einsum('ij,ij->', FOCKcopy, pyscfRDM1) + 0.5 * np.einsum('ijkl,ijkl->', TEI, pyscfRDM2)
        # print("ECCSD1 =", ECCSD)
        # print("ECCSD2 =", ECCSDbis)
        
        # # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
        # ImpurityEnergy = CONST \
        #                + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
        #                + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp]) \
        #                + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
        #                + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
        #                + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
        #                + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
    
    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
    
    return ( ImpurityEnergy, pyscfRDM1 )
