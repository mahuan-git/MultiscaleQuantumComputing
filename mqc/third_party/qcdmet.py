import numpy as np
import localintegrals
from dmet import dmet
import time
from scipy import optimize

class MyDMET(dmet):
    def __init__(self, 
                 theInts : localintegrals.localintegrals, 
                 impurityClusters : list, 
                 isTranslationInvariant : bool =False, 
                 method : str='ED', 
                 SCmethod :str ='LSTSQ', 
                 fitImpBath : bool =True, 
                 use_constrained_opt: bool =False):
        method = method.upper()
        super().__init__(theInts, impurityClusters, isTranslationInvariant, method, SCmethod, fitImpBath, use_constrained_opt)
        assert method in ['CC','ED','MP2','VQE','VQECHEM',"FCI",'ADAPT','GET_CIRC']
        
        ## add setting for freeze orbitals.
        self.freez_flag = False
        if (self.TransInv ) and (len(self.impClust[0])!=(self.make_imp_size()[0]*len(self.impClust))):
            self.freez_flag = True
            n_frag_orbs = self.Norb//len(self.impClust)
            self.frag_imp_orbs = self.impClust[0][:n_frag_orbs]
            self.frag_freez_orbs=np.ones(self.Norb//len(self.impClust),dtype=int)-self.frag_imp_orbs
    
    def doexact( self, chempot_imp=0.0 ):
        OneRDM = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        if ( self.doDET == True ) and ( self.doDET_NO == True ):
            self.NOvecs = []
            self.NOdiag = []
        
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
        remainingOrbs = np.ones( [ len( self.impClust[ 0 ] ) ], dtype=float )
        
        for counter in range( maxiter ):
            flag_rhf = np.sum(self.impClust[ counter ]) < 0
            impurityOrbs = np.abs(self.impClust[ counter ])
            numImpOrbs   = np.sum( impurityOrbs )
            if ( self.BATH_ORBS == None ):
                numBathOrbs = numImpOrbs
            else:
                numBathOrbs = self.BATH_ORBS[ counter ]
            numBathOrbs, loc2dmet, core1RDM_dmet = self.helper.constructbath( OneRDM, impurityOrbs, numBathOrbs )
            if ( self.BATH_ORBS == None ):
                core_cutoff = 0.01
            else:
                core_cutoff = 0.5
            for cnt in range(len(core1RDM_dmet)):
                if ( core1RDM_dmet[ cnt ] < core_cutoff ):
                    core1RDM_dmet[ cnt ] = 0.0
                elif ( core1RDM_dmet[ cnt ] > 2.0 - core_cutoff ):
                    core1RDM_dmet[ cnt ] = 2.0
                else:
                    print("Bad DMET bath orbital selection: trying to put a bath orbital with occupation", core1RDM_dmet[ cnt ], "into the environment :-(.")
                    assert( 0 == 1 )

            Norb_in_imp  = numImpOrbs + numBathOrbs
            Nelec_in_imp = int(round(self.ints.Nelec - np.sum( core1RDM_dmet )))
            core1RDM_loc = np.dot( np.dot( loc2dmet, np.diag( core1RDM_dmet ) ), loc2dmet.T )
            self.dmetOrbs.append( loc2dmet[ :, :Norb_in_imp ] ) 
            assert( Norb_in_imp <= self.Norb )
            dmetOEI  = self.ints.dmet_oei(  loc2dmet, Norb_in_imp )
            dmetFOCK = self.ints.dmet_fock( loc2dmet, Norb_in_imp, core1RDM_loc )
            dmetTEI  = self.ints.dmet_tei(  loc2dmet, Norb_in_imp )

            if ( self.NI_hack == True ):
                dmetTEI[:,:,:,numImpOrbs:]=0.0
                dmetTEI[:,:,numImpOrbs:,:]=0.0
                dmetTEI[:,numImpOrbs:,:,:]=0.0
                dmetTEI[numImpOrbs:,:,:,:]=0.0
                umat_rotated = np.dot(np.dot(loc2dmet.T, self.umat), loc2dmet)
                umat_rotated[:numImpOrbs,:numImpOrbs]=0.0
                dmetOEI += umat_rotated[:Norb_in_imp,:Norb_in_imp]
                dmetFOCK = np.array( dmetOEI, copy=True )
            
            print("DMET::exact : Performing a (", Norb_in_imp, "orb,", Nelec_in_imp, "el ) DMET active space calculation.")
            if ( flag_rhf ):
                import pyscf_rhf
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_rhf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )
            elif ( self.method == 'ED' ):
                import chemps2
                print('start chemps2.solve')
                IMP_energy, IMP_1RDM = chemps2.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, chempot_imp )
                print('chemps2.solve completed')
            elif ( self.method == 'CC' ):
                import pyscf_cc
                assert( Nelec_in_imp % 2 == 0 )
                print('start_cc_solve')
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_cc.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, self.CC_E_TYPE, chempot_imp )
                print("end_cc_solve")
            elif ( self.method == 'MP2' ):
                import pyscf_mp2
                assert( Nelec_in_imp % 2 == 0 )
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_mp2.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )

                print('end vqe solver')
            elif ( self.method == 'VQECHEM' ):
                from mqc.solver.dmet_solver_vqechem import solve
                print('start vqe solver from vqechem')
                IMP_energy, IMP_1RDM = solve(dmetOEI,dmetFOCK, dmetTEI,Nelec_in_imp,numImpOrbs,chempot_imp)
                print('end vqe solver')
            elif ( self.method == 'FCI' ):
                from mqc.solver.dmet_solver_fci import solve
                assert( Nelec_in_imp % 2 == 0 )
                print('start fci solve')
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp//2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, self.CC_E_TYPE, chempot_imp )
                print('end fci solver')
            '''
            elif ( self.method == 'VQE' ):
                import vqe
                print('start vqe solver from qcqc')
                IMP_energy, IMP_1RDM = vqe.solve(dmetOEI,dmetFOCK, dmetTEI,Nelec_in_imp,numImpOrbs,chempot_imp)
            elif (self.method=='VQECHEM_TEST'):
                import vqe_test
                ansatz, params = vqe_test.solve(dmetOEI,dmetFOCK, dmetTEI,Nelec_in_imp,numImpOrbs,chempot_imp)
                return ansatz,params#
            elif (self.method=='GET_CIRC'):
                import circ_generate
                circ_generate.circ_save(counter,dmetOEI,dmetFOCK,dmetTEI,Nelec_in_imp,numImpOrbs,chempot_imp)
                IMP_energy=0
                IMP_1RDM=None
            elif (self.method == "ADAPT"):
                import adapt_vqe
                print('start adapt_vqe solver')
                IMP_energy, IMP_1RDM = adapt_vqe.solve(dmetOEI,dmetFOCK, dmetTEI,Nelec_in_imp,chempot_imp)
                print('end adapt_vqe solver')
            '''
            self.energy += IMP_energy
            self.imp_1RDM.append( IMP_1RDM )
            if ( self.doDET == True ) and ( self.doDET_NO == True ):
                RDMeigenvals, RDMeigenvecs = np.linalg.eigh( IMP_1RDM[ :numImpOrbs, :numImpOrbs ] )
                self.NOvecs.append( RDMeigenvecs )
                self.NOdiag.append( RDMeigenvals )
    
            remainingOrbs -= impurityOrbs
        if (self.method=='GET_CIRC'):
            print('all circuits saved exit')
            exit()

        if ( self.doDET == True ) and ( self.doDET_NO == True ):
            self.NOrotation = self.constructNOrotation()
        
        Nelectrons = 0.0
        for counter in range( maxiter ):
            Nelectrons += np.trace( self.imp_1RDM[counter][ :self.imp_size[counter], :self.imp_size[counter] ] )

        if (self.freez_flag==True):
            freez_orbs=np.zeros(self.Norb, dtype = float)
            for i in range(len(self.frag_freez_orbs)):
                if self.frag_freez_orbs[i]==1:
                    freez_orbs[i]=1
            transfo = np.eye( self.Norb, dtype=float )
            totalOEI  = self.ints.dmet_oei(  transfo, self.Norb )
            totalFOCK = self.ints.dmet_fock( transfo, self.Norb, OneRDM )
            self.energy += 0.5 * np.einsum( 'ij,ij->', OneRDM[freez_orbs==1,:], \
                totalOEI[freez_orbs==1,:] + totalFOCK[freez_orbs==1,:] )
            Nelectrons += np.trace( (OneRDM[freez_orbs==1,:])[:,freez_orbs==1] )
            print('number of electrons in the fragment')
            print(Nelectrons)
            print('energy of the fragment')
            print(self.energy)
        else:
            pass

        if ( self.TransInv == True ):
            Nelectrons = Nelectrons * len( self.impClust )
            self.energy = self.energy * len( self.impClust )
            remainingOrbs[:] = 0
        # When an incomplete impurity tiling is used for the Hamiltonian, self.energy should be augmented with the remaining HF part
        if ( np.sum( remainingOrbs ) != 0 ): 
            if ( self.CC_E_TYPE == 'CASCI' ):
                '''
                If CASCI is passed as CC energy type, the energy of the one and only full impurity Hamiltonian is returned.
                The one-electron integrals of this impurity Hamiltonian is the full Fock operator of the CORE orbitals!
                The constant part of the energy still needs to be added: sum_occ ( 2 * OEI[occ,occ] + JK[occ,occ] )
                                                                         = einsum( core1RDM_loc, OEI ) + 0.5 * einsum( core1RDM_loc, JK )
                                                                         = 0.5 * einsum( core1RDM_loc, OEI + FOCK )
                '''
                assert( maxiter == 1 )
                transfo = np.eye( self.Norb, dtype=float )
                totalOEI  = self.ints.dmet_oei(  transfo, self.Norb )
                totalFOCK = self.ints.dmet_fock( transfo, self.Norb, core1RDM_loc )
                self.energy += 0.5 * np.einsum( 'ij,ij->', core1RDM_loc, totalOEI + totalFOCK )
                Nelectrons = np.trace( self.imp_1RDM[ 0 ] ) + np.trace( core1RDM_loc ) # Because full active space is used to compute the energy
            else:
                assert (np.array_equal(self.ints.active, np.ones([self.ints.mol.nao_nr()], dtype=int)))
                from pyscf import scf
                from types import MethodType
                mol_ = self.ints.mol
                mf_  = scf.RHF(mol_)
                impOrbs = remainingOrbs==1
                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                hc  = -chempot_imp * np.dot(xorb[:,impOrbs], xorb[:,impOrbs].T)
                dm0 = np.dot(self.ints.ao2loc, np.dot(OneRDM, self.ints.ao2loc.T))

                def mf_hcore (self, mol=None):
                    if mol is None: mol = self.mol
                    return scf.hf.get_hcore(mol) + hc
                mf_.get_hcore = MethodType(mf_hcore, mf_)
                mf_.scf(dm0)
                assert (mf_.converged)

                rdm1 = mf_.make_rdm1()
                jk   = mf_.get_veff(dm=rdm1)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                rdm1 = np.dot(xorb.T, np.dot(rdm1, xorb))
                oei  = np.dot(self.ints.ao2loc.T, np.dot(mf_.get_hcore()-hc, self.ints.ao2loc))
                jk   = np.dot(self.ints.ao2loc.T, np.dot(jk, self.ints.ao2loc))

                ImpEnergy = \
                   + 0.50 * np.einsum('ji,ij->', rdm1[:,impOrbs], oei[impOrbs,:]) \
                   + 0.50 * np.einsum('ji,ij->', rdm1[impOrbs,:], oei[:,impOrbs]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[:,impOrbs], jk[impOrbs,:]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[impOrbs,:], jk[:,impOrbs])
                self.energy += ImpEnergy
                Nelectrons += np.trace(rdm1[np.ix_(impOrbs,impOrbs)])

            remainingOrbs[ remainingOrbs==1 ] -= 1
        assert( np.all( remainingOrbs == 0 ) )            
        self.energy += self.ints.const()
        print('total energy')
        print(self.energy)
        return Nelectrons

    def doselfconsistent( self ):
    
        iteration = 0
        u_diff = 1.0
        convergence_threshold = 1e-5
        print("RHF energy =", self.ints.fullEhf)
        
        while ( u_diff > convergence_threshold ):
        
            iteration += 1
            print("DMET iteration", iteration)
            umat_old = np.array( self.umat, copy=True )
            rdm_old = self.transform_ed_1rdm() # At the very first iteration, this matrix will be zero
            
            # Find the chemical potential for the correlated impurity problem
            start_ed = time.time()
            if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )
            else:
                self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp ,tol = 1e-8,maxiter=50,disp=False)
                print("   Chemical potential =", self.mu_imp)
            stop_ed = time.time()
            self.time_ed += ( stop_ed - start_ed )
            print("   Energy =", self.energy)
            # self.verify_gradient( self.square2flat( self.umat ) ) # Only works for self.doSCF == False!!
            if ( self.SCmethod != 'NONE' and not(self.altcostfunc) ):
                self.hessian_eigenvalues( self.square2flat( self.umat ) )
            
            # Solve for the u-matrix
            start_cf = time.time()
            if ( self.altcostfunc and self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.alt_costfunction, self.square2flat( self.umat ), jac=self.alt_costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            elif ( self.SCmethod == 'LSTSQ' ):
                result = optimize.leastsq( self.rdm_differences, self.square2flat( self.umat ), Dfun=self.rdm_differences_derivative, factor=0.1 )
                self.umat = self.flat2square( result[ 0 ] )
            elif ( self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.costfunction, self.square2flat( self.umat ), jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
            if ( self.altcostfunc ):
                print("   Cost function after convergence =", self.alt_costfunction( self.square2flat( self.umat ) ))
            else:
                print("   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) ))
            stop_cf = time.time()
            self.time_cf += ( stop_cf - start_cf )
            
            # Possibly print the u-matrix / 1-RDM
            if self.print_u:
                self.print_umat()
            if self.print_rdm:
                self.print_1rdm()
            
            # Get the error measure
            u_diff   = np.linalg.norm( umat_old - self.umat )
            rdm_diff = np.linalg.norm( rdm_old - self.transform_ed_1rdm() )
            self.umat = self.relaxation * umat_old + ( 1.0 - self.relaxation ) * self.umat
            print("   2-norm of difference old and new u-mat =", u_diff)
            print("   2-norm of difference old and new 1-RDM =", rdm_diff)
            print("******************************************************")
            
            if ( self.SCmethod == 'NONE' ):
                u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print("Time cf func =", self.time_func)
        print("Time cf grad =", self.time_grad)
        print("Time dmet ed =", self.time_ed)
        print("Time dmet cf =", self.time_cf)
        
        return self.energy