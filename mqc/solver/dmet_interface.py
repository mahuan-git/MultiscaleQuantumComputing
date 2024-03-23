import numpy
from openfermion import FermionOperator, normal_ordered
from openfermion.linalg import get_sparse_operator

from vqechem.tools import s2_operator, mapping
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from vqechem.vqe_opt import optimizer

def get_1e_2e_integral(dmet0):
    '''CAS sapce one-electron hamiltonian

    Args:
        scf0 : a scfinit object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    n_orb = dmet0._n_orb
    one_body_int = numpy.zeros([n_orb * 2] * 2)
    two_body_int = numpy.zeros([n_orb * 2] * 4)


    for p in range(n_orb):
        for q in range(n_orb):
            one_body_int[2 * p][2 * q] = dmet0._one_body_mo[p][q]
            one_body_int[2 * p + 1][2 * q + 1] = dmet0._one_body_mo[p][q]
            for r in range(n_orb):
                for s in range(n_orb):
                    two_body_int[2 * p][2 * q][2 * r][2 * s] = \
                        dmet0._two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = \
                        dmet0._two_body_mo[p][s][q][r]
                    two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = \
                        dmet0._two_body_mo[p][s][q][r]
                    two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = \
                        dmet0._two_body_mo[p][s][q][r]
    return one_body_int , two_body_int


class DMET():
    ''' Prepare Hamiltonian for VQE using PySCF
    '''
    def __init__(
            self,
            oei_mo,
            one_body_mo,
            two_body_mo,
            n_imp_elec : int,
            n_imp_orbs :int,
            chempot_imp :float,
            options):
        ''' construct hamiltonian with integrals from dmet

        '''
        if chempot_imp == None:
            chempot_imp = 0.0
        self._chempot_imp = chempot_imp
        self._oei_mo = oei_mo
        self._one_body_mo = one_body_mo
        self._two_body_mo = two_body_mo
        self._n_orb = one_body_mo.shape[0]
        self._n_imp_elec = n_imp_elec
        self._n_imp_orbs = n_imp_orbs ## number of impurity orbitals (n_orb = n_imp_orbs + n_bath_orbs)

        self._name = 'dmet'
        # mapping from fermion to qubit
        self._mapping = options['mapping'].upper() 
        # number of active orbitals
        ncas = one_body_mo.shape[0]
        # number of core orbitals
        ncore = 0
        # list of active orbitals
        mo_list = options['mo_list']
        '''currently please only do full active space
        '''
        # penalty coefficient for spin constraint
        shift = options['shift']

        noa = n_imp_elec//2
        nob = n_imp_elec - noa #!!!
        self._Enuc = 0
        #self._nbasis =           # number of basis sets
        if mo_list is None:
            if ncas is None: ncas = one_body_mo.shape[0]
            if ncore is None: ncore = 0
            mo_list = [i for i in range(ncore,ncore+ncas)]
        self._mo_act_list = mo_list
        self._mo_core_list = []
        for i in range(0,noa):
            if i not in mo_list:
                self._mo_core_list.append(i)

        self._ncore = len(self._mo_core_list)
        self._ncas = len(mo_list)
        self._noa = noa - self._ncore        # number of active occupied orbitals (alpha)
        self._nob = nob - self._ncore        # number of active occupied orbitals (beta)
        self._nva = self._ncas - self._noa
        self._nvb = self._ncas - self._nob
        self._nqubit = self._ncas*2
        self._mol = None

        self._hf = None
        self._mo_coeff = None
        self._nmo = None

        self._h1, self._h2 = get_1e_2e_integral(self)
        self._fermi_s2, Sz = s2_operator(self._ncas)
        self._qubit_s2 = mapping(self._fermi_s2, self._mapping, self._nqubit, self._noa*2)
        self._s2 = get_sparse_operator(self._qubit_s2)

        self._fermi_ham_bare = self.hamiltonian()

        self._qubit_ham_bare = mapping(self._fermi_ham_bare, self._mapping, self._nqubit, self._noa*2)
        self._ham_bare = get_sparse_operator(self._qubit_ham_bare)

        self._fermi_ham = self._fermi_ham_bare + shift/2 * self._fermi_s2
        self._qubit_ham = mapping(self._fermi_ham, self._mapping, self._nqubit, self._noa*2)    
        self._ham = get_sparse_operator(self._qubit_ham)
        if self._mapping == 'SCBK':
            self._ncas = self._ncas - 1
            self._nqubit = self._nqubit - 2

        self.print_scf_info()


    def hamiltonian_qubit(self, shift=0):
        ''' Build second quantazied Hamiltonian
            H = \sum_{pq} h1_{pq} a_p^+ a_q 
              + 1/2 \sum_{pqrs} h2_{psqr} a_p^+ a_q^+ a_r a_s
        '''
        chempot_imp=self._chempot
        n_orb = self._ncas
        nqubit = self._ncas*2
        one_body_mo = self._one_body_mo
        two_body_mo = self._two_body_mo
        one_body_int = numpy.zeros([n_orb * 2] * 2)
        two_body_int = numpy.zeros([n_orb * 2] * 4)


        print("Constructing qubit hamiltonian...")

        for p in range(n_orb):
            for q in range(n_orb):
                one_body_int[2 * p][2 * q] = one_body_mo[p][q]
                one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
                for r in range(n_orb):
                    for s in range(n_orb):
                        two_body_int[2 * p][2 * q][2 * r][2 * s] = \
                            two_body_mo[p][s][q][r]
                        two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = \
                            two_body_mo[p][s][q][r]
                        two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = \
                            two_body_mo[p][s][q][r]
                        two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = \
                            two_body_mo[p][s][q][r]


        # The h_ij a^+_i a_j term
        hamiltonian_fermOp_1 = FermionOperator()
        # The h_ijkl a^+_i a^+_j a_k a_l term
        hamiltonian_fermOp_2 = FermionOperator()

        for p in range(n_orb * 2):
            for q in range(n_orb * 2):
                hamiltonian_fermOp_1 += FermionOperator(
                ((p, 1), (q, 0)),
                (one_body_int[p][q])
                )

        for p in range(n_orb * 2):
            for q in range(n_orb * 2):
                for r in range(n_orb * 2):
                    for s in range(n_orb * 2):
                        hamiltonian_fermOp_2 += FermionOperator(
                            ((p, 1), (q, 1), (r, 0), (s, 0)),
                            two_body_int[p][q][r][s] * 0.5
                        )
        hamiltonian_fermOp_new =FermionOperator()
        if (chempot_imp!=0.0):
            for p in range(self._n_imp_orbs*2):
                hamiltonian_fermOp_new += FermionOperator(
                (( p , 1 ), ( p , 0 )),
                (one_body_int[p][p]-chempot_imp)
                )
        hamiltonian_fermOp_1.terms.update(hamiltonian_fermOp_new.terms)
        hamiltonian_fermOp_1 = normal_ordered(hamiltonian_fermOp_1)
        hamiltonian_fermOp_2 = normal_ordered(hamiltonian_fermOp_2)
        ham = hamiltonian_fermOp_1 + hamiltonian_fermOp_2
        if (shift != 0):
            ham = ham + shift * s2_operator(self._ncas)[0]

        return normal_ordered(ham)


    def hamiltonian(self,shift=0):
        ''' Build second quantazied Hamiltonian
            H = \sum_{pq} h1_{pq} a_p^+ a_q 
              + 1/2 \sum_{pqrs} h2_{psqr} a_p^+ a_q^+ a_r a_s
        '''
        n_orb = self._n_orb
        #ham = FermionOperator((),self._Enuc)
        hamiltonian_fermOp_1 = FermionOperator()
        # The h_ijkl a^+_i a^+_j a_k a_l term
        hamiltonian_fermOp_2 = FermionOperator()
        chempot_imp = self._chempot_imp
        for p in range(n_orb * 2):
            for q in range(n_orb * 2):
                hamiltonian_fermOp_1 += FermionOperator(
                ((p, 1), (q, 0)),
                (self._h1[p][q])
                )

        for p in range(n_orb * 2):
            for q in range(n_orb * 2):
                for r in range(n_orb * 2):
                    for s in range(n_orb * 2):
                        hamiltonian_fermOp_2 += FermionOperator(
                            ((p, 1), (q, 1), (r, 0), (s, 0)),
                            self._h2[p][q][r][s] * 0.5
                        )
        hamiltonian_fermOp_new =FermionOperator()
        if (chempot_imp!=0.0):
            for p in range(self._n_imp_orbs*2):
                hamiltonian_fermOp_new += FermionOperator(
                (( p , 1 ), ( p , 0 )),
                (self._h1[p][p]-chempot_imp)
                )
        hamiltonian_fermOp_1.terms.update(hamiltonian_fermOp_new.terms)
        hamiltonian_fermOp_1 = normal_ordered(hamiltonian_fermOp_1)
        hamiltonian_fermOp_2 = normal_ordered(hamiltonian_fermOp_2)

        ham = hamiltonian_fermOp_1 + hamiltonian_fermOp_2
        if shift != 0:
            ham = ham + shift * s2_operator(self._nmo)[0]

        return normal_ordered(ham)

    def fci(self):
        '''
        '''
        hamiltonian = self._ham + 0.5*self._s2
        [e,v] = eigsh(hamiltonian, 6, which='SA')
        for i in range(len(e)):
            S2 = v[:,i].conj().T.dot(self._s2.dot(v[:,i])).real
            print("    State %4i: %12.8f au  <S2>: %12.8f" %(i,e[i],S2))
        return e
        
    def print_scf_info(self):
        #Escf = self._hf.e_tot
        Enuc = self._Enuc 
        #nbasis = self._nbasis
        nmo = self._ncas
        nelec = self._n_imp_elec
        print('******************************************************')
        print('                   SCF Information                    ')
        print('                                                      ')
        #print('          Number of basis sets      :%15i'   %nbasis   )
        print('          Number of active orbitals :%15i'   %nmo      )
        print('          Number of active electron :%15i'   %nelec    )
        #print('          SCF energy                :%15.8f' %Escf     )
        print('                                                      ')
        print('                    FCI Energy:                       ')
        self._ref_energy = self.fci()
        print('******************************************************')
        print('\n')


class imp_optimizer(optimizer):
    '''vqe optimizer for dmet impurities
    '''
    def run(self, ansatz, params=None):
        ''' Run Iterative VQE Optimization Procedure
        '''
        if params is None: 
            if ansatz.get_nparams_opt() == 0:
                params = []
            else:
                params = ansatz._params
        params = numpy.asarray(params).flatten()

        options = {'gtol':self._cgtol, 'disp':False, 'maxiter':self._cgiter}
        # Iterative VQE procedure
        print('\n\n Begin Iterative VQE Optimization Procedure')
        for niter in range(self._maxiter):
            print(" ********************************************* ")
            print("              VQE Iteration %4d              " %niter)
            print(" ")
            params = ansatz.update(params, niter)
            print(" Iteration        Energy           Max Gradient    ")
            result = minimize(ansatz.energy, params, method=self._method, jac=ansatz.gradient,
                              options=options, callback=ansatz.callback)
            if ansatz._niter > 4:
                print("\n %4i  %20.10f        %10.1e" %(ansatz._niter,ansatz._energy,ansatz._max_grad))
            params = result['x']
            converged = self.check_converged(ansatz,result)
            if self._print_ham > 0:
                print(' Number of  Hamiltonian terms  = %12d'   %(ansatz.get_number_of_hamiltonian_terms(result['x'])))
            if converged:
                self.print_info(ansatz,result)
                break
            print(" ")
            print(" ")

        if not converged:
            print('Maximum number of iterations %4d is approached!!!' %self._maxiter)
            self.print_info(ansatz, result)

        return ansatz , params


