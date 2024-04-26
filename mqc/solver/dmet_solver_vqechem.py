import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy

from mqc.solver.dmet_interface import DMET , imp_optimizer
from vqechem.set_options import set_options
from vqechem.fermion_operator import FermionOps
from vqechem.qubit_operator import QubitOps
from vqechem.hamiltonian_operator import HamiltonianOps
from vqechem.algorithms import reference
from vqechem.ansatz_pool import ADAPT
import time
import openfermion
from mqc.tools.rdm import make_strings_qubit, make_rdm1,make_rdm12,make_rdm12s

def solve(
        oei_mo,
        one_body_mo,
        two_body_mo,
        n_imp: int, ## number of electrons in the impurity
        n_imp_orbs: int,  ## number of orbitals in the impurity
        chempot_imp = 0.0,
        skip_vqe: bool = True):
    n_imp = n_imp
    start = time.clock()
    # Integrals in spin-orbitals (n_orb * 2)
    n_orb = one_body_mo.shape[0]
    n_qubits = n_orb * 2
    print('running %d qubit vqechem calculation'%n_qubits)
    ''' main driver for iterative variational quantum algorithms
        we need to prepare the reference state and operator pool
        for constructing wave function Ansatz.
    '''
    options = {
               'vqe' : {'algorithm':'adapt-vqe'},
               'scf' : {'shift':0.5},
               'ops' : {'class':'fermionic','spin_sym':'sa'},
               'ansatz' : {'method':'adapt','form':'unitary','Nu':10},
               'opt' : {'maxiter':300, 'tol':0.001}
              }
    options = set_options(options)
    imp = DMET(
        oei_mo,
        one_body_mo,
        two_body_mo,
        n_imp,
        n_imp_orbs,
        chempot_imp ,
        options['scf'],
        )
    ops_class = options['ops']['class'].lower()
    if ops_class=='fermionic':
        pool = FermionOps(imp, options['ops'])
    elif ops_class=='qubit':
        pool = QubitOps(imp, options['ops'])
    elif ops_class=='hamiltonian':
        pool = HamiltonianOps(imp, options['ops'])
    else:
        print('Incorrect operator pool')
        exit()

    ref = reference(imp, options)

    ansatz = ADAPT(options['ansatz'], imp, pool, ref)

    opt = imp_optimizer(options)

    ansatz , params = opt.run(ansatz)
    time1= time.clock()
    wfn = ansatz.state(params)

    #print(wfn)
    rdm1,rdm2 = get_rdm12(wfn,n_imp,n_orb,Sz = 0)
    imp_energy = 0.0
    imp_energy += 0.5 * numpy.einsum(
        'ij,ij->',
        rdm1[:n_imp_orbs, :], oei_mo[:n_imp_orbs, :] + one_body_mo[:n_imp_orbs, :])
    imp_energy += 0.5 * numpy.einsum(
        'ijkl,ijkl->',
        rdm2[:n_imp_orbs, :, :, :], two_body_mo[:n_imp_orbs, :, :, :])
    time2 = time.clock()
    print("Impurity energy: %20.16f" % (imp_energy))
    print("one-rdm: \n", rdm1)
    #print("2-rdm: \n", rdm2)
    print('time for vqechem calculation = %f'%(time1-start))
    print('time for constructing rdm1 and rdm2 = %f '%(time2-time1))
    return imp_energy, rdm1

def qubit_inverse(norb,idx):
    nqubits = 2*norb
    bin_idx = bin(idx)
    bin_idx = bin_idx[2:]
    ndigits = len(bin_idx)
    bin_idx = (nqubits - ndigits)*"0" +bin_idx
    bin_idx_rev = bin_idx[::-1]
    new_idx = int(bin_idx_rev,2)
    return new_idx 

def get_rdm1(wfn,nelec, norb,Sz = 0):
    strings= make_strings_qubit(norb=norb,nelec = nelec, Sz = Sz)
    fcivec = []
    for idx in strings:
        idx = qubit_inverse(norb,idx)
        fcivec.append(wfn[idx])
    rdm1 = make_rdm1(fcivec=fcivec , norb = norb, nelec = nelec,Sz = Sz)
    return rdm1

def get_rdm12(wfn,nelec,norb,Sz = 0):
    strings= make_strings_qubit(norb=norb,nelec = nelec, Sz = Sz)
    fcivec = []
    for idx in strings:
        idx = qubit_inverse(norb,idx)
        fcivec.append(wfn[idx,0])
    rdm1,rdm2 = make_rdm12(fcivec=fcivec , norb = norb, nelec = nelec,Sz = Sz,strs = strings)
    return rdm1 , rdm2


def get_rdm12_openfermion(wfn,n_orb):
    n_qubits = n_orb*2
    wfn =wfn.toarray() ##this may encounter an oom error
    S=wfn.conj().T.dot(wfn)[0,0]

    rdm1 = numpy.zeros([n_orb] * 2)
    rdm2 = numpy.zeros([n_orb] * 4)
    for p in range(0, n_orb * 2, 2):
        for q in range(0, n_orb * 2, 2):
            rdm1_op = openfermion.FermionOperator(
                ((p, 1), (q, 0)),
                1.0
            )
            # Here we use matrix operations. For quantum circuit simulations,
            # this can be done through measurements.
            rdm1_sparse_mat = openfermion.get_sparse_operator(
                rdm1_op, n_qubits=n_qubits)
            rdm1[p // 2][q // 2] = wfn.T.conj().dot(
                rdm1_sparse_mat.dot(wfn))[0, 0] * 2 / S

            for r in range(0, n_orb * 2, 2):
                for s in range(0, n_orb * 2, 2):
                    rdm2_op = openfermion.FermionOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)),
                        1.0
                    )
                    rdm2_sparse_mat = openfermion.get_sparse_operator(
                        rdm2_op, n_qubits=n_qubits)
                    rdm2[p // 2][s // 2][q // 2][r // 2] += wfn.T.conj().dot(
                        rdm2_sparse_mat.dot(wfn))[0, 0]/S

                    rdm2_op = openfermion.FermionOperator(
                        ((p + 1, 1), (q + 1, 1), (r + 1, 0), (s + 1, 0)),
                        1.0
                    )
                    rdm2_sparse_mat = openfermion.get_sparse_operator(
                        rdm2_op, n_qubits=n_qubits)
                    rdm2[p // 2][s // 2][q // 2][r // 2] += wfn.T.conj().dot(
                        rdm2_sparse_mat.dot(wfn))[0, 0]/S

                    rdm2_op = openfermion.FermionOperator(
                        ((p + 1, 1), (q, 1), (r, 0), (s + 1, 0)),
                        1.0
                    )
                    rdm2_sparse_mat = openfermion.get_sparse_operator(
                        rdm2_op, n_qubits=n_qubits)
                    rdm2[p // 2][s // 2][q // 2][r // 2] += wfn.T.conj().dot(
                        rdm2_sparse_mat.dot(wfn))[0, 0]/S

                    rdm2_op = openfermion.FermionOperator(
                        ((p, 1), (q + 1, 1), (r + 1, 0), (s, 0)),
                        1.0
                    )
                    rdm2_sparse_mat = openfermion.get_sparse_operator(
                        rdm2_op, n_qubits=n_qubits)
                    rdm2[p // 2][s // 2][q // 2][r // 2] += wfn.T.conj().dot(
                        rdm2_sparse_mat.dot(wfn))[0, 0]/S

            print("Finished %d/%d..." % (1 + p//2 * n_orb + q//2, n_orb ** 2),
                  end="\r")
    return rdm1,rdm2
