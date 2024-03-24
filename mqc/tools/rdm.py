from pyscf.fci.cistring import make_strings , cre_des_sign
from pyscf.fci.fci_slow import reorder_rdm
import numpy as np
import time
def make_strings_qubit(orb_list, nelec,Sz = 0):
    n_orbs = len(orb_list)
    #assert (all(orb_list[:-1] < orb_list[1:]))
    nelec_a = (nelec+Sz)/2
    nelec_b = nelec - nelec_a
    stra = make_strings(orb_list[::2],nelec_a)
    strb = make_strings(orb_list[1::2],nelec_b)
    strings = []
    for i in stra:
        for j in strb:
            strings.append(i+j)
    return np.asarray(strings, dtype=np.int64)

def gen_linkstr_index_qubit_v1(orb_list, nelec,Sz=0, strs=None):
    if strs is None:
        strs = make_strings_qubit(orb_list, nelec,Sz=Sz)
    strdic = dict(zip(strs,range(strs.__len__())))
    def propagate1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        linktab = []
        for i in occ:
            linktab.append((i, i, strdic[str0], 1))
        for i in occ:
            for a in vir:
                if i%2==a%2:
                    str1 = str0 ^ (1 << i) | (1 << a)
                    try:
                        addr = strdic[str1]
                        linktab.append((a, i, addr, cre_des_sign(a, i, str0)))
                    except KeyError:
                        continue
                    #linktab.append((a, i, addr, cre_des_sign(a, i, str0)))
                else:
                    pass
        return linktab

    t = [propagate1e(s) for s in strs.astype(np.int64)]
    return t
    #return np.array(t, dtype=np.int32)

def expand_strings(n_orb, strs):
    orb_list = range(2*n_orb)
    def expand(str0):
        strs_list = [str0]
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        for i in occ:
            for a in vir:
                if i%2==a%2:
                    str1 = str0 ^ (1 << i) | (1 << a)
                    strs_list.append(str1)
        return strs_list
    strs_set = set()
    for s in strs.astype(np.int64):
        strs_list = expand(s)
        for string in strs_list:
            strs_set.add(string)
    strs_new = list(strs_set)
    return strs_new

def gen_linkstr_index_qubit(orb_list, nelec,Sz=0, strs_all = None,strs=None):
    if strs_all is None:
        strs_all = make_strings_qubit(orb_list, nelec,Sz=Sz)
    if strs is None:
        strs = strs_all
    strdic = dict(zip(strs_all,range(strs_all.__len__())))
    idx = []
    for str0 in strs:
        idx.append(strdic[str0])
    strdic_1 = dict(zip(strs,idx))
    def propagate1e(str0):
        occ = []
        vir = []
        for i in orb_list:
            if str0 & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        linktab = []
        for i in occ:
            linktab.append((i, i, strdic[str0], 1))
        for i in occ:
            for a in vir:
                if i%2==a%2:
                    str1 = str0 ^ (1 << i) | (1 << a)
                    try:
                        addr = strdic_1[str1]
                        linktab.append((a, i, addr, cre_des_sign(a, i, str0)))
                    except KeyError:
                        continue
                else:
                    pass
        return linktab

    t = [propagate1e(s) for s in strs_all.astype(np.int64)]
    return t
    #return np.array(t, dtype=np.int32)

def make_rdm1(fcivec,norb,nelec,Sz=0,strs = None,opt=None):
    link_index = gen_linkstr_index_qubit_v1(range(2*norb), nelec,Sz=Sz,strs = strs)
    rdm1 = np.zeros((norb,norb),dtype ="complex64")
    for str0 , tab in enumerate(link_index):
        for a,i,str1,sign in tab:
            rdm1[a//2][i//2]+=sign * fcivec[str0].conj()*fcivec[str1]
    rdm1 = np.array(rdm1,dtype = "float32")
    return rdm1

def make_rdm1s(fcivec,norb,nelec,Sz=0,strs = None,opt=None):
    link_index = gen_linkstr_index_qubit_v1(range(2*norb), nelec,Sz=Sz,strs = strs)
    rdm1a = np.zeros((norb,norb),dtype ="complex64")
    rdm1b = np.zeros((norb,norb),dtype ="complex64")
    for str0 , tab in enumerate(link_index):
        for a,i,str1,sign in tab:
            if a%2 == i%2 == 0:
                rdm1a[a//2][i//2]+=sign * fcivec[str0].conj()*fcivec[str1]
            elif a%2 == i%2 == 1:
                rdm1b[a//2][i//2]+=sign * fcivec[str0].conj()*fcivec[str1]
            else:
                raise ValueError("configure incorrect")
    rdm1a = np.array(rdm1a,dtype = "float32")
    rdm1b = np.array(rdm1b,dtype = "float32")
    return rdm1a, rdm1b


def make_rdm2(fcivec, norb, nelec,Sz=0, strs = None,opt=None):
    rdm1 = make_rdm1(fcivec=fcivec,norb=norb,nelec=nelec,Sz=Sz,strs = strs,opt=opt)
    print("start making link_index")
    time0 = time.perf_counter()
    link_index = gen_linkstr_index_qubit(range(2*norb), nelec,Sz=Sz,strs = strs)
    time1 = time.perf_counter()
    print("end making link_index, time used: ", (time1 - time0))
    #rdm1 = np.zeros((norb,norb),dtype = "complex64")
    rdm2 = np.zeros((norb,norb,norb,norb),dtype = "complex64")
    for str0, tab in enumerate(link_index):
        #t1 = np.zeros((norb,norb),dtype = "complex64")
        for r, s, str1, sign1 in link_index[str0]:
            for p, q, str2, sign2 in link_index[str1]:
                rdm2[p//2,q//2,r//2,s//2] += sign1*sign2*fcivec[str0].conj() * fcivec[str2]
        #rdm1+=fcivec[str0].conj()*t1
        #rdm2 += np.einsum('ij,kl->jikl', t1.conj(), t1)
        #rdm2 += np.einsum('ij,kl->jkil', t1.conj(), t1)
        #rdm2 +=np.outer(t1.conj(),t1).reshape(norb,norb,norb,norb)
    #rdm1 = rdm1.astype("float32")
    rdm2 = rdm2.astype("float32")
    #return rdm2
    return reorder_rdm(rdm1, rdm2)

def make_rdm12_v1(fcivec, norb, nelec,Sz=0, strs = None,opt=None):
    print("start making link_index")
    time0 = time.perf_counter()
    link_index = gen_linkstr_index_qubit(range(2*norb), nelec,Sz=Sz,strs = strs)
    time1 = time.perf_counter()
    print("end making link_index, time used: ", (time1 - time0))
    rdm1 = np.zeros((norb,norb),dtype = "complex64")
    rdm2 = np.zeros((norb,norb,norb,norb),dtype = "complex64")
    for str0, tab in enumerate(link_index):
        t1 = np.zeros((norb,norb),dtype = "complex64")
        for a, i, str1, sign in link_index[str0]:
            t1[a//2,i//2] += sign * fcivec[str1]
        rdm1+=fcivec[str0].conj()*t1
        rdm2 += np.einsum('ij,kl->jikl', t1.conj(), t1)
        #rdm2 += np.einsum('ij,kl->jkil', t1.conj(), t1)
        #rdm2 +=np.outer(t1.conj(),t1).reshape(norb,norb,norb,norb)
    rdm1 = rdm1.astype("float32")
    rdm2 = rdm2.astype("float32")
    return reorder_rdm(rdm1, rdm2)


def make_rdm12(fcivec, norb, nelec,Sz=0, strs_all = None,strs = None,opt=None):
    print("start making link_index")
    time0 = time.perf_counter()
    link_index = gen_linkstr_index_qubit(range(2*norb), nelec,Sz=Sz,strs_all = strs_all,strs = strs)
    time1 = time.perf_counter()
    print("end making link_index, time used: ", (time1 - time0))
    rdm1 = np.zeros((norb,norb),dtype = "complex64")
    rdm2 = np.zeros((norb,norb,norb,norb),dtype = "complex64")
    for str0, tab in enumerate(link_index):
        t1 = np.zeros((norb,norb),dtype = "complex64")
        for a, i, str1, sign in link_index[str0]:
            t1[a//2,i//2] += sign * fcivec[str1]
        rdm1+=fcivec[str0].conj()*t1
        rdm2 += np.einsum('ij,kl->jikl', t1.conj(), t1)
        #rdm2 += np.einsum('ij,kl->jkil', t1.conj(), t1)
        #rdm2 +=np.outer(t1.conj(),t1).reshape(norb,norb,norb,norb)
    rdm1 = rdm1.astype("float32")
    rdm2 = rdm2.astype("float32")
    return reorder_rdm(rdm1, rdm2)


def make_rdm12s(fcivec, norb, nelec,Sz=0, strs_all = None,strs = None,opt=None):
    print("start making link_index")
    link_index = gen_linkstr_index_qubit(range(2*norb), nelec,Sz=Sz,strs_all = strs_all,strs = strs)
    #na = len(link_index)
    #fcivec = fcivec.reshape(na,na)
    print("end making link_index")
    rdm1a = np.zeros((norb,norb),dtype = "complex64")
    rdm1b = np.zeros((norb,norb),dtype = "complex64")

    rdm2aa = np.zeros((norb,norb,norb,norb),dtype = "complex64")
    rdm2bb = np.zeros((norb,norb,norb,norb),dtype = "complex64")
    rdm2ab = np.zeros((norb,norb,norb,norb),dtype = "complex64")

    for str0, tab in enumerate(link_index):
        t1 = np.zeros((norb,norb),dtype = "complex64")
        t2 = np.zeros((norb,norb),dtype = "complex64")
        for a, i, str1, sign in link_index[str0]:
            if a%2 == i%2==0:
                t1[i//2,a//2] += sign * fcivec[str1]
            elif a%2 == i%2 == 1:
                t2[i//2,a//2] += sign * fcivec[str1]
            else:
                raise ValueError("configure incorrect")
        rdm1a += fcivec[str0].conj()*t1
        rdm1b += fcivec[str0].conj()*t2
        rdm2aa += np.einsum('ij,kl->jikl', t1.conj(), t1)
        rdm2bb += np.einsum('ij,kl->jikl', t2.conj(), t2)
        rdm2ab += np.einsum('ij,kl->jikl', t1.conj(), t2)
        #rdm2aa  += np.outer(t1.conj(),t1).reshape(norb,norb,norb,norb)
        #rdm2bb  += np.outer(t2.conj(),t2).reshape(norb,norb,norb,norb)
        #rdm2ab  += np.outer(t1.conj(),t2).reshape(norb,norb,norb,norb)
    rdm1a, rdm2aa = reorder_rdm(rdm1a, rdm2aa, inplace=True)
    rdm1b, rdm2bb = reorder_rdm(rdm1b, rdm2bb, inplace=True)
    rdm1a = rdm1a.astype("float32")
    rdm1b = rdm1b.astype("float32")
    rdm2aa = rdm2aa.astype("float32")
    rdm2bb = rdm2bb.astype("float32")
    rdm2ab = rdm2ab.astype("float32")
    return rdm1a,rdm1b,rdm2aa,rdm2bb,rdm2ab
