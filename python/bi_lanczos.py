#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:35:19 2024

@author: maitri
"""

from __future__ import print_function, division

#

import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3

os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel

os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

#

quspin_path = os.path.join(os.getcwd(),"../../")

sys.path.insert(0,quspin_path)



from quspin.operators import hamiltonian

from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d

from quspin.basis.user import user_basis # Hilbert space user basis

from quspin.basis.user import pre_check_state_sig_32,op_sig_32,map_sig_32 # user basis data types

from numba import carray,cfunc # numba helper functions

from numba import uint32,int32 # numba data types

import numpy as np

import matplotlib.pyplot as plt

import math

import cmath

from scipy.linalg import expm

np.object = object


#### THE MODEL AND HAMILTONIAN

N = 10

basis = spin_basis_1d(L=N,a=1,kblock=0)


J = 1
hx = -1.0
hz = 0.0
g = hx

h = hz

term1 = [[-J,i,(i+1)%N] for i in range(N)] #PBC

term2 = [[-g,i] for i in range(N)] # PBC

term3 = [[-h,i] for i in range(N)] # PBC


# static and dynamic lists

static = [["zz",term1],["x",term2],["z",term3]]

dynamic=[]

H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)

H_dag = np.conjugate(H.toarray()).T

E,V = H.eigh()

H = H.toarray()

dim= basis.Ns

###-----------XX-----------------------

### THE MODEL AND HAMILTONIAN-II







###---------------------------XXX----------------------------------------------




###-------------EIGENVALUE SPECTRUM AND EIGENSTATE IPR-------------------------


#eigenvalue spectrum plot

reE = E.real
imE = E.imag

plt.figure()

plt.plot(reE,imE)

plt.title('Complex Energy Spectra')

plt.show()




#eigenstate IPR

eig_IPR_list = []

for i in range(dim):
    
    v = V.T[i]
    
    ipr = np.sum(np.abs(v)**4)
    
    eig_IPR_list.append(ipr)

#plot IPR

plt.figure()

plt.plot(np.arange(0,dim,1),eig_IPR_list)

plt.title('Eigenstate IPR')

plt.show()



###------------------SuFF------------------------------------------------------



###----------Survival Form Factor (SuFF) at beta=0 :---------------------------
    
psi0 = np.zeros(dim,dtype=np.complex128)


for i in range(dim):
    
    psi0 += V.T[i]


psi0 = psi0/np.linalg.norm(psi0)



def act_U(psi,dt):
    
    psif = expm(-1j*dt*H) @ psi
    psif = psif/np.linalg.norm(psif)
    
    return psif


    
t0 = 0*dim
tf = 100*dim
dt = 0.01*dim
tList = np.arange(t0,tf,dt)
SuFFList = []

psitrunning = psi0

for t in tList:
    
    psit = act_U(psitrunning,dt)
    Samp = psit.T.conj() @ psi0
    SuFF = np.abs(Samp)**2
    SuFFList.append(SuFF)
    psitrunning = psit




plt.figure()
plt.plot(tList/dim,SuFFList)
plt.xscale('log')
plt.yscale('log')
plt.title(f'Survival Form Factor (SuFF), g={g}, h={h}')
plt.show()

    
##-----------------------------------------------------------------------------


#######--------------------COMPLEXITY------------------------------------------


###-------------------ARNOLDI ALGORITHM----------------------------------------

  #Krylov basis from Arnoldi

dim= basis.Ns


psi0 = np.zeros(dim,dtype=np.complex128)

for i in range(dim):
    
    psi0 += V.T[i]  # for TFD

psi0 = psi0/np.linalg.norm(psi0)  # for TFD

a0_Ar = psi0.T.conj() @ H @ psi0

an_List_Ar = [a0_Ar]
bn_List_Ar = [1]
kn_List_Ar = [psi0]

for n in range(dim): # Arnoldi iteration
    
    w = H @ kn_List_Ar[-1]
    
    proj = np.zeros(dim,dtype=np.complex128)
    
    for k in range(n):
        
        proj += (kn_List_Ar[k].T.conj() @ w) * kn_List_Ar[k]
        
    w = w - proj
    
    proj = np.zeros(dim,dtype=np.complex128)
    
    for k in range(n):
        
        proj += (kn_List_Ar[k].T.conj() @ w) * kn_List_Ar[k]
        
    w = w - proj
    
    b_Ar = np.sqrt(w.T.conj() @ w)
    k_Ar = w/b_Ar
    a_Ar = k_Ar.T.conj() @ H @ k_Ar
    
    if b_Ar > 10**(-5):
        
        kn_List_Ar.append(k_Ar)
        bn_List_Ar.append(b_Ar)
        an_List_Ar.append(a_Ar)
        
    else:
        break
    
# plot Arnoldi bn

plt.figure()
plt.plot(np.arange(0,len(bn_List_Ar),1),bn_List_Ar)
plt.title(f'bn vs. n for Arnoldi Iteration, g={g},  h={h}')
plt.xlabel('n')
plt.ylabel('bn')
plt.show()

###----------------------------------------------------------------------------  


###----------------BI-LANCZOS ALGORITHM----------------------------------------

  # Constructing Krylov basis for bi-Lanczos

dim= basis.Ns


psi0 = np.zeros(dim,dtype=np.complex128)

for i in range(dim):
    
    psi0 += V.T[i]  # for TFD

psi0 = psi0/np.linalg.norm(psi0)  # for TFD

#psi0 = np.zeros(dim,dtype = np.complex128) 

#psi0[int(dim/2)] = 1  # for middle-1

a0 = np.conjugate(psi0).T @ H @ psi0

an_list = [a0]
bn_list = [0]
cn_list = [0]
pn_list = [psi0]
qn_list = [psi0]

for n in range(dim):
    if n==0:
        A1 = H@pn_list[0] - an_list[0] * pn_list[0]
        B1 = H_dag @ qn_list[0] - np.conjugate(an_list[0]) * qn_list[0]
        w1 = np.conjugate(A1).T @ B1
        c1 = np.sqrt(np.abs(w1))
        b1 = np.conjugate(w1)/c1
        p1 = A1/c1
        q1 = B1/np.conjugate(b1)
        a1 = np.conjugate(q1).T @ H @ p1
        bn_list.append(b1)
        cn_list.append(c1)
        an_list.append(a1)
        pn_list.append(p1)
        qn_list.append(q1)
    else:
        A_np1 = H @ pn_list[n] - an_list[n]*pn_list[n] - bn_list[n]*pn_list[n-1]
        B_np1 = H_dag @ qn_list[n] - np.conjugate(an_list[n]) * qn_list[n] - np.conjugate(cn_list[n])*qn_list[n-1]
        
        projA = np.zeros(dim,dtype = np.complex128)
        projB = np.zeros(dim,dtype = np.complex128)
        
        for m in range(n+1):
            projA += (np.conjugate(qn_list[m]).T @ A_np1)*pn_list[m]
            projB += (np.conjugate(pn_list[m]).T @ B_np1)*qn_list[m]
            
        A_np1 = A_np1 - projA
        B_np1 = B_np1 - projB
        
        projA = np.zeros(dim,dtype = np.complex128)
        projB = np.zeros(dim,dtype = np.complex128)
        
        for m in range(n+1):
            projA += (np.conjugate(qn_list[m]).T @ A_np1)*pn_list[m]
            projB += (np.conjugate(pn_list[m]).T @ B_np1)*qn_list[m]
            
        A_np1 = A_np1 - projA
        B_np1 = B_np1 - projB
            
            
        
        w_np1 = np.conjugate(A_np1).T @ B_np1
        c_np1 = np.sqrt(np.abs(w_np1))
        b_np1 = np.conjugate(w_np1)/c_np1
        
        p_np1 = A_np1 / c_np1
        
        q_np1 = B_np1 / np.conjugate(b_np1)
        
        if c_np1>10**(-5):
            a_np1 = np.conjugate(q_np1).T @ H @ p_np1
            bn_list.append(b_np1)
            cn_list.append(c_np1)
            an_list.append(a_np1)
            pn_list.append(p_np1)
            qn_list.append(q_np1)
        else:
            break
        
plt.figure()
plt.plot(np.arange(0,len(cn_list),1),cn_list)
plt.title(f'cn vs. n for bi-Lanczos algorithm, g={g},  h={h}')
plt.xlabel('n')
plt.ylabel('cn')
plt.show()

        
###----------------------------------------------------------------------------            
        
        
###-------------------------COMPUTING COMPLEXITY-------------------------------
        
        
        
t0 = 0*dim
tf = 5*dim
dt = 0.01*dim
tList = np.arange(t0,tf,dt)

# for bi-Lanczos

complexityList_Bi = []
normList_Bi = []
KIPRList_Bi = []
KentList_Bi = []

#for Arnoldi

complexityList_Ar = []
normList_Ar = []
KIPRList_Ar = []
KentList_Ar = []


#initiate
psitrunning = psi0


for t in tList:
    
    psit = act_U(psitrunning, dt)
    psit = psit/(np.linalg.norm(psit))
    psitrunning = psit
    
    ## for bi-Lanczos
    
    l = len (qn_list)
    k = 0
    nor= 0
    KIPR = 0
    Kent = 0
    
    for n in range (l):
        phin_q = np.conjugate(qn_list[n]).T @ psit
        phin_p = np.conjugate(pn_list[n]).T @ psit
        k += n*np.abs(np.conjugate(phin_q) * phin_p)
        nor +=  np.abs(np.conjugate(phin_q) * phin_p)
    
    k = k/nor   
    
    for n in range (l):
        phin_q = (np.conjugate(qn_list[n]).T @ psit)
        phin_p = (np.conjugate(pn_list[n]).T @ psit)
        probn = np.abs(np.conjugate(phin_q) * phin_p)/nor
        KIPR += probn**2
        Kent +=  -probn*np.log(probn)
    
    normList_Bi.append(nor)
    complexityList_Bi.append(k)
    KIPRList_Bi.append(KIPR)
    KentList_Bi.append(Kent)
    
    
    ## for Arnoldi
    
    
    dimK = len(kn_List_Ar)
    complexity =  0
    norm = 0
    KIPR = 0
    Kent = 0
    
    for n in range(dimK):
        
        kn = kn_List_Ar[n]
        phin = kn.T.conj() @ psit
        probn = np.abs(phin)**2
        norm += probn
        complexity += n*probn
        KIPR += probn**2
        Kent += - probn*np.log(probn)
        
    
    
    normList_Ar.append(norm)
    complexityList_Ar.append(complexity)
    KIPRList_Ar.append(KIPR)
    KentList_Ar.append(Kent)
    

# for bi-Lanczos

complexityList_Bi = np.array(complexityList_Bi)
normList_Bi = np.array(normList_Bi)
KIPRList_Bi = np.array(KIPRList_Bi)
KentList_Bi = np.array(KentList_Bi)

#for Arnoldi

complexityList_Ar = np.array(complexityList_Ar)
normList_Ar = np.array(normList_Ar)
KIPRList_Ar = np.array(KIPRList_Ar)
KentList_Ar = np.array(KentList_Ar)







###----------PLOTTING COMPLEXITY-----------------------------------------------

plt.figure()
plt.plot(tList/dim,complexityList_Bi/dim)
plt.title(f'complexity vs. time for bi-Lanczos, g={g}, h= {h}')
plt.xlabel('t/dim')
plt.ylabel('C(t)/dim')
plt.show()

plt.figure()
plt.plot(tList/dim,complexityList_Bi/dim)
plt.title(f'complexity vs. time for Arnoldi, g={g}, h= {h}')
plt.xlabel('t/dim')
plt.ylabel('C(t)/dim')
plt.show()


###----------------------------------------------------------------------------












