#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:16:13 2024

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


L = 12

Delta = 1

def realization(w):
    random_numbers = np.random.uniform(-1,1,L)


    basis = spin_basis_1d(L, pauli=True,Nup=L//2)

    #basis = spin_basis_1d(L, pauli=True, m=0)

    term1 = [[1/2,i,(i+1)%L] for i in range(L)]
    term2 = [[1/2,i,(i+1)%L] for i in range(L)]
    term3 = [[Delta/2, i, (i+1)%L] for i in range(L)]
    onsite_pot = [[w*random_numbers[i],i] for i in range(L)]

    static = [["xx",term1],["yy",term2],["zz",term3],["z",onsite_pot]]

    dynamic=[]

    H0 = hamiltonian(static,dynamic,basis=basis,check_symm=False,dtype=np.complex128)

    E_full,V = H0.eigh()

    El = E_full[200:700]
    #---------------------------------- Check the level statistics--------------------------------------


    r_tilde_p =[]

    for i in range(len(El)-2):
        E1 = El[i]
        E2 = El[i+1]
        E3 = El[i+2]
        del1 = E2-E1
        del2 = E3-E2
        r_tilde = min(del1,del2)/max(del1,del2)
        r_tilde_p.append(r_tilde)
        
    E = E_full   
    psi0 = np.zeros(len(E))

    for v in (V.T):
        psi0 = psi0 + v
        
    dim = len(E)    
        
    '''psi0 = np.zeros(dim,dtype = np.complex128) 

    psi0[int(dim/2)] = 1 ''' 

    psi0 = psi0/(np.linalg.norm(psi0)) # initial TFD state at infinite temperature

    H0 = H0.toarray()

    anlist = [np.conjugate(psi0).T @ H0 @ psi0]

    bnlist = [1]

    knlist = [psi0]


    #----------------------------------------------------- computing the krylov basis---------------

    for n in np.arange(1,dim,1):
        An = H0 @ knlist[-1]
        summ =np.zeros(dim,dtype = np.complex128)
        
        for m in range(n):
            proj = np.conjugate(knlist[m]).T @ An
            summ += proj * knlist[m]
            
        An = An - summ
        
        summ =np.zeros(dim,dtype = np.complex128)
        
        for m in range(n):
            proj = np.conjugate(knlist[m]).T @ An
            summ += proj * knlist[m]
            
        An = An - summ
        
        bn = np.sqrt(np.conjugate(An).T @ An)
        
        kn = An/bn
        
        an = np.conjugate(kn).T @ H0 @ kn
        
        if bn < 10**-6:
            break
        else:
            anlist.append(an)
            bnlist.append(bn)
            knlist.append(kn)
        
    #---------------------------- Calculating the complexity-------------------------------    

    complexity_list = []

    norm_list = []
    
    krylov_entropy_list = []
    IPR_list = []

    tf = 1.5*dim
    dt = 0.01*dim
    num_steps =int( tf/dt)

    t_list = np.linspace(0,tf,num_steps)

    for i in range(num_steps):
        t = t_list[i]
        psit = expm(-1j*t*H0) @ psi0
        krylov_basis = knlist
        l  = len(krylov_basis)
        complexity = 0
        norm = 0
        entr = 0
        ipr = 0
        
        for n in range(l):
            phi_n = np.conjugate(krylov_basis[n]).T @ psit
            
            norm += np.abs(phi_n)**2
            complexity += n * np.abs(phi_n)**2
            pn = np.abs(phi_n)**2
            entr += -pn*np.log(pn)
            ipr += pn**2
            
            
            
            
        complexity_list.append(complexity)
        norm_list.append(norm)
        krylov_entropy_list.append(entr)
        IPR_list.append(ipr)
        
        
        
    n_list = np.arange(0,len(anlist),1)     
    return E_full,r_tilde_p,anlist,bnlist,np.array(complexity_list)/dim,t_list /dim ,np.array( krylov_entropy_list), np.array(IPR_list )
    


os.chdir("/home/maitri/kry/MBL_Data")

w_list = [0,0.01,0.1,0.2,0.5,1,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.5]

for w in w_list:
    r_tilde = []
    complexity =[]
    an_listi = []
    bn_listi = []
    krylov_entropy = []
    IPR = []


    R = 10


    for i in range(R):
        realization_p = realization(w)
        r_tilde_p = realization_p[1]
        r_tilde = r_tilde+r_tilde_p
        complexi = realization_p[4]
        complexity.append(complexi)
        an = realization_p[2]
        bn = realization_p[3]
        an_listi.append(an)
        bn_listi.append(bn)
        t_list = realization_p[5]
        krylov_entrop = realization_p[6]
        krylov_entropy.append(krylov_entrop)
        iprr = realization_p[7]
        IPR.append(iprr)
    complexity_list = (1/R) * sum(complexity) 
    krylov_entropy_list = (1/R)*sum(krylov_entropy)
    IPR_list = (1/R)*sum(IPR)
    
    np.savetxt("Delta1_lamb0_TFD/complexity_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),complexity_list)
    np.savetxt("Delta1_lamb0_TFD/krylov_entropy_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),krylov_entropy_list)
    np.savetxt("Delta1_lamb0_TFD/krylov_IPR_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),IPR_list)
    np.savetxt("Delta1_lamb0_TFD/t_list_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),t_list)
    np.savextxt("Delta1_lamb0_TFD/an_last_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),an_listi[-1])
    np.savextxt("Delta1_lamb0_TFD/bn_last_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),bn_listi[-1])
    np.savextxt("Delta1_lamb0_TFD/r_tilde_total_for_all_realization_Delta_{}_L_{}_w_{}_R_{}_pbc.txt".format(Delta,L,w,R),r_tilde)
    
    
    
    


#an_list = (1/R)*sum(an_listi)
#bn_list = (1/R)*sum(bn_listi) 

    
'''    
r_tilde_p = r_tilde    
    
# Create histogram without plotting
counts, bins, _ = plt.hist(r_tilde_p, bins=np.linspace(0,1,40), alpha=0,density=True)

# Calculate midpoints of bins
bin_midpoints = 0.5 * (bins[:-1] + bins[1:])

plt.figure()
# Plot scatter plot with midpoints and counts
plt.scatter(bin_midpoints, counts,label="Data points")
plt.xlabel(r"$\tilde{r}$",fontsize =18)
plt.ylabel(r"$  P( \tilde{r} )$",fontsize =18)

# Add labels and title

#plt.title('Level Spacing  for L= 18, half filling plus three within fragment')
#plt.show()

def P1(r):
    return 2/(1+r)**2
def P2(r):
    return (27/4)*((r+r**2)/(1+r+r**2)**2.5)

r_tilde = np.linspace(0,1,101)
plt.plot(r_tilde, [P1(i) for i in r_tilde], 'k--',label = "Poisson")
plt.plot(r_tilde, [P2(i) for i in r_tilde], 'r-.',label ="GOE")


    
    
  



plt.figure()
plt.plot(t_list,np.array(complexity_list))
plt.title("complexity_list")
plt.show()
#plt.savefig("/home/maitri/kry/complexity.pdf")

'''
