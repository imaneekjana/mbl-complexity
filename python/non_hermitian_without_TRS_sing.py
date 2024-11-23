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

from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d

from quspin.basis.user import user_basis # Hilbert space user basis

from quspin.basis.user import pre_check_state_sig_32,op_sig_32,map_sig_32 # user basis data types

from numba import carray,cfunc # numba helper functions

from numba import uint32,int32 # numba data types

import numpy as np

import matplotlib.pyplot as plt

import math

import cmath


from scipy.linalg import expm,eig

np.object = object

os.chdir("/home/user/Documents/Official/Courses/krylov/non_hermitian_MBL/without_TRS/L14")



#### THE MODEL AND HAMILTONIAN

L = 14

basis = boson_basis_1d(L=L,a=1,sps=2,Nb = L//2)

dim = basis.Ns


J = 1

U = 2

g = 0.0

h = 2

gamma = 0.1



def create_ham(hi):
    
    random = np.random.uniform(-hi,hi,L)


    term1 = [[-J*np.exp(g),i,(i+1)%L] for i in range(L)] #PBC

    term2 = [[-J*np.exp(-g),i,(i+1)%L] for i in range(L)] #PBC

    term3 = [[U,i,(i+1)%L] for i in range(L)] #PBC


    term4 = [[random[i]-1j*gamma*(-1)**i,i] for i in range(L)] # PBC




    # static and dynamic lists

    static = [["+-",term1],["-+",term2],["nn",term3],["n",term4]]

    dynamic=[]

    H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128,check_herm=False).toarray()

    H_dag = np.conjugate(H.T)

    E,V = eig(H)
    
    _,SingVal,_ = np.linalg.svd(H)
    
    return H, H_dag, E, V, SingVal







###----------XX------------------------




###--------EIGENVALUE SPECTRUM AND EIGENSTATE IPR-----------

H, H_dag, E, V, SingVal = create_ham(h)


#eigenvalue spectrum plot

reE = E.real
imE = E.imag

plt.figure()

plt.scatter(reE,imE)

plt.ylim(-0.5,0.5)

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



####----------------DISORDER AVERAGING-----------------------------------------





realizations = 100

h_list = [2,2.5,3,3.5,4,4.5,5.0,5.5,6,7,8,9]



for h in h_list:
    
    folder_name = f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}'
    
    os.makedirs(folder_name) 

    ######----------------------RUNNING OVER REALIZATIONS--------------------------



    for r in range(realizations):
        
        ###------------------SuFF------------------------------------------------------



        ###----------Survival Form Factor (SuFF) at beta=0 :---------------------------

        
        H, H_dag, E, V, SingVal = create_ham(h)
        
        # store the singular values :)
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/SingVal_set_{r}.txt',SingVal)
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/EigVal_set_{r}.txt',E)
        
        #
        
        
        
        
        psi0 = np.zeros(dim,dtype=np.complex128)


        for i in range(dim):
            
            psi0 += V.T[i]


        psi0 = psi0/np.linalg.norm(psi0)
        
        psi0_eig = np.array([1 for i in range(dim)])
        
        



        def act_U(psi,dt):
            
            psif = expm(-1j*dt*H) @ psi
            psif = psif/np.linalg.norm(psif)
            
            return psif
        
        def act_U_eig(psi_eig,dt):
            
            psif_eig = np.array([cmath.exp(-1j*dt*E[i])*psi_eig[i] for i in range(dim)])
            
            return psif_eig

        
            
        
        
        ##-----------------------------------------------------------------------------


        #######--------------------COMPLEXITY------------------------------------------


        ###-------------------ARNOLDI ALGORITHM----------------------------------------

          #Krylov basis from Arnoldi

        dim= basis.Ns


        psi0 = np.zeros(dim,dtype=np.complex128)

        for i in range(dim):
            
            psi0 += V.T[i]  # for TFD

        psi0 = psi0/np.linalg.norm(psi0)  # for TFD
        
        psi0_eig = np.array([1 for i in range(dim)])
        
        '''

        a0_Ar = psi0.T.conj() @ H @ psi0

        an_List_Ar = [a0_Ar]
        bn_List_Ar = [1]
        kn_List_Ar = [psi0]

        for n in range(dim): # Arnoldi iteration
            
            w = H @ kn_List_Ar[-1]
            
            proj = np.zeros(dim,dtype=np.complex128)
            
            for k in range(n+1):
                
                proj += (kn_List_Ar[k].T.conj() @ w) * kn_List_Ar[k]
                
            w = w - proj
            
            proj = np.zeros(dim,dtype=np.complex128)
            
            for k in range(n+1):
                
                proj += (kn_List_Ar[k].T.conj() @ w) * kn_List_Ar[k]
                
            w = w - proj
            
            b_Ar = np.sqrt(w.T.conj() @ w)
            k_Ar = w/b_Ar
            a_Ar = k_Ar.T.conj() @ H @ k_Ar
            
            if b_Ar > 10**(-10):
                
                kn_List_Ar.append(k_Ar)
                bn_List_Ar.append(b_Ar)
                an_List_Ar.append(a_Ar)
                
            else:
                break
            
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/an_List_Ar_set_{r}.txt',an_List_Ar)
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/bn_List_Ar_set_{r}.txt',bn_List_Ar)
        #
        #

        ###----------------------------------------------------------------------------  


        ###----------------BI-LANCZOS ALGORITHM----------------------------------------

          # Constructing Krylov basis for bi-Lanczos

        dim= basis.Ns


        psi0 = np.zeros(dim,dtype=np.complex128)

        for i in range(dim):
            
            psi0 += V.T[i]  # for TFD

        psi0 = psi0/np.linalg.norm(psi0)  # for TFD
        
        psi0_eig = np.array([1 for i in range(dim)])

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
                
                if c_np1>10**(-10):
                    a_np1 = np.conjugate(q_np1).T @ H @ p_np1
                    bn_list.append(b_np1)
                    cn_list.append(c_np1)
                    an_list.append(a_np1)
                    pn_list.append(p_np1)
                    qn_list.append(q_np1)
                else:
                    break
                
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/an_List_Bi_set_{r}.txt',an_list)
        np.savetxt(f'sing_gamma_{gamma}_h_{h}_L_{L}_R_{realizations}/cn_List_Bi_set_{r}.txt',cn_list)
        
'''





    
    
    
    
    
    
    
    



















    



























































