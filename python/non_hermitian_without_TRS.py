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

os.chdir("/home/user/Documents/Official/Courses/krylov/non_hermitian_MBL/without_TRS/L12")


#### THE MODEL AND HAMILTONIAN

L = 12

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



h_list = [2,2.5,3,3.5,4,4.5,5,5.5,6]


for h in h_list:
    ######----------------------RUNNING OVER REALIZATIONS--------------------------

    realizations = 150



    complexity_Bi_av = []
    KIPR_Bi_av = []
    Kent_Bi_av = []

    complexity_Ar_av = []
    KIPR_Ar_av = []
    Kent_Ar_av = []

    SuFF_av = []


    t0 = 0*dim
    tf = 10*dim
    dt = 0.001*dim
    tListEarly = np.arrange(0,4*dt,4*dt/2000)
    tListSuFF = np.arange(4*dt,tf,dt)
    tListSuFF = np.concatenate((tListEarly,tListSuFF))


    t0 = 0*dim
    tf = 2.5*dim
    dt = 0.01*dim
    tListEarly = np.arrange(0,4*dt,4*dt/100)
    tListComp = np.arange(4*dt,tf,dt)
    tListComp = np.concatenate((tListEarly,tListComp))


    for r in range(realizations):
        
        ###------------------SuFF------------------------------------------------------



        ###----------Survival Form Factor (SuFF) at beta=0 :---------------------------

        
        H, H_dag, E, V, SingVal = create_ham(h)
        
        #store the singular values :)
        
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

        tList = tListSuFF
        SuFFList = []

        psitrunning = psi0
        
        psit_running_eig = psi0_eig
        
        '''

        for t in tList:
            
            #psit = act_U(psitrunning,dt)
            
            psit_eig = act_U_eig(psit_running_eig, dt)
            psit = U @ psit_eig
            norm = np.linalg.norm(psit)
            psit = psit/norm
            psit_running_eig = psit_eig/norm
            
            Samp = psit.T.conj() @ psi0
            SuFF = np.abs(Samp)**2
            SuFFList.append(SuFF)
            
            
            psitrunning = psit
            
        SuFF_av.append(np.array(SuFFList))
            
        '''  
        
        
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
                
        

        ###----------------------------------------------------------------------------            
                
                
        ###-------------------------COMPUTING COMPLEXITY-------------------------------
                
                
                
        
        tList = tListComp

        # for bi-Lanczos

        complexityList_Bi = [0]
        normList_Bi = [1]
        KIPRList_Bi = [1]
        KentList_Bi = [0]

        #for Arnoldi

        complexityList_Ar = [0]
        normList_Ar = [1]
        KIPRList_Ar = [1]
        KentList_Ar = [0]


        #initiate
        psitrunning = psi0
        
        psit_running_eig = psi0_eig
        
        


        for t in tList:
            
            # for direct evolution
            #psit = act_U(psitrunning, dt)
            #psit = psit/(np.linalg.norm(psit))
            
            
            #for eigenbasis evolution
            psit_eig = act_U_eig(psit_running_eig, dt)
            psit = U @ psit_eig
            norm = np.linalg.norm(psit)
            psit = psit/norm
            psit_running_eig = psit_eig/norm
            
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

        complexityList_Bi = np.array(complexityList_Bi[0:-1])
        normList_Bi = np.array(normList_Bi[0:-1])
        KIPRList_Bi = np.array(KIPRList_Bi[0:-1])
        KentList_Bi = np.array(KentList_Bi[0:-1])
        
        complexity_Bi_av.append(complexityList_Bi)
        KIPR_Bi_av.append(KIPRList_Bi)
        Kent_Bi_av.append(KentList_Bi)

        #for Arnoldi

        complexityList_Ar = np.array(complexityList_Ar[0:-1])
        normList_Ar = np.array(normList_Ar[0:-1])
        KIPRList_Ar = np.array(KIPRList_Ar[0:-1])
        KentList_Ar = np.array(KentList_Ar[0:-1])
        
        complexity_Ar_av.append(complexityList_Ar)
        KIPR_Ar_av.append(KIPRList_Ar)
        Kent_Ar_av.append(KentList_Ar)
        
    ###-----------RUNNING OVER REALIZATIONS ENDED----------------------------------



    ###--------COMPUTING DISORDER AVERAGES-----------------------------------------   

    complexity_Bi_av = (1/realizations) * sum(complexity_Bi_av)
    KIPR_Bi_av = (1/realizations) * sum(KIPR_Bi_av)
    Kent_Bi_av = (1/realizations) * sum(Kent_Bi_av)

    complexity_Ar_av = (1/realizations) * sum(complexity_Ar_av)
    KIPR_Ar_av = (1/realizations) * sum(KIPR_Ar_av)
    Kent_Ar_av = (1/realizations) * sum(Kent_Ar_av)

    #SuFF_av = (1/realizations) * sum(SuFF_av)


    ###----------------------------------------------------------------------------

    ####-------------------create directory and save data------------------------

    os.makedirs('gamma_pt1_h_{}_R_150'.format(h))

    np.savetxt("gamma_pt1_h_{}_R_150/complexity_Bi_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),complexity_Bi_av)
    np.savetxt("gamma_pt1_h_{}_R_150/KIPR_Bi_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),KIPR_Bi_av)
    np.savetxt("gamma_pt1_h_{}_R_150/Kent_Bi_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),Kent_Bi_av)
    np.savetxt("gamma_pt1_h_{}_R_150/complexity_Ar_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),complexity_Ar_av)
    np.savetxt("gamma_pt1_h_{}_R_150/KIPR_Ar_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),KIPR_Ar_av)
    np.savetxt("gamma_pt1_h_{}_R_150/Kent_Ar_av_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),Kent_Ar_av)
    np.savetxt("gamma_pt1_h_{}_R_150/tListComp_J_{}_U_{}_gamma_{}_h_{}_R_{}.txt".format(h,J,U,gamma,h,realizations),tListComp)
    
    
    
        



    
    








      
    
    
    
    
    



























###--------PLOTTING SuFF-------------------------------------------------------
'''
plt.figure()
plt.plot(tListSuFF/dim,SuFF_av)
plt.xscale('log')
plt.yscale('log')
plt.title(rf'Survival Form Factor (SuFF), $gamma$={gamma}, h={h}')
plt.show()
'''

###----------PLOTTING COMPLEXITY-----------------------------------------------

plt.figure()
plt.plot(tListComp/dim,complexity_Bi_av/dim)
plt.title(rf'complexity vs. time for bi-Lanczos, $gamma$={gamma}, h= {h}')
plt.xlabel('t/dim')
plt.ylabel('C(t)/dim')
plt.show()

plt.figure()
plt.plot(tListComp/dim,complexity_Ar_av/dim)
plt.title(rf'complexity vs. time for Arnoldi, $gamma$={gamma}, h= {h}')
plt.xlabel('t/dim')
plt.ylabel('C(t)/dim')
plt.show()


###----------------------------------------------------------------------------
















    



























































