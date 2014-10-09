"""
                 estimation_utilities.py
==============================================================

This script defines estimator/filter utility functions used in mechanizing
sensor fusion algorithms:

1)


    Author:		Demoz Gebre-Egziabher 
    Created:		July 4, 2013   
    Last Modified:	June 11, 2014    
    Copywrite 2013  Demoz Gebre-Egziabher
    License: BSD, see bsd.txt for details 
=============================================================

"""
#   Import relevant libraries

import numpy as np
import math as mt
import scipy.linalg as spl

def continous_to_discrete(A,B,dt):
     
     m = A.shape[0]
     n = A.shape[1]
     mb = B.shape[0]
     nb = B.shape[1]
     
     S = np.zeros((m+nb,n+nb))
     S[0:m,0:n] = np.copy(A)
     S[0:m,n:n+nb] = np.copy(B)
     Q = spl.expm(S*dt)
     F = np.copy(Q[0:n,0:n])
     G = np.copy(Q[0:n,n:n+nb])
     
     return F, G

def discrete_process_noise(F,G,dt,Q):   # calculate discrete Q

    s = mt.sqrt(np.size(F))
    I = np.eye(s)
    M = dt*np.dot(G,np.dot(Q,G.transpose()))
    Qd = np.dot((I + dt*F),M)
    return Qd

def discrete_Q(F,G,dt,Q):
    
     nf = F.shape[0]
     mf = F.shape[1]
     ZF = np.zeros((nf,mf))
     M = np.zeros((2*nf,2*nf))
     M[0:nf,0:nf] = np.copy(-1*F)
     M[nf:(2*nf),0:nf] = np.copy(ZF)
     M[0:nf,nf:(2*nf)] = np.copy(G.dot(Q.dot(G.transpose())))
     M[nf:(2*nf),nf:(2*nf)] = np.copy(F.transpose())
     P = spl.expm(M*dt)
     P12 = np.copy(P[0:nf,nf:(2*nf)])
     P22 = np.copy(P[nf:(2*nf),nf:(2*nf)])
     Qd = np.dot(P22.transpose(),P12)
     return Qd

def markov_noise(sigma,tau,dt,Tf):
    
    Q = 2.0*mt.pow(sigma,2)/tau
    ac = np.array([-1.0/tau])
    bc = np.array([1.0])
    Qd = discrete_process_noise(ac,bc,dt,Q)
    a = np.array([mt.exp(-dt/tau)])
    b = np.array([a*tau*(mt.exp(dt/tau) - 1.0)])
    t = np.arange(0,Tf,dt)
    u = mt.sqrt(Qd)*np.random.randn(np.size(t))
    x = np.zeros(np.size(t))
    for k in range(len(u)-1):
     #   x[k+1] = a[0]*x[k] + b[0]*u[k]
        x[k+1] = a[0]*x[k] + u[k]
    return x
    
def markov_noise2(sigma,tau,t):
    
    Q = np.array([2.0*(sigma**2)/tau])
    ac = np.reshape(np.array([-1.0/tau]),(1,1))
    bc = np.reshape(np.array([1.0]),(1,1))
    dt = np.mean(np.diff(t,n=1,axis=0))
    Qd = discrete_Q(ac,bc,dt,Q)
    a, b = continous_to_discrete(ac,bc,dt)
    drl = len(t)
    u = np.sqrt(Qd)*np.random.randn(drl,1)
    x = np.zeros((drl,1))
    for k in range(drl-1):
        #x[k+1,0] = a.dot(x[k,0]) + b.dot(u[k,0])
        x[k+1,0] = a.dot(x[k,0]) + u[k,0]
    return x
    
def allan_variance(x,f,c):
     """
     Computes the root allan variance of a time series signal, x = x(t), 
     recorded at a sampling frequency of f (in Hertz).  It returns two arrays 
     (or vectors) containing the clustering times, tau, in seconds and the 
     associated root allan variance, rav, in the same units of x(t).
     
     Parameters
     ----------
     x:	Time series signal
     f:	Sampling frequency in units of Hz
     
     Returns
     -------
     tau:	Array of clustering times
     rav:	Array of root Allan variance values at each tau.
     """    
     
     drl = np.max(x.shape)
     x = np.reshape(x,(drl,1))

     #	Define intervals over which allan variance will be computed

     tau_1 = np.arange(1.0,10.0,1.0)
     tau_2 = np.arange(10.0,91.0,10.0)
     tau_3 = np.arange(100.0,901.0,100.0)
     tau_4 = np.arange(100.0,(0.2*drl/f),100.0)
     tau_5 = np.arange(1000.0,(0.2*drl/f),500)

     if ((0.2*drl/f) <= 1100):
	  tau = np.concatenate((tau_1,tau_2,tau_4),axis=None)
     else:
	  tau = np.concatenate((tau_1,tau_2,tau_3,tau_5),axis=None)
     drl2 = np.max(tau.shape)
     tau = np.reshape(tau,(drl2,1))

     #	Define cluster sizes and number of clusters per tau

     M = [int(k) for k in (f*tau)]	# number of samples per cluster
     K = [int(drl/k) for k in M]	# number of clusters for each tau

     allan_v = np.zeros((len(K),1))

     for k in range(len(K)):
	  
	  bs = 0		# Beginning sample index
	  es = M[k]		# Ending sample index
	  
	  clusters = range(K[k])
	  x_bar = np.zeros((len(clusters),1))
	  for k1 in clusters:
	       
	       x_bar[k1,0] = np.mean(x[bs:es])
	       bs = es #+ 1
	       es = bs + M[k] #- 1
	       
	  delta = x_bar[1:] - x_bar[0:-1]
	  allan_v[k,0] = mt.sqrt(np.dot(delta.transpose(),delta)/(2*(K[k]-1)))

     return tau, allan_v

