# -*- coding: utf-8 -*-
"""
POD basis generation for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os
import sys

from scipy.fftpack import dst, idst
from numpy import linalg as LA
import matplotlib.pyplot as plt

#%% Define functions

#Elliptic coupled system solver for 2D Boussinesq equation:
def poisson_fst(nx,ny,dx,dy,w):

    f = np.zeros([nx-1,ny-1])
    f = np.copy(-w[1:nx,1:ny])

    #DST: forward transform
    ff = np.zeros([nx-1,ny-1])
    ff = dst(f, axis = 1, type = 1)
    ff = dst(ff, axis = 0, type = 1) 
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    u1 = ff/alpha
        
    #IDST: inverse transform
    u = idst(u1, axis = 1, type = 1)
    u = idst(u, axis = 0, type = 1)
    u = u/((2.0*nx)*(2.0*ny))

    ue = np.zeros([nx+1,ny+1])
    ue[1:nx,1:ny] = u
    
    return ue

def import_fom_data(Re,nx,ny,n):
    folder = '/data_'+ str(nx) + '_' + str(ny)              
    filename = './Results/Re_'+str(int(Re))+folder+'/data_' + str(int(n))+'.npz'

    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

def POD_svd(wdata,tdata,nr):
    
    ne,nd,ns = wdata.shape
    Aw = np.zeros([nd,ns*ne])
    At = np.zeros([nd,ns*ne])
    #stack data along first axis
    for p in range(ne):
        Aw[:,p*ns:(p+1)*ns] = wdata[p,:,:]
        At[:,p*ns:(p+1)*ns] = tdata[p,:,:]
       
    #mean subtraction
    wm = np.mean(Aw,axis=1)
    tm = np.mean(At,axis=1)

    Aw = Aw - wm.reshape([-1,1])
    At = At - tm.reshape([-1,1])
    
    #singular value decomposition
    Uw, Sw, Vhw = LA.svd(Aw, full_matrices=False)
    Ut, St, Vht = LA.svd(At, full_matrices=False)
   
    Phiw = Uw[:,:nr]  
    Lw = Sw**2
    #compute RIC (relative importance index)
    RICw = np.cumsum(Lw)/np.sum(Lw)*100   

    Phit = Ut[:,:nr]  
    Lt = St**2
    #compute RIC (relative importance index)
    RICt = np.cumsum(Lt)/np.sum(Lt)*100   
    
    return wm,Phiw,Lw,RICw,tm,Phit,Lt,RICt 

def PODproj_svd(u,Phi): #Projection
    a = np.dot(Phi.T,u)  # u = Phi * a if shape of a is [nr,ns]
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a)    
    return u

#%% Main program
# Inputs
lx = 8
ly = 1
nx = 512
ny = int(nx/8)

ReList = [7e2, 9e2, 10e2, 11e2, 13e2]
ReTrain = [7e2, 9e2, 11e2, 13e2]

Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = np.int(np.round(Tm/dt))

ns = 200
freq = np.int(nt/ns)

#%% grid

dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Loading data
wdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #vorticity
sdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #streamfunction
tdata = np.zeros([len(ReList),(nx+1)*(ny+1),ns+1]) #temperature
    
nstart= 0
nend= nt
nstep= freq
for p, Re in enumerate(ReList):
    ii = 0
    for n in range(nstart,nend+1,nstep):
        w,s,t = import_fom_data(Re,nx,ny,n)
        wdata[p,:,ii] = w.reshape([-1,])
        sdata[p,:,ii] = s.reshape([-1,])
        tdata[p,:,ii] = t.reshape([-1,])
        ii = ii + 1

#%% POD basis generation
nr = 100 #number of basis to store [we might not need to *use* all of them]

#compute  mean field and basis functions for potential voriticity
mask = [ReList.index(Re) for Re in ReTrain] 
inc = 1
wmean,wbasis,weigenvalues,wric,tmean,tbasis,teigenvalues,tric = POD_svd(wdata[mask,:,::inc],tdata[mask,:,::inc],nr)
    
#%% Compute Streamfunction mean and basis functions
# from those of potential vorticity using Poisson equation

tmp = wmean.reshape([nx+1,ny+1])
tmp = poisson_fst(nx,ny,dx,dy,tmp)
smean = tmp.reshape([-1,])

sbasis = np.zeros([(nx+1)*(ny+1),nr])
for k in range(nr):
    tmp = np.copy(wbasis[:,k]).reshape([nx+1,ny+1])
    tmp = poisson_fst(nx,ny,dx,dy,tmp)
    sbasis[:,k] = tmp.reshape([-1,])

#%% compute true modal coefficients

wcoeff = np.zeros([len(ReList),nr,ns+1])
tcoeff = np.zeros([len(ReList),nr,ns+1])

for p, Re in enumerate(ReList):
    
    w = np.copy(wdata[p,:,:])
    t = np.copy(tdata[p,:,:])

    tmp = w-wmean.reshape(-1,1)
    wcoeff[p,:,:] = PODproj_svd(tmp,wbasis)
        
    tmp = t-tmean.reshape(-1,1)
    tcoeff[p,:,:] = PODproj_svd(tmp,tbasis)
    
#%% Save data
folder = 'data_'+ str(nx) + '_' + str(ny)       
if not os.path.exists('./Results/'+folder):
    os.makedirs('./Results/'+folder)

filename = './Results/'+folder+'/POD_data.npz'
np.savez(filename, wmean=wmean, wbasis=wbasis,\
                    smean=smean, sbasis=sbasis,\
                    tmean=tmean, tbasis=tbasis,\
                    wcoeff=wcoeff, tcoeff=tcoeff,\
                    weigenvalues=weigenvalues, wric=wric,\
                    teigenvalues=teigenvalues, tric=tric)
    
