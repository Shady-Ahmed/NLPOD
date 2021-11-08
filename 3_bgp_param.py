# -*- coding: utf-8 -*-
"""
Galerkin POD (GPOD) framework implementation for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import time as timer

#%% Define functions

# compute jacobian using arakawa scheme
# computed at all internal physical domain points (1:nx-1,1:ny-1)
def jacobian(nx,ny,dx,dy,q,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    #Arakawa 1:nx,1:ny   
    j1 = gg*( (q[2:nx+1,1:ny]-q[0:nx-1,1:ny])*(s[1:nx,2:ny+1]-s[1:nx,0:ny-1]) \
             -(q[1:nx,2:ny+1]-q[1:nx,0:ny-1])*(s[2:nx+1,1:ny]-s[0:nx-1,1:ny]))

    j2 = gg*( q[2:nx+1,1:ny]*(s[2:nx+1,2:ny+1]-s[2:nx+1,0:ny-1]) \
            - q[0:nx-1,1:ny]*(s[0:nx-1,2:ny+1]-s[0:nx-1,0:ny-1]) \
            - q[1:nx,2:ny+1]*(s[2:nx+1,2:ny+1]-s[0:nx-1,2:ny+1]) \
            + q[1:nx,0:ny-1]*(s[2:nx+1,0:ny-1]-s[0:nx-1,0:ny-1]))
    
    j3 = gg*( q[2:nx+1,2:ny+1]*(s[1:nx,2:ny+1]-s[2:nx+1,1:ny]) \
            - q[0:nx-1,0:ny-1]*(s[0:nx-1,1:ny]-s[1:nx,0:ny-1]) \
            - q[0:nx-1,2:ny+1]*(s[1:nx,2:ny+1]-s[0:nx-1,1:ny]) \
            + q[2:nx+1,0:ny-1]*(s[2:nx+1,1:ny]-s[1:nx,0:ny-1]) )
    jac = (j1+j2+j3)*hh
    return jac

def laplacian(nx,ny,dx,dy,w):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    lap = aa*(w[2:nx+1,1:ny]-2.0*w[1:nx,1:ny]+w[0:nx-1,1:ny]) \
        + bb*(w[1:nx,2:ny+1]-2.0*w[1:nx,1:ny]+w[1:nx,0:ny-1])
    return lap    


def const_term(nr,wm,sm,tm,Phiw,Phit,Re,Ri,Pr):
    
    #vorticity equation
    #jacobian
    w = wm.reshape([nx+1,ny+1])
    s = sm.reshape([nx+1,ny+1])

    tmp1 = -jacobian(nx,ny,dx,dy,w,s)
        
    #laplacian
    w = wm.reshape([nx+1,ny+1])
    tmp2 = (1/Re)*laplacian(nx,ny,dx,dy,w)
    
    #conduction
    dd = 1.0/(2.0*dx)
    t = tm.reshape([nx+1,ny+1])
    tmp3 = Ri*dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])
    
    #compute constant term
    b_c = np.zeros(nr)
    for k in range(nr):
        tmp = np.zeros([nx+1,ny+1])
        tmp[1:nx,1:ny] = (tmp1 + tmp2 + tmp3)
        tmp = tmp.reshape([(nx+1)*(ny+1),])
        b_c[k] = tmp.T @ Phiw[:,k]
        
    cterm = [b_c]
        
    # temperature equation
    #jacobian
    t = tm.reshape([nx+1,ny+1])
    s = sm.reshape([nx+1,ny+1])
    tmp1 = -jacobian(nx,ny,dx,dy,t,s)
        
    #laplacian
    t = tm.reshape([nx+1,ny+1])
    tmp2 = 1/(Re*Pr)*laplacian(nx,ny,dx,dy,t)
    
    #compute constant term
    b_c = np.zeros(nr)
    for k in range(nr):
        tmp = np.zeros([nx+1,ny+1])
        tmp[1:nx,1:ny] = (tmp1 + tmp2)
        tmp = tmp.reshape([(nx+1)*(ny+1),])
        b_c[k] = tmp.T @ Phit[:,k]
        
    cterm.append(b_c)
    return cterm


def lin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re,Ri,Pr):
    
    #vorticity equation
    b_l = np.zeros([nr,nr])

    for i in range(nr):
        #L1
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        tmp1 = (1/Re)*laplacian(nx,ny,dx,dy,w)
        
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        s = np.copy(sm.reshape([nx+1,ny+1]))
        tmp2 = -jacobian(nx,ny,dx,dy,w,s)
        
        w = np.copy(wm.reshape([nx+1,ny+1]))
        s = np.copy(Phis[:,i].reshape([nx+1,ny+1]))
        tmp3 = -jacobian(nx,ny,dx,dy,w,s)
                
        for k in range(nr):
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1+tmp2+tmp3)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_l[i,k] = tmp.T @ Phiw[:,k]   
            
    lterm = [b_l]

    #temperature equation
    b_l= np.zeros([nr,nr])
    for i in range(nr):
        #L1
        t = np.copy(Phit[:,i].reshape([nx+1,ny+1]))
        tmp1 = 1/(Re*Pr)*laplacian(nx,ny,dx,dy,t)
        
        t = np.copy(Phit[:,i].reshape([nx+1,ny+1]))
        s = np.copy(sm.reshape([nx+1,ny+1]))
        tmp2 = -jacobian(nx,ny,dx,dy,t,s)
    
        for k in range(nr):
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1+tmp2)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_l[i,k] = tmp.T @ Phit[:,k]   
            
    lterm.append(b_l)

    return lterm


def crosslin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re,Ri,Pr):
    
    #vorticity equation
    b_cl= np.zeros([nr,nr])

    for i in range(nr):       
        #L2
        dd = 1.0/(2.0*dx)
        t = np.copy(Phit[:,i].reshape([nx+1,ny+1]))
        tmp1 = Ri*dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])
        
        for k in range(nr):
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_cl[i,k] = tmp.T @ Phiw[:,k]  
            
    clterm = [b_cl]

    #temperature equation
    b_cl= np.zeros([nr,nr])
    for i in range(nr):
        #L1
        t = np.copy(tm.reshape([nx+1,ny+1]))
        s = np.copy(Phis[:,i].reshape([nx+1,ny+1]))
        tmp1 = -jacobian(nx,ny,dx,dy,t,s)
                
        for k in range(nr):
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1)
            tmp = tmp.reshape([(nx+1)*(ny+1),])
            b_cl[i,k] = tmp.T @ Phit[:,k]   

    clterm.append(b_cl)
    return clterm


def nonlin_term(nr,Phiw,Phis,Phit,nx,ny):
    
    #vorticity equation
    b_nl = np.zeros([nr,nr,nr])
    for i in range(nr):
        w = np.copy(Phiw[:,i].reshape([nx+1,ny+1]))
        for j in range(nr):
            s = np.copy(Phis[:,j].reshape([nx+1,ny+1]))
            tmp1 = -jacobian(nx,ny,dx,dy,w,s)
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1)
            for k in range(nr):
                tmp = tmp.reshape([(nx+1)*(ny+1),])
                b_nl[i,j,k] = tmp.T @ Phiw[:,k]   
                
    nlterm = [b_nl] 
           
    #temperature equation            
    b_nl = np.zeros([nr,nr,nr])
    for i in range(nr):
        t = np.copy(Phit[:,i].reshape([nx+1,ny+1]))
        for j in range(nr):
            s = np.copy(Phis[:,j].reshape([nx+1,ny+1]))
            tmp1 = -jacobian(nx,ny,dx,dy,t,s)
            tmp = np.zeros([nx+1,ny+1])
            tmp[1:nx,1:ny] = np.copy(tmp1)
            for k in range(nr):
                tmp = tmp.reshape([(nx+1)*(ny+1),])
                b_nl[i,j,k] = tmp.T @ Phit[:,k]  
        
    nlterm.append(b_nl)
    return nlterm


# Right Handside of Galerkin Projection
def GROMrhs(nr, cterm, lterm, clterm, nlterm, a, b): 
    
    #vorticity equation
    r1w, r2w, r3w = [np.zeros(nr) for _ in range(3)]
    r1t, r2t, r3t = [np.zeros(nr) for _ in range(3)]

    r1w = cterm[0]
    r1t = cterm[1]

    a = a.ravel()
    b = b.ravel()

    for k in range(nr):
        r2w[k] = np.sum(lterm[0][:,k]*a) + np.sum(clterm[0][:,k]*b) 
        r2t[k] = np.sum(lterm[1][:,k]*b) + np.sum(clterm[1][:,k]*a) 

    for k in range(nr):
        for i in range(nr):
            #r3w[k] = r3w[k] + np.sum(nlterm[0][i,:,k]*a)*a[i]
            #r3t[k] = r3t[k] + np.sum(nlterm[1][i,:,k]*b)*a[i]
            for j in range(nr):
                r3w[k] = r3w[k] + nlterm[0][i,j,k]*a[i]*a[j]
                r3t[k] = r3t[k] + nlterm[1][i,j,k]*a[j]*b[i]
                
    rw = r1w + r2w + r3w
    rt = r1t + r2t + r3t
    
    return rw,rt


# time integration using third-order Runge Kutta method
def RK3(nr, rhs, cterm, lterm, clterm, nlterm, a, b, dt):
    c1 = 1.0/3.0
    c2 = 2.0/3.0
    
    #stage-1
    rw,rt = rhs(nr, cterm, lterm, clterm, nlterm, a, b)
    a0 = a + dt*rw
    b0 = b + dt*rt

    #stage-2
    rw,rt = rhs(nr, cterm, lterm, clterm, nlterm, a0, b0)
    a0 = 0.75*a + 0.25*a0 + 0.25*dt*rw
    b0 = 0.75*b + 0.25*b0 + 0.25*dt*rt

    #stage-3
    rw,rt = rhs(nr, cterm, lterm, clterm, nlterm, a0, b0)
    a = c1*a + c2*a0 + c2*dt*rw
    b = c1*b + c2*b0 + c2*dt*rt

    return a,b

def PODproj_svd(u,Phi): #Projection
    a = np.dot(Phi.T,u)  # u = Phi * a if shape of a is [nr,ns]
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a)    
    return u

#%%
def import_fom_data(Re,nx,ny,n):
    folder = '/data_'+ str(nx) + '_' + str(ny)              
    filename = './Results/Re_'+str(int(Re))+folder+'/data_' + str(int(n))+'.npz'

    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

#%% Main program
# Inputs
lx = 8
ly = 1
nx = 512
ny = int(nx/8)

ReList = [7e2, 9e2, 10e2, 11e2, 13e2]

Re = 1000
Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = np.int(np.round(Tm/dt))

ns = 200
freq = np.int(nt/ns)

dtrom = dt*freq

nr = 74
#%% grid
dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Load POD data
folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './Results/'+folder+'/POD_data.npz'
data = np.load(filename) 
wmean = data['wmean']
smean = data['smean']
tmean = data['tmean']
wbasis = data['wbasis']
sbasis = data['sbasis']
tbasis = data['tbasis']
wcoeff = data['wcoeff'][ReList.index(Re),:,:]
tcoeff = data['tcoeff'][ReList.index(Re),:,:]

#%% Select the first nr
Phiw = wbasis[:,:nr]
Phis = sbasis[:,:nr]
Phit = tbasis[:,:nr]
wm = wmean
sm = smean
tm = tmean

#%% Compute GP coefficients for w-equation
cterm = const_term(nr,wm,sm,tm,Phiw,Phit,Re,Ri,Pr)
lterm = lin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re,Ri,Pr)
clterm = crosslin_term(nr,wm,sm,tm,Phiw,Phis,Phit,Re,Ri,Pr)
nlterm = nonlin_term(nr,Phiw,Phis,Phit,nx,ny)

#%% Time integration
aGP, bGP = [np.zeros([nr,ns+1]) for _ in range(2)]

testing_time = 0

testing_time_init = timer.time()
aGP[:,0] = np.copy(wcoeff[:nr,0])
bGP[:,0] = np.copy(tcoeff[:nr,0])
testing_time += timer.time() - testing_time_init 

for n in range(ns):
    testing_time_init = timer.time()
    aGP[:,n+1], bGP[:,n+1] =  RK3(nr, GROMrhs, cterm, lterm, clterm, nlterm, aGP[:,n], bGP[:,n], dtrom)
    testing_time += timer.time() - testing_time_init 

#%% Reconstruct fields
wGP = PODrec_svd(aGP,Phiw) + wmean.reshape(-1,1)
sGP = PODrec_svd(aGP,Phis) + smean.reshape(-1,1)
tGP = PODrec_svd(bGP,Phit) + tmean.reshape(-1,1)

    
#%% #%% Save data

folder = 'data_'+ str(nx) + '_' + str(ny)       
if not os.path.exists('./Results/'+folder):
    os.makedirs('./Results/'+folder)

filename = './Results/'+folder+'/GP_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
np.savez(filename, aGP=aGP, bGP=bGP,\
                    wGP=wGP, sGP=sGP, tGP=tGP)

#%% Document CPU times
cpu = open("CPU_GPOD_"+str(nr)+".txt", "w")
message = 'Total testing time in seconds = ' + str(testing_time) + '\n'
cpu.write(message)
cpu.close()
    
