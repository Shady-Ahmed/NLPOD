# -*- coding: utf-8 -*-
"""
Forward solver for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os
from scipy.fftpack import dst, idst

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


def initial(nx,ny):
    #resting flow
    w = np.zeros([nx+1,ny+1])
    s = np.zeros([nx+1,ny+1])
    #masrigli flow [for temperature IC]
    t = np.zeros([nx+1,ny+1])
    t[:int(nx/2)+1,:] = 1.5
    t[int(nx/2)+1:,:] = 1
    
    return w,s,t

# time integration using third-order Runge Kutta method
def RK3(rhs,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt):
    aa = 1.0/3.0
    bb = 2.0/3.0

    ww = np.zeros([nx+1,ny+1])
    tt = np.zeros([nx+1,ny+1])

    ww = np.copy(w)
    tt = np.copy(t)
    
    #stage-1
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,w,s,t)
    ww[1:nx,1:ny] = w[1:nx,1:ny] + dt*rw
    tt[1:nx,1:ny] = t[1:nx,1:ny] + dt*rt    
    s = poisson_fst(nx,ny,dx,dy,ww)
    tt = tbc(tt)

    #stage-2
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,ww,s,tt)
    ww[1:nx,1:ny] = 0.75*w[1:nx,1:ny] + 0.25*ww[1:nx,1:ny] + 0.25*dt*rw
    tt[1:nx,1:ny] = 0.75*t[1:nx,1:ny] + 0.25*tt[1:nx,1:ny] + 0.25*dt*rt
    s = poisson_fst(nx,ny,dx,dy,ww)
    tt = tbc(tt)

    #stage-3
    rw,rt = rhs(nx,ny,dx,dy,Re,Pr,Ri,ww,s,tt)
    w[1:nx,1:ny] = aa*w[1:nx,1:ny] + bb*ww[1:nx,1:ny] + bb*dt*rw
    t[1:nx,1:ny] = aa*t[1:nx,1:ny] + bb*tt[1:nx,1:ny] + bb*dt*rt
    s = poisson_fst(nx,ny,dx,dy,w)
    t = tbc(t)

    return w,s,t

def tbc(t):
    t[0,:] = t[1,:]
    t[-1,:] = t[-2,:]
    t[:,0] = t[:,1]
    t[:,-1] = t[:,-2]
    
    return t

#Elliptic coupled system solver for 2D Boussinesq equations
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

#Right hand-side for the 2D Boussinesq equations
def BoussRHS(nx,ny,dx,dy,Re,Pr,Ri,w,s,t):
    
    # w-equation
    rw = np.zeros([nx-1,ny-1])
    rt = np.zeros([nx-1,ny-1])

    #laplacian terms
    Lw = laplacian(nx,ny,dx,dy,w)
    Lt = laplacian(nx,ny,dx,dy,t)

    #conduction term
    dd = 1.0/(2.0*dx)
    Cw = dd*(t[2:nx+1,1:ny]-t[0:nx-1,1:ny])
    
    #Jacobian terms
    Jw = jacobian(nx,ny,dx,dy,w,s)
    Jt = jacobian(nx,ny,dx,dy,t,s)
    
    rw = -Jw + (1/Re)*Lw + Ri*Cw
    rt = -Jt + (1/(Re*Pr))*Lt
    
    return rw,rt

#compute velocity components from streamfunction (internal points)
def velocity(nx,ny,dx,dy,s):
    u =  np.zeros([nx-1,ny-1])
    v =  np.zeros([nx-1,ny-1])
    # u = ds/dy
    u = (s[1:nx,2:ny+1] - s[1:nx,0:ny-1])/(2*dy)
    # v = -ds/dx
    u = -(s[2:nx+1,1:ny] - s[0:nx-1,1:ny])/(2*dx)
    return u,v

def export_data(Re,nx,ny,n,w,s,t):
    folder = '/data_'+ str(nx) + '_' + str(ny)       
    if not os.path.exists('./Results/Re_'+str(int(Re))+folder):
        os.makedirs('./Results/Re_'+str(int(Re))+folder)
    filename = './Results/Re_'+str(int(Re))+folder+'/data_' + str(int(n))+'.npz'
    np.savez(filename,w=w,s=s,t=t)

#%% Main program
# Inputs
lx = 8
ly = 1
nx = 512
ny = int(nx/8)
ReList = [7e2, 9e2, 10e2, 11e2, 13e2]

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

for Re in ReList:

    # initialize
    n= 0
    time=0
    w,s,t = initial(nx,ny)
    export_data(Re,nx,ny,n,w,s,t)
    
    #time integration
    for n in range(1,nt+1):
        time = time+dt
        
        w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
        
        u,v = velocity(nx,ny,dx,dy,s)
        umax = np.max(np.abs(u))
        vmax = np.max(np.abs(v))
        cfl = np.max([umax*dt/dx, vmax*dt/dy])
        
        if cfl >= 0.8:
            print('CFL exceeds maximum value')
            break
        
        if n%500==0:
            print(n, " ", time, " ", np.max(w), " ", cfl)
    
        if n%freq==0:
            export_data(Re,nx,ny,n,w,s,t)