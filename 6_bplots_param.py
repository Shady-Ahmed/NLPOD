# -*- coding: utf-8 -*-
"""
Plotting scripts for the 2D Marsigli flow problem
This code was used to generate the results accompanying the following paper:
    "Nonlinear proper orthogonal decomposition for convection-dominated flows"
    Authors: Shady E Ahmed, Omer San, Adil Rasheed, and Traian Iliescu
    Published in the Physics of Fluids journal

For any questions and/or comments, please email me at: shady.ahmed@okstate.edu
"""

#%% Import libraries
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

#%% Misc functions
def PODproj_svd(u,Phi): #Projection
    a = np.dot(Phi.T,u)  # u = Phi * a if shape of a is [nr,ns]
    return a

def PODrec_svd(a,Phi): #Reconstruction    
    u = np.dot(Phi,a)    
    return u

def import_fom_data(Re,nx,ny,n):
    folder = '/data_'+ str(nx) + '_' + str(ny)              
    #filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
    filename = './Results/Re_'+str(int(Re))+folder+'/data_' + str(int(n))+'.npz'

    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

def tbc(t):
    t[-1,:] = t[-2,:]
    t[:,-1] = t[:,-2]
    return t


#%% Define Keras functions
def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))
      
    cls = Model
    cls.__reduce__ = __reduce__


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
              training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

#%%
# Inputs
lx = 8
ly = 1
nx = 512
ny = int(nx/8)

Re = 1e3
Ri = 4
Pr = 1

Tm = 8
dt = 5e-4
nt = np.int(np.round(Tm/dt))

ns = 200
freq = np.int(nt/ns)

ReList = [7e2, 9e2, 10e2, 11e2, 13e2]
ens_size = 10

#%% grid
dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% load FOM data
tfom = np.zeros([len(ReList),nx+1,ny+1,ns+1])
for p, Re in enumerate(ReList):
    for i in range(ns+1):
        n = i*freq
        w,s,t = import_fom_data(Re,nx,ny,n)
        tfom[p,:,:,i] = np.copy(t)

#%% Load true POD data

folder = 'data_'+ str(nx) + '_' + str(ny)       
filename = './Results/'+folder+'/POD_data.npz'
poddata = np.load(filename)
wmean = poddata['wmean']
smean = poddata['smean']
tmean = poddata['tmean']

wbasis = poddata['wbasis']
sbasis = poddata['sbasis']
tbasis = poddata['tbasis']

weigenvalues = poddata['weigenvalues']
teigenvalues = poddata['teigenvalues']

wric = poddata['wric']
tric = poddata['tric']

wcoeff = poddata['wcoeff']
tcoeff = poddata['tcoeff']

#%% Compute true projection fields
tPOD2 = np.zeros([len(ReList),nx+1,ny+1,ns+1])
tPOD74 = np.zeros([len(ReList),nx+1,ny+1,ns+1])
for p, Re in enumerate(ReList):
    for n in range(ns+1):
        nr = 2
        tmp = PODrec_svd(tcoeff[p,:nr,n],tbasis[n,:nr]) + tmean
        tPOD2[p,:,:,n] = tmp.reshape([nx+1,ny+1])
        
        nr = 74
        tmp = PODrec_svd(tcoeff[p,:nr,n],tbasis[n,:nr]) + tmean
        tPOD74[p,:,:,n] = tmp.reshape([nx+1,ny+1])

#%% load GP data
tGP2 = np.zeros([len(ReList),nx+1,ny+1,ns+1])
tGP74 = np.zeros([len(ReList),nx+1,ny+1,ns+1])

for p, Re in enumerate(ReList):
    nr = 2
    folder = 'data_'+ str(nx) + '_' + str(ny)       
    filename = './Results/'+folder+'/GP_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
    gpdata = np.load(filename)
    tmp1 = gpdata['tGP']

    nr = 74
    folder = 'data_'+ str(nx) + '_' + str(ny)       
    filename = './Results/'+folder+'/GP_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
    gpdata = np.load(filename)
    tmp2 = gpdata['tGP']
    for n in range(ns+1):
        tGP2[p,:,:,n] = tmp1[:,n].reshape([nx+1,ny+1])
        tGP74[p,:,:,n] = tmp2[:,n].reshape([nx+1,ny+1])
        
#%% Loading CAE Data

latent_dim = 2
t_ae_cae = np.zeros([len(ReList),ens_size,nx+1,ny+1,ns+1])
t_lstm_cae = np.zeros([len(ReList),ens_size,nx+1,ny+1,ns+1])
latent_ae_cae = np.zeros([len(ReList),ens_size,latent_dim,ns+1])
latent_lstm_cae = np.zeros([len(ReList),ens_size,latent_dim,ns+1])

for p, Re in enumerate(ReList):
    filename = './Results/'+folder+'/GPU_CNN_AE_data_Re='+str(int(Re))+'_z='+str(latent_dim)+'.npz'
    CNNdata = np.load(filename)
    tmp1 = CNNdata['latent_ae']
    tmp2 = CNNdata['latent_lstm']
    tmp3 = CNNdata['t_ae']
    tmp4 = CNNdata['t_lstm']
    for n in range(ns+1):
        latent_ae_cae[p,:,:,n] = tmp1[:,n,:]
        latent_lstm_cae[p,:,:,n] = tmp2[:,n,:]
        t_ae_cae[p,:,:-1,:-1,n] = tmp3[:,n,:,:,0]
        t_lstm_cae[p,:,:-1,:-1,n] = tmp4[:,n,:,:,0]
        for m in range(ens_size):
            t_ae_cae[p,m,:,:,n] = tbc(t_ae_cae[p,m,:,:,n])
            t_lstm_cae[p,m,:,:,n] = tbc(t_lstm_cae[p,m,:,:,n])

#%% load CAE models
make_keras_picklable()
pickle_dir = "__PICKLES"
loadfrom = os.path.join(pickle_dir, "CAE_Autencoders.pickle")

infile = open(loadfrom,'rb')
in_results = pickle.load(infile)
infile.close()

CAE_encoders = in_results['encoders']
CAE_decoders = in_results['decoders']
CAE_autoencoders = in_results['autoencoders']
CAE_ae_train_losses = in_results['ae_train_losses']
CAE_ae_valid_losses = in_results['ae_valid_losses']
CAE_ae_training_times = in_results['ae_training_times']
           
#%% Loading Nonlinear POD Data

nr = 74
t_ae_nlpod = np.zeros([len(ReList),ens_size,nx+1,ny+1,ns+1])
t_lstm_nlpod = np.zeros([len(ReList),ens_size,nx+1,ny+1,ns+1])
latent_ae_nlpod = np.zeros([len(ReList),ens_size,latent_dim,ns+1])
latent_lstm_nlpod = np.zeros([len(ReList),ens_size,latent_dim,ns+1])

for p, Re in enumerate(ReList):
    filename = './Results/'+folder+'/CPU_NonPOD_data_Re='+str(int(Re))+'_nr='+str(nr)+'.npz'
    nonPODdata = np.load(filename)
    tmp1 = nonPODdata['latent_ae']
    tmp2 = nonPODdata['latent_lstm']
    tmp3 = nonPODdata['t_ae']
    tmp4 = nonPODdata['t_lstm']
    for n in range(ns+1):
        latent_ae_nlpod[p,:,:,n] = tmp1[:,n,:]
        latent_lstm_nlpod[p,:,:,n] = tmp2[:,n,:]
        for m in range(ens_size):
            tmp = tmp3[m,:,n]
            t_ae_nlpod[p,m,:,:,n] = tmp.reshape([nx+1,ny+1])
            tmp = tmp4[m,:,n]
            t_lstm_nlpod[p,m,:,:,n] = tmp.reshape([nx+1,ny+1])

#%% load NLPOD models
make_keras_picklable()
pickle_dir = "__PICKLES"
loadfrom = os.path.join(pickle_dir, "NLPOD_Autencoders_nr="+str(nr)+".pickle")

infile = open(loadfrom,'rb')
in_results = pickle.load(infile)
infile.close()

NLPOD_encoders = in_results['encoders']
NLPOD_decoders = in_results['decoders']
NLPOD_autoencoders = in_results['autoencoders']
NLPOD_ae_train_losses = in_results['ae_train_losses']
NLPOD_ae_valid_losses = in_results['ae_valid_losses']
NLPOD_ae_training_times = in_results['ae_training_times']

#%% Computing l2 error
err_cae_rec = np.zeros([len(ReList),ens_size])
err_cae_pred = np.zeros([len(ReList),ens_size])
err_nlpod_rec = np.zeros([len(ReList),ens_size])
err_nlpod_pred = np.zeros([len(ReList),ens_size])

for p in range(len(ReList)):
    for m in range(ens_size):
        err_cae_rec[p,m] = np.mean( (tfom[p,:,:,:] - t_ae_cae[p,m,:,:,:])**2 )
        err_cae_pred[p,m] = np.mean( (tfom[p,:,:,:] - t_lstm_cae[p,m,:,:,:])**2 )
        err_nlpod_rec[p,m] = np.mean( (tfom[p,:,:,:] - t_ae_nlpod[p,m,:,:,:])**2 )
        err_nlpod_pred[p,m] = np.mean( (tfom[p,:,:,:] - t_lstm_nlpod[p,m,:,:,:])**2 )


err_cae_rec_mean = np.mean(err_cae_rec,axis=1)
err_cae_pred_mean = np.mean(err_cae_pred,axis=1)
err_nlpod_rec_mean = np.mean(err_nlpod_rec,axis=1)
err_nlpod_pred_mean = np.mean(err_nlpod_pred,axis=1)

err_cae_rec_std = np.std(err_cae_rec,axis=1)
err_cae_pred_std = np.std(err_cae_pred,axis=1)
err_nlpod_rec_std = np.std(err_nlpod_rec,axis=1)
err_nlpod_pred_std = np.std(err_nlpod_pred,axis=1)


#%%
nrplot = 150
ncut1 = 74
ncut0 = 2
index = np.arange(1,nrplot+1)

plt.figure(figsize=(10,6))
plt.plot(index,tric[:nrplot],color='k',linewidth=3)

plt.plot([ncut0,ncut0],[tric[0],tric[ncut0-1]],color='k', linestyle='--')
plt.plot([0,ncut0],[tric[ncut0-1],tric[ncut0-1]],color='k', linestyle='--')

plt.plot([ncut1,ncut1],[tric[0],tric[ncut1-1]],color='k', linestyle='--')
plt.plot([0,ncut1],[tric[ncut1-1],tric[ncut1-1]],color='k', linestyle='--')

plt.fill_between(index[:ncut0], tric[:ncut0], y2=tric[0],alpha=0.6,color='orange')#,
plt.fill_between(index[:ncut1], tric[:ncut1], y2=tric[0],alpha=0.2,color='blue')

plt.text(1.1, 78, r'$'+str(np.round(tric[ncut0-1],decimals=2))+'\%$', va='center',fontsize=18)
plt.text(5.1, 101, r'$'+str(np.round(tric[ncut1-1],decimals=2))+'\%$', va='center',fontsize=18)
plt.text(2.1, 68, r'$r='+str(ncut0)+'$', va='center',fontsize=18)
plt.text(78, 80, r'$r='+str(ncut1)+'$', va='center',fontsize=18)

plt.xlim(left=1,right=nrplot)
plt.ylim(bottom=tric[0], top=104)
plt.xscale('log')
#%
plt.xlabel(r'\bf POD index ($k$)')
plt.ylabel(r'\bf RIC ($\%$)')
plt.savefig('./Plots/RIC.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./Plots/RIC.pdf', dpi = 500, bbox_inches = 'tight')

#%%
Re = 1000
nlvls = 30
x_ticks = [0,1,2,3,4,5,6,7,8]
y_ticks = [0,1]

colormap = 'RdBu_r'

low = 1.05
high = 1.45
v = np.linspace(low, high, nlvls, endpoint=True)

ctick = np.linspace(low, high, 5, endpoint=True)

fig, axs = plt.subplots(5,2,figsize=(16,10))

for j, ind in enumerate([150,200]):

    cs = axs[0][j].contour(X,Y,tfom[ReList.index(Re),:,:,ind],v,cmap=colormap,linewidths=1,extend='both')
    cs.set_clim([low, high])
    
    cs = axs[1][j].contour(X,Y,tGP2[ReList.index(Re),:,:,ind],v,cmap=colormap,linewidths=1,extend='both')
    cs.set_clim([low, high])

    cs = axs[2][j].contour(X,Y,tGP74[ReList.index(Re),:,:,ind],v,cmap=colormap,linewidths=1,extend='both')
    cs.set_clim([low, high])

    cs = axs[3][j].contour(X,Y,np.mean(t_lstm_cae[ReList.index(Re),:,:,:,ind],axis=0),v,cmap=colormap,linewidths=1,extend='both')
    cs.set_clim([low, high])

    cs = axs[4][j].contour(X,Y,np.mean(t_lstm_nlpod[ReList.index(Re),:,:,:,ind],axis=0),v,cmap=colormap,linewidths=1,extend='both')
    cs.set_clim([low, high])

    axs[4][j].set_xlabel('$x$')
    for i in range(5):
        axs[i][j].set_ylabel('$y$')

axs[0][0].set_title(r'$t=6$')
axs[0][1].set_title(r'$t=8$')

plt.text(-11.5, 6.5, r'\bf{FOM}', va='center',fontsize=18)
plt.text(-11.5, 5.1, r'\bf{GPOD}', va='center',fontsize=18)
plt.text(-11.5, 4.85, r'\bf{($r=2$)}', va='center',fontsize=18)
plt.text(-11.5, 3.6, r'\bf{GPOD}', va='center',fontsize=18)
plt.text(-11.5, 3.35, r'\bf{($r=74$)}', va='center',fontsize=18)
plt.text(-11.5, 2.1, r'\bf{CAE}', va='center',fontsize=18)
plt.text(-11.5, 1.85, r'\bf{($r=2$)}', va='center',fontsize=18)
plt.text(-11.5, 0.6, r'\bf{NLPOD}', va='center',fontsize=18)
plt.text(-11.5, 0.35, r'\bf{($r=2$)}', va='center',fontsize=18)

fig.subplots_adjust(bottom=0.12,wspace=0.15,hspace=0.5)
plt.savefig('./Plots/contours.png', dpi = 500, bbox_inches = 'tight')
plt.savefig('./Plots/contours.pdf', dpi = 500, bbox_inches = 'tight')
