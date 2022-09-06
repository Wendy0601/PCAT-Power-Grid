import os  
import numpy as np
from scipy import *
from numpy import dot, multiply, diag, power, pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv,norm
from scipy.linalg import svd, svdvals 
import scipy.io as sio  
import re 
import torch

def loadline(rootPath): 
    data = sio.loadmat(os.path.join(rootPath, 'train_original_sig.mat'))
    line = data['line']
    Y = data['Y'] 
    labels = data['Labels'][0]
    Yabs = abs(Y)
    Yabs1= Yabs.todense() 
    line_neib = {}
    for r in range(line.shape[0]):
        busl , busr = line[r, :2]
        nl = np.r_[np.where(line[:, 0] ==  busl )[0], np.where(line[:, 1] ==  busl )[0] ]
        nr = np.r_[np.where(line[:, 0] ==  busr )[0], np.where(line[:,1] ==  busr )[0] ]  
        line_neib[r + 1] = np.unique(np.r_[nl+1, nr+1])
    return line, Y.todense(),    line_neib 



    
def load_data_VI_new(w,path,name,mag = False): 
    base = 100.0
    PathName = os.path.join(path, name)
    data=sio.loadmat(PathName);   
    mag0= data['V0_all']  
    theta0= data['theta0_all'] 
    mag1= data['V1_all'] 
    theta1= data['theta1_all'] 
    Imag0= data['I0_all'] 
    Itheta0= data['Itheta0_all'] 
    Imag1= data['I1_all'] 
    Itheta1= data['Itheta1_all'] 
    Y_ad= data['Y'] 
    V0 = mag0 * np.cos(theta0  ) + 1j * mag0 * np.sin(theta0  )
    V1 = mag1 * np.cos(theta1 ) + 1j * mag1 * np.sin(theta1  )
    num_bus, num_sample = np.shape(V0)
    dV = V1 - V0 
    if not mag:  
        dreal = np.zeros((num_bus, num_sample))
        dimg = np.zeros((num_bus, num_sample))  
        all_real = np.real(dV)
        all_imag = np.imag(dV) 
        dreal[w,:] =all_real[w,:]  
        dimg[w,:] = all_imag[w,:]  
        train_x = np.float64(np.r_[dreal, dimg].T) 
    else:
        dmag = np.zeros((num_bus, num_sample))
        dtheta = np.zeros((num_bus, num_sample))  
        all_mag = np.abs(dV)
        all_theta = np.angle(dV) 
        dmag[w,:] =all_mag[w,:]  
        dtheta[w,:] = all_theta[w,:]  
        train_x = np.float64(np.r_[dmag, dtheta].T) 
    train_labels = data['Labels'][0] 
    col, buses = np.shape(train_x)  
    train_x = torch.FloatTensor(train_x)
    train_x = torch.unsqueeze(train_x, 1)
    train_x = torch.unsqueeze(train_x, 3) 
    train_y = np.reshape(train_labels, [col,])  
    return train_x,   torch.LongTensor(train_y) ,col  
 

def choose_w(line, thres, num_bus = 68): 
    '''choose the measured buses by the threshold of degrees'''
    all_freq =np.zeros((1,num_bus))
    for i in range( num_bus):
        ifreq = np.shape(np.where(line[:,0] == i+1 ))[1] + np.shape(np.where(line[:,1] == i+1))[1]
        all_freq[0][i] = ifreq 
    w = [i for i in range(num_bus) if all_freq[0][i] >thres]  
    return w

# load data
def load_all_data_VI(w,rootPath, trainName ): 
    train_x,   train_labels, train_num = load_data_VI_new(w,rootPath, trainName)  
    samples,buses,_, _  = np.shape(train_x)  
    return train_x,    train_labels, train_num 

def one_hot_neib(test_labels, line_neib):
    num_class = torch.max(test_labels)  + 1
    num_sample = test_labels.shape[0]
    y_neib = torch.zeros((num_sample, num_class)) 
    for i in range(num_sample): 
        ind = (test_labels[i].cpu() ).numpy()   
        y_neib[i,  line_neib[int(ind)  ]  ] = 1  
    return y_neib  

def current_dist(rootPath):
    data = sio.loadmat(os.path.join(rootPath, 'distri_cur_nofault.mat'))
    up_real = data['up_real']
    up_imag = data['up_imag']
    down_real = data['down_real']
    down_imag = data['down_imag'] 
    up_limit = np.r_[up_real, up_imag]
    down_limit = np.r_[down_real, down_imag]
    return  up_limit, down_limit

def vol_dist(rootPath):
    data = sio.loadmat(os.path.join(rootPath, 'distri_vol_nofault.mat'))
    up_real = data['up_real']
    up_imag = data['up_imag']
    down_real = data['down_real']
    down_imag = data['down_imag'] 
    up_limit = np.r_[up_real, up_imag]
    down_limit = np.r_[down_real, down_imag]
    return  up_limit, down_limit


 