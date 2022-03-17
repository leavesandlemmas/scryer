# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:16:57 2022

@author: fso
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d


# DERIVATIVE APPROXIMATES

def forward_diff(y,t):
    out =np.empty_like(y)
    out[...,:-1] = (y[...,1:] - y[...,:-1])/(t[1:] - t[:-1])
    out[...,-1] = (y[...,-1] - y[...,-2])/(t[-1] - t[-2])
    return out

def center_diff(y,t):
    out =np.empty_like(y)
    out[...,1:-1] = (y[...,2:] - y[...,:-2])/(t[2:] - t[:-2])
    out[...,0] = (y[...,1] - y[...,0])/(t[1] - t[0])
    out[...,-1] = (y[...,-1] - y[...,-2])/(t[-1] - t[-2])
    return out

def fourier_diff(y,t):
    dt = t[1] - t[0]
    ws = 2*np.pi*np.fft.fftfreq(t.shape[0],dt)
    yf = np.fft.fft(y)
    return np.fft.ifft(1j*ws*yf).real

def diff(y,t,n=1):
    k= 0
    out = y.copy()
    while k < n: 
        out[...] = center_diff(out,t)
        k += 1
    return out 



# SMOOTHING

w = signal.boxcar(30)
w /= w.sum()
def moving_average(y):
    return signal.convolve(y,w,mode='same')

def lowpass(y, n = 365//2):
    z = np.fft.rfft(y)
    z[n:] = 0.
    return np.fft.irfft(z)

def smooth(x,t,n=24):
    y = lowpass(x)
    return signal.resample(y,num=y.shape[0]*n,t=t)

def resonant_filter(y, freq = 25, harmonic=10):
    yf = np.fft.rfft(y)
    zf = np.zeros_like(yf)
    zf[::freq][:harmonic] = yf[::freq][:harmonic]
    out = np.fft.irfft(zf)
    return out

# PHASE SPACE EMBEDDINGS

def embed(*args):
    return np.stack([arg for arg in args])

def delay(x,shift=1):
    N = x.shape[0]
    shape = (1+shift,N-shift)
    out = np.empty(shape,dtype=x.dtype)
    for i in range(shift+1):
        out[i] = x[i:N - shift + i]
    return out

def derivative_embeding(x,t,n=1):
    N = x.shape[0]
    shape = (n+1,N)
    out = np.empty(shape, dtype=x.dtype)
    out[0]= x
    for i in range(n):
        out[i+1] = center_diff(out[i],t)
    return out
    
def cyl_embed(y,t, n = 10):
    '''
    Return a cylinder embedding of a 1D time-series.
    Decomposition of y = y_trend + y_seasonal using a fourier transform
    to identify low frequency components. A linear detrending is applied
    first to minimize the boundary effect in the fourier transform.

    Parameters
    ----------
    y : array_like, 1D
        Time-series.
    t : array_like, 1D
        Times of measurement, assumes evenly spaced.
    n : positive integer
        Number of low frequency components to keep.

    Returns
    -------
    y_trend : array_like, 1D
        Time-series with seasonal oscillations removed.
    y_seasonal : TYPE
        Time-series of seasonal oscillations.

    '''
    a = y.mean()
    tbar =t.mean()
    b= np.dot(y-a , t -tbar )/np.dot(t-tbar,t-tbar)
    y_per = y - b*(t-tbar)
    y_per_fft = np.fft.rfft(y_per)

    y_per_trend = np.fft.irfft(y_per_fft[:n],y.shape[0]) 
    y_trend = y_per_trend + b*(t-tbar)
    y_seasonal = y - y_trend
    return y_trend, y_seasonal



# POINCARE MAPS

# def nullcline_pmap(x,t):
#     dxdt = center_diff(x,t)
#     ind = []
#     for i in range(x.shape[-1]):
#         dxdt[i]


def cross(x, a=0, direction=0):
    '''
    Return indices of x which cross the value a; if a in [x[i], x[i+1]]
    then i is returned. The direction 

    Parameters
    ----------
    x : array_like, 1D
        Time-series.
    a : float, optional
        Crossing point. The default is 0.
    direction : integer, optional
        Direction of crossing to detect. Positive values indicate
        crossings from negative to positve are returned. Zero indicates 
        all crossings regardless of direction are returned. Integers are
        preferred for checking equality with zero.
        The default is 0. 

    Returns
    -------
    crossing : array_like, 1D
        list of indices where time-series crosses; first index of
        interval is returned.

    '''
    s = np.sign(x-a)
    if direction == 0:
        return np.where(s[1:]*s[:-1] < 0)      
    grad = s[1:] - s[:-1]
    if direction < 0:
        return np.where((s[1:]*s[:-1] < 0)  & (grad < 0))
    else:
        return np.where((s[1:]*s[:-1] < 0)  & (grad > 0))

def envelope(x,t=None,xdot=None, bounds=True):
    if xdot is None:
        if t is None:
            t = np.arange(x.shape[0])
        xdot = diff(x,t)
    maxima = cross(xdot,0,-1)
    minima = cross(xdot,0, 1)
    upr = interp1d(t[maxima], x[maxima], kind = 'quadratic')
    lwr = interp1d(t[minima], x[minima], kind = 'quadratic')
    if bounds:
        tmax = t[maxima]
        tmin = t[minima]
        bnd = [max(tmin.min(), tmax.min()), min(tmin.max(),tmax.max())]
        return lwr, upr, bnd
    else:
        return lwr, upr

def normalize_by_envelope(x, t=None, xdot=None, return_indices=True):
    if t is None:
        t = np.arange(x.shape[0])
    lwr, upr, bnd = envelope(x, t, xdot)
    valid = (t >= bnd[0]) & (t <= bnd[1])
    l = lwr(t[valid])
    u = upr(t[valid])
    y = (x[valid]-l)/(u-l)
    out = t[valid], y
    if return_indices:
        out += (valid,)
    return out 
# PLOTTING FUNCTIONS

def pair(*args,**kwargs):
    x = np.stack(args)
    return pairs(x,**kwargs)

def pairs(x,*args,**kwargs):
    mdim,ndim = x.shape
    fig, axes = plt.subplots(mdim,mdim)
    for i in range(mdim):
        for j in range(mdim):
            axes[i,j].plot(x[i],x[j],*args,**kwargs)
            axes[i,j].grid()
            
    return fig, axes


def svd_pair(*args, **kwargs):
    x = np.stack(args)
    svd = np.linalg.svd(x)
    y = svd[2][:x.shape[0]]
    return pairs(y, **kwargs)      

def svd_biplot(*args, ax=None, **kwargs):
    x = np.stack(args)
    svd = np.linalg.svd(x)
    y = svd[2][:2]
    if ax is None:
        fig, ax = plt.subplots()
        ax.plot(y[0],y[1],**kwargs)
        return fig, ax, y
    else:
        ax.plot(y[0],y[1])
        return y
    
def derivplot(y,t,n=4,fac=False):
    fig, ax = plt.subplots(1,n)
    x = y.copy()
    xdot = diff(x,t)
    m = 2
    for i in range(n):
        ax[i].plot(x,xdot)
        x,xdot = xdot, diff(xdot,t)
        if fac:
            xdot /= m
            m *= (i + 3)
        print(m)
    return fig, ax
