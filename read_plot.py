# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:40:43 2022

@author: fso
"""
from json import load as jsonload
import numpy as np
import matplotlib.pyplot as plt
import analysis as al
from scipy import signal

# LOAD DATA AND FORMAT
filename = './simple_ELM-output/model_output.json'

with open(filename) as jsonfile:
    data = jsonload(jsonfile)
    
# numpify
print("VARIABLES AVAILABLE TO PLOT:")
for key in data:
    datum = data[key]
    if isinstance(datum, list):
        print("    ",key)
        data[key] = np.asarray(datum)

print("Site:",  data['site'])
for kwarg in data['kwargs']:
    print(kwarg,'   :   ',data['kwargs'][kwarg])
    
# CALCULATIONS    
days_per_year = 365
pft_num, day_num = data['gpp_pft'].shape
t = np.arange(day_num)/days_per_year

# find total c biomass
pools = ['leafc_pft', 'leafc_stor_pft', 'frootc_pft',
         'frootc_stor_pft', 'livestemc_pft', 'deadstemc_pft', 
         'livecrootc_pft','deadcrootc_pft']




pft = 0

cmass = sum(data[pool] for pool in pools)
nsc_mass = data['cstor_pft']
cmass += nsc_mass 

nsc = nsc_mass[pft]/cmass[pft] # fraction

A = data['gpp_pft'][pft]/cmass[pft]*days_per_year
R = (data['gpp_pft'][pft] - data['npp_pft'][pft])/cmass[pft]*days_per_year
G = A - R - al.forward_diff(nsc_mass,t)[pft]/cmass[pft]


N = A - data['mr_pft'][pft]/cmass[pft]*days_per_year
azote = data['nstor_pft'][pft]
for var in [A,R,G, nsc, cmass[pft],nsc_mass[pft], N, azote]:
    var[...] = al.lowpass(var)


npp = A - R

# xs = al.delay(nsc[0,::30],2)
# al.pairs(xs)
# PLOTTING
fig ,ax = plt.subplots(2,3,sharex=True)
ax[0,0].plot(t,A,'g',label='A/cmass')
ax[0,0].plot(t,N,'orange',label='(A - Rm)/cmass')
# ax[0,0].plot(t[180::365],A[pft,180::365],'g--')
ax[1,0].plot(t,R,'r', label= 'resp/cmass')
ax[1,0].plot(t,G,'b', label = 'growth/cmass')

ax[0,1].plot(t,nsc,'k',label='[nsc]')

ax[1,1].plot(t,cmass[pft],'k', label='C mass')

ax[1,1].twinx().plot(t,nsc_mass[pft],color='orange',label='cstor')


ax[0,2].plot(t, azote, label='nstor')        
for a in ax:
    for b in a:
        b.grid()
        b.legend()
        
        

fig, ax = plt.subplots(1,3)

nsc_dot = al.center_diff(nsc,t)
azote_dot = al.center_diff(azote, t)
N_dot = al.center_diff(N, t)

nsc_trend, nsc_seasonal = al.cyl_embed(nsc,t)
# # ind = al.cross(nsc_dot, direction = 0)
ax[0].plot(N, al.diff(nsc_seasonal,t))
ax[1].plot(azote,al.diff(nsc_trend,t))
ax[2].plot(nsc_seasonal, al.center_diff(nsc_seasonal,t))

# al.pair(azote_dot, N, nsc, nsc_dot)
# al.svd_pair(azote_dot, N, nsc, nsc_dot)

# al.svd_biplot(azote_dot,N,nsc_trend, nsc_seasonal)
plt.show()