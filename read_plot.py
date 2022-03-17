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
filename = 'c:/users/fso/documents/Github/simple_ELM-output/novary_test_v2.json'
print(filename)
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

gpp = data['gpp_pft'][pft]
A = gpp/cmass[pft]*days_per_year
R = (data['gpp_pft'][pft] - data['npp_pft'][pft])


azote = data['nstor_pft'][pft]
# for var in [A,R,G, nsc, cmass[pft],nsc_mass[pft], N, azote,ncl[pft],gpp]:
#     var[...] = al.lowpass(var, 100)


# xs = al.delay(nsc[0,::30],2)
# al.pairs(xs)
# PLOTTING
fig ,ax = plt.subplots(2,3,sharex=True)
ax[0,0].plot(t,gpp,'g',label='A')
# ax[0,0].plot(t[180::365],A[pft,180::365],'g--')
ax[1,0].plot(t,R,'r', label= 'R')

ax[0,1].plot(t,nsc,'k',label='[nsc]')

ax[1,1].plot(t,cmass[pft],'k', label='C mass')

ax[1,2].plot(t,nsc_mass[pft],color='orange',label='cstor')


ax[0,2].plot(t, azote, label='nstor')        
# ax[1,2].plot(t, ncl[pft])
for a in ax.flatten():
    a.grid()
    a.legend()
        # b.legend()
        
        

# fig, axes = plt.subplots(2,2)

# nsc_mass_dot = al.forward_diff(nsc_mass[pft],t)
# ncl_dot = al.forward_diff(ncl[pft], t)
# # N_dot = al.center_diff(N, t)

# # nsc_trend, nsc_seasonal = al.cyl_embed(nsc,t)
# # # # ind = al.cross(nsc_dot, direction = 0)
# # # ax[0].plot(N, al.diff(nsc_seasonal,t))
# # # ax[1].plot(azote,al.diff(nsc_trend,t))
# # # ax[2].plot(nsc_seasonal, al.center_diff(nsc_seasonal,t))

# maxima = al.cross(nsc_mass_dot,0,-1)
# lwr,upr, bnd = al.envelope(nsc_mass[pft],t,nsc_mass_dot)
# valid = (t >= bnd[0])*(t <= bnd[1])
# ts = t[valid]
# avg = (lwr(ts) + upr(ts))/2
# amp = (-lwr(ts) + upr(ts))/2

# nsc_mass_norm = al.normalize_by_envelope(nsc_mass[pft],t,nsc_mass_dot)


# azote_dot = al.forward_diff(azote,t)
# nlwr,nupr, nbnd = al.envelope(azote,t,azote_dot)
# nvalid = (t >= nbnd[0])*(t <= nbnd[1])
# nts = t[nvalid]

# navg = (nlwr(nts) + nupr(nts))/2
# namp = (-nlwr(nts) + nupr(nts))/2

# ind = valid & nvalid 

# azote_norm = al.normalize_by_envelope(azote, t, azote_dot)

# axes[0,0].plot(ncl[pft],nsc_mass[pft],'C0')

# axes[0,0].plot(ncl[pft][maxima],nsc_mass[pft][maxima],'C1--o')
# axes[0,0].axline((700.,700.),slope=1,color='k',ls='--')
# axes[0,1].plot(ncl_dot,nsc_mass_dot,'C0')
# axes[0,1].plot(ncl_dot[maxima],nsc_mass_dot[maxima],'C1--o')

# axes[0,1].axhline(0,ls='--',color='k')
# axes[0,1].axvline(0,ls='--',color='k')

# axes[1,0].plot(gpp[valid], nsc_mass_norm[1],'C2')
# # axes[1,1].plot(gpp[nvalid], azote_norm[1],'C3')
# # axes[1,1].plot(n[maxima],nsc_mass_dot[maxima])
# axes[1,1].plot(azote_dot, nsc_mass_dot)
# for ax in axes.flatten():
#     ax.grid()
# # # al.pair(azote_dot, N, nsc, nsc_dot)
# # # al.svd_pair(azote_dot, N, nsc, nsc_dot)

# # al.derivplot(azote,t,fac=True)
# # # al.svd_biplot(azote_dot,N,nsc_trend, nsc_seasonal)

# fig, axes =plt.subplots(1,4)
# # axes[0].plot(t, nsc_mass[pft])
# # axes[0].plot(ts,lwr(ts),'C1--')
# # axes[0].plot(ts,upr(ts), 'C1--')
# # axes[0].plot(ts,avg, 'C3--')

# axes[0].plot(gpp,al.diff(gpp,t),'C2')
# axes[1].plot(nsc_mass[pft],nsc_mass_dot,'C0')
# axes[2].plot(nsc_mass_norm[1],al.diff(nsc_mass_norm[1],nsc_mass_norm[0]),'C1')
# axes[3].plot(avg,amp,'C3')
# for ax in axes.ravel():
#     ax.grid()

# axes[1].plot(t, azote)

# axes[1].plot(nts, nlwr(nts), 'C1--')
# axes[1].plot(nts, nupr(nts), 'C1--')

# axes[1].plot(nts, navg, 'C3--')

plt.show()