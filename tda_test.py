# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:14:25 2022

@author: fso
"""
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_point_cloud
import numpy as np
from matplotlib import pyplot as plt


def make_cylinder(num = 200):
    pts = np.empty((num,3))
    w = 2*np.pi
    t = np.random.rand(num)
    s = np.random.rand(num)
    z = 2*np.random.rand(num) -1 
    pts[:,0] = np.cos(w*t)  + 2*np.sign(z)
    pts[:,1] = np.sin(w*t)
    pts[:,2] = s
    return pts
    

cyl = make_cylinder(100)

VR = VietorisRipsPersistence(homology_dimensions=[0,1,2])
# help(VR)
point_clouds = [cyl]
Xt = VR.fit_transform(point_clouds)

# diagrams = VR.plot(Xt, sample=0)


fig, ax =plt.subplots()
c = Xt[0,:,2]
cmax = round(c.max())
x = Xt[0,:,0]
y = Xt[0,:,1]
pers= y - x
ind = np.argsort(pers)
print(Xt[0,ind[-10:]])
ax.scatter(x,y,c=c,cmap='jet')
ax.axline((0,0),slope=1,color='k',ls='--')
ax.grid()

fig, ax = plt.subplots(cmax+1,1)
for i in range(Xt.shape[1]):
    cn = round(c[i])
    ax[cn].plot([x[i],y[i]],[i]*2,'-',color="C{}".format(cn))

# print(Xt)
# plot_point_cloud(cyl)
plt.show()