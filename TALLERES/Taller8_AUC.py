# -*- coding: utf-8 -*-
"""
Created on Mon May 13 04:04:10 2019

@author: Edier Aristizabal
"""


import numpy as np
import pandas as pd
from osgeo import gdal
from numpy import cumsum

#importar mapa de factor de seguridad FS
FS_hazard  = np.genfromtxt('G:/My Drive/PAPERS/ELABORACION/SHIA_George/resultados/FS_AMEACA.txt')
FS_vector=FS_hazard.ravel()
FS_vector1=FS_vector[FS_vector!=-9999]
FS_max=FS_vector1.max()
FS_min=FS_vector1.min()

#importar inventario de movimientso en masa
raster = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/inventario.tif')
inventario=raster.GetRasterBand(1)
inv_raster=inventario.ReadAsArray()
inv_raster=np.where(FS_hazard==-9999,np.nan,inv_raster)
inv_vector=inv_raster.ravel()
inv_vector1=np.where(FS_vector==-9999,-9999,inv_vector)
inv_vector1=inv_vector1[inv_vector1!=-9999]
inv_vector1=np.where(inv_vector1==5,1,inv_vector1)

#hist, bins = np.histogram(FS_vector1, density=True,bins=[FS_min,0,0.5,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,FS_max])

FS_vector1=pd.Series(FS_vector1)

a=b=c=d=e=f=g=h=i=j=k=l=m=n=o=p=0
for row in FS_vector1:
        if (FS_min <= row) & (row < 0):
          a+=1
        elif (0 <= row) & (row < 0.5):
          b+=1
        elif (0.5 <= row) & (row < 1):
          c+=1
        elif (1 <= row) & (row < 1.1):
          d+=1
        elif (1.1 <= row) & (row < 1.2):
          e+=1
        elif (1.2 <= row) & (row < 1.3):
          f+=1
        elif (1.3 <= row) & (row < 1.4):
          g+=1
        elif (1.4 <= row) & (row < 1.5):
          h+=1
        elif (1.5 <= row) & (row < 1.6):
          i+=1
        elif (1.6 <= row) & (row < 1.7):
          j+=1
        elif (1.7 <= row) & (row < 1.8):
          k+=1
        elif (1.8 <= row) & (row < 1.9):
          l+=1
        elif (1.9 <= row) & (row < 2):
          m+=1
        elif (2 <= row) & (row < 3):
          n+=1
        elif (3 <= row) & (row < 4):
          o+=1
        if (row>=4):
          p+=1

total=a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p

x_FS=[]
x1=100*p/total
x_FS.append(x1)
x2=100*o/total
x_FS.append(x2)  
x3=100*n/total
x_FS.append(x3)
x4=100*m/total
x_FS.append(x4)
x5=100*l/total
x_FS.append(x5)
x6=100*k/total
x_FS.append(x6)
x7=100*j/total
x_FS.append(x7)
x8=100*i/total
x_FS.append(x8)
x9=100*h/total
x_FS.append(x9)
x10=100*g/total
x_FS.append(x10)
x11=100*f/total
x_FS.append(x11)
x12=100*e/total
x_FS.append(x12)
x13=100*d/total
x_FS.append(x13)
x14=100*c/total
x_FS.append(x14)
x15=100*b/total
x_FS.append(x15)
x16=100*a/total
x_FS.append(x16)

x_FS = list(cumsum(x_FS))

produc=FS_vector1*inv_vector1

a1=b1=c1=d1=e1=f1=g1=h1=i1=j1=k1=l1=m1=n1=o1=p1=0
for row in produc:
        if (FS_min <= row) & (row < 0):
          a1+=1
        elif (0 < row) & (row < 0.5):
          b1+=1
        elif (0.5 <= row) & (row < 1):
          c1+=1
        elif (1 <= row) & (row < 1.1):
          d1+=1
        elif (1.1 <= row) & (row < 1.2):
          e1+=1
        elif (1.2 <= row) & (row < 1.3):
          f1+=1
        elif (1.3 <= row) & (row < 1.4):
          g1+=1
        elif (1.4 <= row) & (row < 1.5):
          h1+=1
        elif (1.5 <= row) & (row < 1.6):
          i1+=1
        elif (1.6 <= row) & (row < 1.7):
          j1+=1
        elif (1.7 <= row) & (row < 1.8):
          k1+=1
        elif (1.8 <= row) & (row < 1.9):
          l1+=1
        elif (1.9 <= row) & (row < 2):
          m1+=1
        elif (2 <= row) & (row < 3):
          n1+=1
        elif (3 <= row) & (row < 4):
          o1+=1
        if (row>=4):
          p1+=1
          
total1=a1+b1+c1+d1+e1+f1+g1+h1+i1+j1+k1+l1+m1+n1+o1+p1

y_FS=[]
y1=100*p1/total1
y_FS.append(y1)
y2=100*o1/total1
y_FS.append(y2)  
y3=100*n1/total1
y_FS.append(y3)
y4=100*m1/total1
y_FS.append(y4)
y5=100*l1/total1
y_FS.append(y5)
y6=100*k1/total1
y_FS.append(y6)
y7=100*j1/total1
y_FS.append(y7)
y8=100*i1/total1
y_FS.append(y8)
y9=100*h1/total1
y_FS.append(y9)
y10=100*g1/total1
y_FS.append(y10)
y11=100*f1/total1
y_FS.append(y11)
y12=100*e1/total1
y_FS.append(y12)
y13=100*d1/total1
y_FS.append(y13)
y14=100*c1/total1
y_FS.append(y14)
y15=100*b1/total1
y_FS.append(y15)
y16=100*a1/total1
y_FS.append(y16)

y_FS = list(cumsum(y_FS))

from sklearn.metrics import auc
import matplotlib.pyplot as plt

fig, ax1=plt.subplots(figsize=(7,5))
plt.plot(y_FS, x_FS, linestyle='-', linewidth=1.5, color='black', label='Curva ROC')
plt.xlabel("FPR", fontsize=22)
plt.ylabel("TPR", fontsize=22)
ax1.set_ylim(0, max(x_FS))
ax1.set_xlim(0, max(y_FS))
plt.yticks(np.arange(0,max(y_FS)+1,max(y_FS)/5.))
plt.xticks(np.arange(0,max(x_FS)+1,max(x_FS)/5.))
ax1.tick_params('y', colors='k', labelsize= 20, length=2)
ax1.tick_params('x', colors='k', labelsize= 20, length=2)
fig.tight_layout()
plt.grid(True)

'''
ax1.fill_between(y_FS[0,74:0,98],x_FS[0,74:0,98],0, facecolor='r', alpha=0.6)
ax1.fill_between(x_FS[0,64:0,75],y_FS[0,64:0,75],0, facecolor='xkcd:yellow',alpha=0.8)
ax1.fill_between(x_FS[0:0,65],y_FS[0:0,65],0, facecolor='xkcd:green',alpha=0.4)

ax1.annotate('Alta',xy=(15,30), size=12)
ax1.annotate('Media',xy=(50,30), size=12)
ax1.annotate('Baja',xy=(85,30), size=12)
'''

area1=round(float(format(auc(y_FS,x_FS)))/max(x_FS),1)

AUC1=str(area1)

ax1.annotate('AUC='+AUC1+'%',xy=(0.5*max(x_FS),0.3*max(x_FS)), size=22)

#Generar matriz de confusion 
FS_hazard=np.where(FS_hazard==-9999,np.nan,FS_hazard)
FS_hazard_r=np.where(FS_hazard<=1,3,FS_hazard) #3: celdas estables para el modelo
FS_hazard_re=np.where(FS_hazard>1,4,FS_hazard_r) # 4: Celdas indestables para el modelo
inv_raster_r=np.where(inv_raster==5,2,inv_raster) #2: celdas con MenM en el inventario
inv_raster_re=np.where(inv_raster==0,1,inv_raster_r) #1: Celdas sin MenM en el inventario

matconf=FS_hazard_re*inv_raster_re

a=np.count_nonzero(matconf == 3) # Falsos positivos= celdas sin MenM y con FS>1
b=np.count_nonzero(matconf == 4) # Verdaderos negativos= celdas sin MenM y con FS<1
c=np.count_nonzero(matconf == 6) # Falsos negativos= celdas con MenM y con FS<1
d=np.count_nonzero(matconf == 8) # Verdaderos Positivos= celdas con MenM y con FS>1
confmat=np.array([[d,a],[c,b]])

import seaborn as sns
sns.heatmap(confmat.T, square=True, annot=True, fmt="d",cbar=False, linewidths=.5, cmap="YlGnBu",
            xticklabels=['1','0'],
            yticklabels=['1','0'])
plt.xlabel('Inventario')
plt.ylabel('Modelo');
