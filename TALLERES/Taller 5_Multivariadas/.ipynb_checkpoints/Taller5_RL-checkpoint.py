# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:28:23 2019

Codigo para implementar el modelo de Rgeresión Ligística (RL) elaborado por Edier Aristizabal (2019)

@author: Edier
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import seaborn as sns
import statsmodels.graphics.api as smg
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/slope_rad')
pendiente = file.GetRasterBand(1)
pendiente = pendiente.ReadAsArray()
pendiente=np.where(pendiente==-3.4028234663852886e+38,np.nan,pendiente)
pendiente=pendiente*180/np.pi
pendiente_vector=pendiente.ravel()
pendiente_vector2=np.nan_to_num(pendiente_vector)
pendiente_vector_MenM=pendiente_vector[~np.isnan(pendiente_vector)] # para eliminar  del vector los datos inf

raster = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/curvatura')
curvatura=raster.GetRasterBand(1)
curvatura=curvatura.ReadAsArray()
curvatura=np.where(curvatura==-3.4028234663852886e+38,np.nan,curvatura)
curvatura_vector=curvatura.ravel()
curvatura_vector2=np.nan_to_num(curvatura_vector)
curvatura_vector_MenM=curvatura_vector[~np.isnan(curvatura_vector)]

raster = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/aspecto')
aspecto=raster.GetRasterBand(1)
aspecto=aspecto.ReadAsArray()
aspecto=np.where(aspecto==-3.4028234663852886e+38,np.nan,aspecto)
aspecto_vector=aspecto.ravel()
aspecto_vector2=np.nan_to_num(aspecto_vector)
aspecto_vector_MenM=aspecto_vector[~np.isnan(aspecto_vector)]

raster = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/flowacum_m2')
flujo=raster.GetRasterBand(1)
flujo=flujo.ReadAsArray()
flujo=np.where(flujo==-3.4028234663852886e+38,np.nan,flujo)
flujo_vector=flujo.ravel()
flujo_vector2=np.nan_to_num(flujo_vector)
flujo_vector_MenM=flujo_vector[~np.isnan(flujo_vector)]

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/Coberturas_Arenosa.tif')
band = file.GetRasterBand(1)
coberturas = band.ReadAsArray()
coberturas=np.where(coberturas==2147483647,np.nan,coberturas)
coberturas_vector=coberturas.ravel()
coberturas_vector2=np.nan_to_num(coberturas_vector)
coberturas_vector_MenM=coberturas_vector[~np.isnan(coberturas_vector)]

cob=np.ndarray.tolist(coberturas_vector_MenM)
for i in range(174):
    a=np.random.randint(2,4)
    cob.append(a)
    
coberturas_vector_MenM=np.asarray(cob)

np.unique(coberturas)

raster = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/inventario.tif')
inventario=raster.GetRasterBand(1)
inventario=inventario.ReadAsArray()
msk=np.where(pendiente>=0,1,np.nan)
inventario=msk*inventario
inventario=np.where(inventario==1,0,inventario)
inventario=np.where(inventario==2,1,inventario)
inventario_vector=inventario.ravel()
inventario_vector_MenM=inventario_vector[~np.isnan(inventario_vector)]

#Dataframe con las variables filtradas 
d={'inventario':inventario_vector_MenM,'cobertura':coberturas_vector_MenM,'pendiente':pendiente_vector_MenM,'flujo_acum':flujo_vector_MenM,'aspecto':aspecto_vector_MenM,
   'curvatura':curvatura_vector_MenM}
x = pd.DataFrame(d)
print(list(x.columns))

resumen=x.describe()
print(resumen)

#numero de 0 y 1 d la variable dependiente
x['inventario'].value_counts()
y=x['inventario']

#Matriz de scattering utilizando seaborn
sns.pairplot(x, hue='inventario')
sns.scatterplot(x="curvatura", y="pendiente", hue="inventario", data=x)
sns.lmplot('curvatura', 'pendiente', data=x, hue='inventario', fit_reg=False);

#Histogramas
sns.distplot(x['pendiente']);
sns.distplot(x['curvatura']);
sns.distplot(x['flujo_acum']);

sns.violinplot(data=x['pendiente']);

#Histograma de frecuencia para CON y SIN MenM
x_sin=x[(x['inventario']==0)]
x_con=x[(x['inventario']==1)]

fig, ax = plt.subplots()
x_sin['pendiente'].plot.kde(ax=ax, label='Sin MenM')
x_con['pendiente'].plot.kde(ax=ax, label='Con MenM')
ax.set_xlim(0,90)
ax.set_xlabel('Pendiente (grados)', color='k', size=12)
ax.set_ylabel('Densidad', color='k', size=12)
ax.legend(loc=1, fontsize=10)
ax.tick_params('y', colors='k', labelsize= 10)

#Media de cada variabel separada por sin MeM y con MenM
x.groupby('inventario').mean()

#figura de barras cruzada de geologia de acuerdo con la variabel dependiente
pd.crosstab(x.cobertura,x.inventario).plot(kind='bar')

x.drop('inventario', axis=1, inplace=True)
dummy_coberturas=pd.get_dummies(x['cobertura'],prefix='cob')
column_name=x.columns.values.tolist()
column_name.remove('cobertura')
x1=x[column_name].join(dummy_coberturas)
x1.head()

#Matriz de correlacion de Pearson y mapa de calor
MatCorre=DataFrame(x.corr())
smg.plot_corr(MatCorre, xnames=list(MatCorre.columns)) ;

#Dataframe de las variables de todo el mapa
f={'cobertura': coberturas_vector2,'pendiente':pendiente_vector2,'flujo_acum':flujo_vector2,'aspecto':aspecto_vector2, 'curvatura':curvatura_vector2}
x_map=pd.DataFrame(f)
dummy_coberturas=pd.get_dummies(x_map['cobertura'],prefix='cob')
column_name=x_map.columns.values.tolist()
column_name.remove('cobertura')
x_map=x_map[column_name].join(dummy_coberturas)
x_map=x_map.drop('cob_0.0',axis=1)



import statsmodels.api as sm

logit_model=sm.Logit(y,x1)
result=logit_model.fit()
print(result.summary())

from sklearn import linear_model

model=linear_model.LogisticRegression()

#Para seleccioanr variables automaticamente
from sklearn import datasets
from sklearn.feature_selection import RFE
#Se peude seleccioanr le numero de variables que se desee, en este caso se definieron 5
rfe=RFE(model,5)
rfe=rfe.fit(x1,y)
print(rfe.support_)
print(rfe.ranking_)

model.fit(x1,y)
print(model.coef_)
print(model.intercept_)
pd.DataFrame(zip(x1.columns,np.transpose(model.coef_)))

map_predic=model.predict(x_map)

#Evaluacion general del modelo
model.score(x1,y)

#Para utilizar validacion cruzada con el 80/20%
x_train,x_validation,y_train,y_validation=model_selection.train_test_split(x1, y, test_size=0.2, random_state=7)
print('Tamaño de variables de entrenamiento:', x_train.shape)
print('Tamaño de labels de entrenamiento:', y_train.shape)
print('Tamaño de variables de validación:', x_validation.shape)
print('Tamaño de labels de validación:', y_validation.shape)

#se crea el modelo de regresion logistica utilizando solo la base de datso de entrenamiento (80%) generada
model1=linear_model.LogisticRegression()
model1.fit(x_train,y_train)

#Las probabilidades entonces para las celdas de validacion (20%) son
y_predic=model1.predict(x_validation) #utiliza por defecto el valro de 0,5, por enciam es uno y por debajo es 0
y_probs=model1.predict_proba(x_validation)

#Para cambiar el umbral de 0,5 por defecto se aplica
prob=y_probs[:,1]
probs_df=pd.DataFrame(prob)
probs_df['predict']=np.where(probs_df[0]>=0.10,1,0)
probs_df.head(10)

#dos formas para evalaur la prediccion del modelo, dan igual.
model1.score(x_validation,y_validation)
print (metrics.accuracy_score(y_validation, y_predic))

#validacion cruzada para generalizar el modelo, es decir evaluar el overfitting dle modelo
kfold = model_selection.KFold(n_splits=10, random_state=7)
cv_results = model_selection.cross_val_score(model1, x_train, y_train, cv=kfold, scoring='accuracy')
#Precision de cada uno de las kfold corridas
print (cv_results)
print (cv_results.mean())
msg = "%s: %f (%f)" % ('Precisión media y desviación estandar del modelo', cv_results.mean(), cv_results.std())
print (msg)

print(confusion_matrix(y_validation, y_predic))
print(classification_report(y_validation,y_predic))

#Curva ROC y AUC utilizando metrics.roc_curve
fpr, sensitivity, _=metrics.roc_curve(y_validation,prob)
plt.plot(fpr,sensitivity,color='r')
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y,linestyle='--',color='b')
plt.grid(True)
plt.xlabel('(1-Specificity)')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
auc=str(round(metrics.auc(fpr,sensitivity)))
plt.text(x=0.8,y=0.1,s='AUC='+auc+'%',size=12)

#construcción de la Curva ROC
S=[] #Sensitivity
E=[] #Specificity
probs_df['predict']=np.where(probs_df[0]>=0.09,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.08,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.07,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.06,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.05,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.04,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.03,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.02,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.01,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

probs_df['predict']=np.where(probs_df[0]>=0.009,1,0)
cm=confusion_matrix(y_validation,probs_df['predict'])
s=cm[1,1]/(cm[1,1]+cm[1,0])
S.append(s)
e=cm[0,1]/(cm[0,0]+cm[0,1])
E.append(e)

#Curva ROC
plt.plot(E,S,marker='o')
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y,linestyle='--',color='b')
plt.grid(True)
plt.xlabel('(1-Specificity)')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')

#prediccion para todo el mapa
IS=model.predict(x_map)
IS_prob=model.predict_proba(x_map)
IS_prob2=IS_prob[:,1]

#mascara para crear el mapa de la cuenca
file = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/slope_rad')
pendiente = file.GetRasterBand(1)
pendiente = pendiente.ReadAsArray()

#Convertir el vector de resultados a la matriz del mapa de la cuenca a aprtir de la matriz de pendiente
IS_2=IS_prob2.reshape(pendiente.shape)
IS_3=np.where(pendiente==-3.4028234663852886e+38,np.nan,IS_2)
plt.imshow(IS_3)
plt.colorbar()