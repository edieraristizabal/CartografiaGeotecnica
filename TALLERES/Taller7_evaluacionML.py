# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:53:32 2019

@author: Edier
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
plt.imshow(curvatura)
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
d={'inventario':inventario_vector_MenM,'pendiente':pendiente_vector_MenM,'flujo_acum':flujo_vector_MenM,'aspecto':aspecto_vector_MenM,
   'curvatura':curvatura_vector_MenM}
x = pd.DataFrame(d)
print(list(x.columns))
y=x['inventario']
x.drop('inventario', axis=1, inplace=True)
x.head()

#Dataframe de las variables de todo el mapa
f={'pendiente':pendiente_vector2,'flujo_acum':flujo_vector2,'aspecto':aspecto_vector2, 'curvatura':curvatura_vector2}
x_map=pd.DataFrame(f)

#Separacion de la base de datos en cedlas de entrenamiento para evaluacion del desempeño y celdas de validacion para prediccion del modelo
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)
print('Tamaño de variables de entrenamiento:', X_train.shape)
print('Tamaño de labels de entrenamiento:', y_train.shape)
print('Tamaño de variables de validación:', X_test.shape)
print('Tamaño de labels de validación:', y_test.shape)


##################################REDES NEURONALES###################################################
from sklearn.neural_network import MLPClassifier

#Estandarizar las variables
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Implementar el modelo de Redes neuronale sdenominado Multilayer Perceptron, en este caso dos capas escondidas de 5 y 2 neuronas
mlp = MLPClassifier(hidden_layer_sizes=(5,2),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

#prediccion de todo el mapa
IS=mlp.predict(x_map)

#mascara para crear el mapa de la cuenca
file = gdal.Open('G:/My Drive/ANALISIS ESPACIAL APLICADO/datos/raster/slope_rad')
pendiente = file.GetRasterBand(1)
pendiente = pendiente.ReadAsArray()

#Convertir el vector de resultados a la matriz del mapa de la cuenca a aprtir de la matriz de pendiente
IS_2=IS.reshape(pendiente.shape)
IS_3=np.where(pendiente==-3.4028234663852886e+38,np.nan,IS_2)
plt.imshow(IS_3)
plt.colorbar()

#Matriz de confusion y reporte de la clasificacion
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
len(mlp.coefs_[0])
len(mlp.coefs_[1])
len(mlp.intercepts_[0])

###################################DECISION TREE###################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=8)

# Train the model on training data
tree.fit(X_train, y_train)

print(f'El Arbol de decisión tiene {tree.tree_.node_count} nodos con una profundidad máxima de {tree.tree_.max_depth}.')

# Para exportar como imagen el arbol de decision, priemro Export as dot
export_graphviz(tree, 'tree.dot', rounded = True, feature_names = X_train.columns, class_names = ['0', '1'], filled = True)
# Use dot file to create a graph
import pydot
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('G:\My Drive/tree.png')

#Obtener las predicciones y probabilidad de las predicciones para lso datos de entrenamiento y los datos de validación
train_probs=tree.predict_proba(X_train)[:,1]
probs = tree.predict_proba(X_test)[:, 1]

train_predictions = tree.predict(X_train)
predictions = tree.predict(X_test)

#prediccion de todo el mapa
IS=tree.predict(x_map)
IS_Pro=tree.predict_proba(x_map)[:,1]

#Evaluacion del desempeño y capacidad de prediccion del modelo
from sklearn.metrics import classification_report
print(f'Model Accuracy: {tree.score(X_train, y_train)}') #para los datso de entrenamiento
print(f'Model Accuracy: {tree.score(X_test, y_test)}') # para los datos de validacion
print(classification_report(y_test, predictions))

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')
print(f'Baseline ROC AUC: {roc_auc_score(y_test, [1 for _ in range(len(y_test))])}')


########################################RANDOM FOREST#################################################
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)

#Para predecir todo el mapa
IS=rf.predict(x_map)

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export as dot
export_graphviz(tree, 'tree.dot', rounded = True, feature_names = X_train.columns, class_names = ['0', '1'], filled = True)
# Use dot file to create a graph
import pydot
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('G:\My Drive/tree.png')

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)

# Extraer un arbol del bosque
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = X_train.columns, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('G:\My Drive/small_tree.png');

#Para saber la importancia de las variable sutilizadas
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#Evaluacion del error y del modelo
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'valor')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors /( y_test))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#############################################ANALISIS DISCRIMINANTE#######################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDA = LinearDiscriminantAnalysis(n_components=2)
data_projected = LDA.fit_transform(X_train,y_train)
print(data_projected.shape)

# PLot the transformed data
markers = ['s','x']
colors = ['r','g']
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for l,m,c in zip(np.unique(y_train),markers,colors):
    ax0.scatter(data_projected[:,0][y_train==l],data_projected[:,1][y_train==l],c=c,marker=m)
plt.xlabel('Linear discriminat 1')
plt.ylabel('Linear discriminat 2')
plt.legend(loc='lower right')
plt.tight_layout()

#predicting the test set results
y_pred = LDA.predict(X_test)

#Para predecir todo el mapa
IS=LDA.predict(x_map)

#Matriz de confusion
from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

#calcular la probabildaid apra cada clase
y_prob=LDA.predict_proba(X_test)
IS_PRO=LDA.predict_proba(x_map)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(LDA.score(X_test, y_test)))

#################################################SUPPORT VECTOR MACHINE######################################
from sklearn.svm import SVC # "Support vector classifier"

model = SVC(kernel='linear', C=1E10)
model.fit(X_train, y_train)

#Para obtener lso puntos que separan conocidos como Support vectors
model.support_vectors_

#Para predecir con lso datso de test
y_pred=model.predict(X_test)

#Para predecir todo el mapa
IS=model.predict(x_map)

#Evaluacion del modelo
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,target_names=X_columns))

###############################FUNCION PARA MATRIZ DE CONFUSION################################

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


cm = confusion_matrix(X_test, predictions)
plot_confusion_matrix(cm, classes = ['Sin MenM', 'Con MenM'],
                      title = 'Matriz de confusión')
