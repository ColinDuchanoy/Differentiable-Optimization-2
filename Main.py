# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:42:24 2022

@author: colin


"""
#%%
"""
                BE : Optimisation différentiable 2
"""
#%%
import numpy as np
import Biblio as bbl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import cv2

#%% 
"""
-------------------1 Les moindres carrées multi-classes avec régularisation-------------------
"""    
#%%
"""
1.1 Version linéaire
"""
#%% 
"""
Verion app global MC
"""     
#%% Importation données Iris
Xbrut=np.genfromtxt("iris.csv" , dtype=str , delimiter=',') #données brutes
train_data=Xbrut[1 : , : -1].astype('float')
train_labels=Xbrut[1 : , -1].astype('int')

nbcat=bbl.nbcat(train_labels)
rho=1
epsilon=10**-2

train_data=bbl.ACP(train_data , 2)

#%%  Importation des données 1
train_data = np.load("train_data1.npy")
train_labels = np.load("labels1.npy")

nbcat=bbl.nbcat(train_labels)
rho=1
epsilon=10**-2

#%%  Importation des données 2
train_data = np.load("train_data2.npy")
train_labels = np.load("labels2.npy")

nbcat=bbl.nbcat(train_labels)
rho=1
epsilon=10**-2


#%% Affichage des points de la base de donnée
fig, ax = plt.subplots()
ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,cmap=plt.cm.Accent,marker='+')
ax.set_title("Données d'apprentissage")
plt.show()

#%% Calcul de la matrice d'aprentissage 
W = bbl.app_global_MC(train_data,train_labels,nbcat,rho,epsilon)

#%% Affichage de l'apprentissage
fig,ax=plt.subplots()
ax.scatter(train_data[:,0],train_data[:,1],c=train_labels,cmap=plt.cm.Accent,marker='+')
ax.set_title("Résultat de l'apprentissage")
resol=00 #npmbre de pixels du tracer des limites des zones (pour avoir des traits plus précis)
x=np.linspace(-1,1,resol)
y=np.linspace(-1,1,resol)
X,Y=np.meshgrid(x,y)
Z=np.zeros((resol,resol)) # Matrice permettant de tracer les limites des zones
for i in range(resol):
    for j in range(resol):
        point=np.array([x[i],y[j]]).reshape(-1,1)
        Z[i,j]=bbl.f(point,W)
ax.contour(X,Y,Z)
plt.show()


#%% 
"""
Version Réseau de Neurones
"""
#%% Calcul de la matrice d'apprentissage
W=bbl.neuroneformel(train_data, train_labels, nbcat, bbl.g, bbl.gprim,10**(-2), 10000, True)    
    
#%% Affichage de l'apprentissage
resol=100
fig, ax = plt.subplots()
ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,cmap=plt.cm.Accent,marker='+')
ax.set_title("Résultat de l'apprentissage avec réseaux de Neurones")
xcoord=np.linspace(-1,1,resol)
ycoord=np.linspace(-1,1,resol)
X,Y=np.meshgrid(xcoord,ycoord)
Z=np.zeros((resol,resol))
for i in range(resol) :
    for j in range(resol) : 
        point=np.array([xcoord[i], ycoord[j]]).reshape(-1,1)
        Z[i,j]=bbl.evaluation(bbl.g,W,point) 
ax.contour(X,Y,Z)


#%% Matrice de Confusion pour données 2D
m, d = np.shape(train_data)
fx=np.zeros((m,1))
for i in range (m):
    x=train_data[i,:].reshape(-1,1)
    fx[i]=bbl.f(x,W)
bbl.matrice_conf(train_labels, fx ,nbcat)
mc = bbl.matrice_conf(train_labels, fx ,nbcat)


#%%
"""
Bases de données MNSIT et Fashion MNIST
"""
#%% 
"""
Importation des données MNIST 
"""
#%% Données Train
datatrain=np.loadtxt('mnist_train.csv', delimiter=',').astype('float') 
train_data=datatrain[:,1:].astype('int')
train_labels=datatrain[:,0].astype('int')

nbcat=bbl.nbcat(train_labels)
rho=1
epsilon=10**-2

#%% Données Test
datatest = np.loadtxt("mnist_test.csv", delimiter=",").astype('float')
test_data=datatest[:,1:].astype('int')
test_labels=datatest[:,0].astype('int')


#%% #%% Centrage des données
fac = 0.99 / 255
train_data = np.asfarray(train_data[:,1:])*fac + 0.01
test_data = np.asfarray(test_data[:,1:])*fac +0.01

#%%
"""
Importation des données Fashion MNIST
"""
#%% Données Train
datatrain=np.genfromtxt('fashion-mnist_train.csv', delimiter=',')[1:,:]
train_data=datatrain[:,1:].astype('int')
train_labels=datatrain[: , 0].astype('int')
nbcat=bbl.nbcat(train_labels)

nbcat=bbl.nbcat(train_labels)
rho=1
epsilon=10**-2
#%% Données Test
datatest=np.genfromtxt('fashion-mnist_test.csv', delimiter=',')[1:,:]
test_data=datatest[:,1:].astype('int')
test_labels=datatest[: , 0].astype('int')

#%% Centrage des données
fac = 0.99 / 255
train_data = np.asfarray(train_data[:,1:])*fac + 0.01
test_data = np.asfarray(test_data[:,1:])*fac +0.01

#%% Matrice de Confusion pour données mnist et fashion_mnist
m, d = np.shape(test_data)
fx=np.zeros((m,1))
for i in range (m):
    x=test_data[i,:].reshape(-1,1)
    fx[i]=bbl.f(x,W) 
bbl.matrice_conf(test_labels, fx ,nbcat)
mc=bbl.matrice_conf(test_labels, fx ,nbcat)

#%%
"""
1.2 Version non-linéaire avec l'astuce des noyaux 
"""
#%% 
"""
Version app global MCK
"""     
#%%
kern=bbl.kerng #choisir ici le kernel désiré selon les kernels disponibles dans la bibliothèque, en modifiant les derniers caractère de l'appel de bbl.kern
Pk , L2=bbl.liste_points(train_data, train_labels, 20)
#%%
W=bbl.app_global_MCK(train_data,train_labels,nbcat,rho,epsilon,kern,Pk)

#%% 
"""Réseaux de neurones avec kernels vu en TD"""
#%%

L1,L2=bbl.liste_points(train_data, train_labels, 40)
g=bbl.gk
gprim=bbl.gprimk 

#%% Calcul de la matrice d'aprentissage 

W=bbl.neuroneformelK(train_data, train_labels, 5, g, gprim,0.001, 10000,kern,L1)

#%% affichage 

fig, ax = plt.subplots()

resol=50

ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,cmap=plt.cm.Accent,marker='*')
ax.set_title("Données")

xcoord=np.linspace(-1,1,resol)
ycoord=np.linspace(-1,1,resol)

X,Y=np.meshgrid(xcoord,ycoord)

Z=np.zeros((resol,resol))
for i in range(resol) :
    for j in range(resol) : 
        point=np.array([xcoord[i], ycoord[j]]).reshape(-1,1)
        Z[i,j]=bbl.evaluationK(g,W,point,kern,L1) 

ax.contour(X,Y,Z)
    
#%%
"""
-------------------2 Un algorithme « alternatif » pour les SVM--------------------
"""
#%%
iris=datasets.load_iris()
X,y=iris.data[:,:2],iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
u_train=X_train[y_train==0,:]
v_train=X_train[y_train==1,:]
u_test=X_test[y_test==0,:]
v_test=X_test[y_test==1,:]

#%%
u=u_train
v=v_train
X=np.concatenate((-u,v),axis=0)
p,q=np.shape(u)[0],np.shape(v)[0]
xsol,lambdsol,compteur=bbl.Admm(X,p,q,u,v,rho=1,epsi=0.1,n=2,vue=1)
Ensemble=np.vstack([u,v])
#%%
taille_image = 28
image_pixels = taille_image * taille_image
#%%
"""
Importer les données mnsi ou fashion mnsit avec le code pkus haut (ligne 120 à 140 ou 148 à 165)
Ne pas appliquer le centrage des données, ne le faire qu'avec le bloc suivant
"""
#%% Centrage des données
fac = 0.99 / 255
train_imgs = np.asfarray(train_data)*fac + 0.01
test_imgs = np.asfarray(test_data)*fac +0.01
# %%
#On itére ce code 9 fois pour enregistrer les wi et bi i€[0,1...9]
for g in range(10):
    
    #Création des ensembles u et v contenant toutes les lignes du nombredetection
    nombredetection =g
    valeurs = train_data [: ,0]
    indiceu =np.where(valeurs==nombredetection )
    indicev =np.where(valeurs!=nombredetection )
    u=train_imgs [indiceu]
    v=train_imgs [indicev]   
    
    n=784
    X=np.concatenate((-u,v),axis=0)
    p,q=np.shape(u)[0],np.shape(v)[0]
    
    #On utilise notre fonction Admm avec un rho et epsi pas trop grand (sinon c'est plus long)
    xsol,lambdsol,compteur=bbl.Admm(X,p,q,u,v,rho=1,epsi=10**(-1),n=784,vue=0)
    
    w=xsol[:784]
    
    img_ordi =w.reshape ((28 ,28) )
    img_ordi = cv2 . normalize ( img_ordi , dst=None , alpha=0 , beta =255 , norm_type = cv2 . NORM_MINMAX , dtype=cv2. CV_8U )
    cv2 . imwrite ("{}_du_SVM.png".format(g), img_ordi)
       
    mu1 =w[:p]
    mu2 =w[p:]
    result1=np.nonzero(mu1)[0]
    result2=np.nonzero(mu2)[0]
    print("{}/9".format(g))
    
    #On calcul une solution « stable » numériquement de b
    b=0
    if result1 .size >0 :
        for i in np. nditer ( result1 ) :
            b =(1/(2* len ( result1 ) ) ) *u[i ,:] @w+b
    if result2 .size >0 :
        for j in np. nditer ( result2 ) :
            b =(1/(2* len ( result2 ) ) ) *v[j ,:] @w+b
            
    ww="w"+str(g)
    bb="b"+str(g)
    np.save(ww,w)
    np.save(bb,b)

#%%
w0,b0=np.load("w0.npy"),np.load("b0.npy")
w1,b1=np.load("w1.npy"),np.load("b1.npy")
w2,b2=np.load("w2.npy"),np.load("b2.npy")
w3,b3=np.load("w3.npy"),np.load("b3.npy")
w4,b4=np.load("w4.npy"),np.load("b4.npy")
w5,b5=np.load("w5.npy"),np.load("b5.npy")
w6,b6=np.load("w6.npy"),np.load("b6.npy")
w7,b7=np.load("w7.npy"),np.load("b7.npy")
w8,b8=np.load("w8.npy"),np.load("b8.npy")
w9,b9=np.load("w9.npy"),np.load("b9.npy")

#Création des matrices w et b pour simplifier les calculs
w=np.hstack([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9]) 
b=np.hstack([b0,b1,b2,b3,b4,b5,b6,b7,b8,b9])

#%%
#Création des matrices w et b pour simplifier les calculs
w=np.hstack([w0,w1,w2,w3,w4,w5,w6,w7,w8,w9]) 
b=np.hstack([b0,b1,b2,b3,b4,b5,b6,b7,b8,b9])

#%% Matrice de confusion
bbl.matrice_conf2(test_imgs,test_data[:,0],1,w,b) 

#%%
x = np.linspace(0, 20, 100)
v=np.vstack([x,bbl.v(x)]).T
u=np.vstack([x,bbl.u(x)]).T

X=np.concatenate((-u,v),axis=0)
p,q=np.shape(u)[0],np.shape(v)[0]
E=np.block([[np.ones((p,1))],[-np.ones((q,1))]])
xsol,lambdsol,compteur=bbl.Admm(X,p,q,u,v,rho=0.1,epsi=0.01,n=2,vue=1)
w=xsol[1:]
