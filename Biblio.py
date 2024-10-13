# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:42:07 2022

@author: colin
"""

#%%
"""
                BE : Optimisation différentiable 2
"""
#%%
import numpy as np
import copy
import matplotlib.pyplot as plt

#%% 
"""
-------------------1 Les moindres carrées multi-classes avec régularisation-------------------
"""    
#%%
"""
1.1 Version linéaire
"""
#%% ACP

def centre_red(E) :
    """
    Parameters
    ----------
    E : array([])
        Matrice d'entée de taille (l,c)
        
    Returns
    -------
    Matrice E centrée réduite
    """
    l,c=np.shape(E)
    ind_G=np.mean(E, axis=0)
    # Matrice centrée seulement :
    Mcentrered= E-np.kron(ind_G,np.ones((l,1)))
    # Lorsqu'une ligne de la matrice centrée est composée seulement de 0, il n'y a pas besoin de réduire. De plus on évite de diviser par 0.
    # On réduit donc seulement losque la ligne est différente de 0
    for i in range(c):
        if np.linalg.norm(Mcentrered[:,i])!=0 :
            Mcentrered[:,i]=(1/np.linalg.norm(Mcentrered[:,i]))*Mcentrered[:,i]
    return Mcentrered

def ACP(X,q):
    """
    Parameters
    ----------
    X : array([]).
        Matrice X de taille (m,n).
        
    q : dimensiosn de l'ACP
    
    Returns
    -------
    Xq : array([]).
        Matrice Xq de taille (m,q), ACP de X.
    """ 
    Xrc=centre_red(X)
    C=Xrc.T@Xrc
    # On peut utiliser la méthode svd sur la matrice de covariance, après que X 
    # ait été centrée et réduite.
    U,S,Vt=np.linalg.svd(C)
    Uq=U[:,:q]
    Xq=Xrc@Uq
    return Xq

#%% Gradient Conjugué Projeté version matricielle

def tr(A,B):
    """
    Permet de remplacer les produits scalaires par des traces
    """
    tr=np.trace(A@B)
    return tr

def GPCmat(A,b,x0,epsilon) :
    """
    Méthode du gradient projeté version matricielle. Les produits scalaires sont remplacés par des traces
    """
    x = copy.copy(x0)
    d = b-tr(A.T,x)
    compteur = 0
    y = x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) > epsilon and compteur < 1000 : 
         y = copy.copy(x)
         t = -( tr(d.T , (A@x-b)) )/tr(d.T , A@d)
         x = x+t*d
         beta = tr(d.T , A@(A@x-b)) / tr(d.T , A@d)
         d = -(A@x-b)+beta*d
         compteur+=1
    return x , compteur

#%% Fonction permettant de récupérer le nombre de catégorie d'une base de donnée
 
def nbcat(train_labels):
    """
    Parameters
    ----------
    train_labels : Array
        Labels d'une base de donnée

    Returns
    -------
    cpt : Int
        Renvoie le nombre de catégorie d'une base de donnée
    """
    cpt = 0
    L=[]
    for i in range (len(train_labels)) :
       if train_labels[i] not in L :
           L.append(train_labels[i])
           cpt+=1
    return cpt

#%% Version linéaire 

def app_global_MC(train_data,train_labels,nbcat,rho,epsilon):
    """
    Parameters
    ----------
    train_data : Array
        données d’entraînements
    train_labels : Array
        labels associés aux données d’entraînements
    nbcat : Int
        le nombre de catégorie d’apprentissage
    rho : Float
        le facteur de régularisation
    epsilon : Float 
        le facteur d’erreur de fin de boucle du gradient conjugué
   
    Returns
    -------
    W : Array (d+1, nbcat)
        Renvoie la matrice d'apprentissage W. De taille (d+1,nbcat) avec d le nombre de coloenne de la base de donée
    """
    m, d = np.shape(train_data)
    D=np.hstack((train_data,np.ones((m,1))))
    B=np.zeros((m,nbcat))
    for j in range (nbcat):
        for i in range(m):
            if train_labels[i]==j:
                B[i,j]=1
    I = np.eye(d+1,d+1)
    A = D.T@D+rho*I
    C=D.T@B
    w0=np.ones((d+1,nbcat))
    W ,cpt = GPCmat(A,C,w0,epsilon) 
    print("GPCmat : La convergence à {} près est obtenue pour {} itérations.".format(epsilon,cpt))
    return W

def f(x,W):
    """
    Renvoie la valeur reconnu par la matrice d'apprenstisage w pour le vecteur de donnée x
    """
    y=np.vstack((x,1))
    val= y.T @ W
    return np.argmax(val)

#%% Réseaux de neurones (Version TD)

def g(x) :
    return np.tanh(x)

def gprim(x) : 
    return 1-g(x)**2 

def neuroneformel(train_data, labels, cat, g, gprim,rho, itermax,error=False) : 
    """
    Parameters
    ----------
    train_data : Array
        données d’entraînements
    labels : Array
        labels associés aux données d’entraînements
    cat : Int
        le nombre de catégorie d’apprentissage
    g : Fonction
    gprim : Fonction
        dérivé de g
    rho : Int ou Float
        le facteur de régularisation
    itermax : Int
        le nombre d'itération maximale
    error : Bool
        option pour afficher ou non l'erreur sur chaque itération. Par défaut n'affiche pas
  
    Returns
    -------
    W : Array (d+1, nbcat)
        Renvoie la matrice d'apprentissage W. De taille (d+1,nbcat) avec d le nombre de coloenne de la base de donée
    """
    l,c=np.shape(train_data)
    W=np.random.rand(c+1,cat)
   
    W0=np.zeros((c+1,cat))
    cpt=0
    index=0
    errmin=100000

    while np.linalg.norm(W-W0) >10**(-6) and cpt<itermax : 
        cpt+=1
        indice=np.random.randint(0,l) 
        x=train_data[indice,:].reshape(-1,1)
        xtilde=np.vstack((1, x))
        u=-np.ones((cat,1))
        u[int(labels[indice])]=1
        sigma=W.T@xtilde 
        deltaE=np.kron(xtilde, (gprim(sigma)*(g(sigma)-u)).T)
        W0=copy.copy(W) 
        W=W-rho*deltaE 
        #L'appel de l'erreur à chacune des itérations ralentie énromément l'exécution, on rajoute pour cela l'option d'afficher ou non cette erreur
        if error==True:
            err=erreur(train_data, labels, cat, g, W)
            if err<errmin:
                errmin=err
                index=cpt
            print("Itération : {} et erreur {}".format(cpt,err))
        if error == False:
            print("Itération : {}".format(cpt))
                
    if error == True :
        print("L'erreur minimale est de {}, atteinte au plus tard à l'itération : {}".format(errmin,index))
    return W

def evaluation(g,W,x) :
    """
    Renvoie la valeur reconnu par la matrice d'apprenstisage w pour le vecteur de donnée x
    """
    xtilde=np.vstack((1, x))
    sigma=W.T@xtilde 
    val=np.where(g(sigma)==np.max(g(sigma)))[0][0]
    return val

def erreur(train_data, labels, cat, g, W) :
    l,c= l,c=np.shape(train_data)
    E=0
    for i in range(l) : 
        x=train_data[i,:].reshape(-1,1)
        val=evaluation(g,W,x)
        v=-np.ones((cat,1))
        v[val]=1
        
        u=-np.ones((cat,1))
        u[int(labels[i])]=1
      
        E+=np.linalg.norm(v-u)/np.sqrt(8) 
    return E    
     
#%% Matrice de confusion

def matrice_conf(labels, sortie ,nbcat):
    """
    Parameters
    ----------
    labels : Array
        Labels de la base de donnée test
    sortie : Array
        Matrice de la même taille que labels, conteannt les valeurs retournées par la fonction d'évaluation
    nbcat :  Int
        le nombre de catégorie d’apprentissage
        
    Returns
    -------
    conf : Array
        renvoie la matrice de confusion 
    """
    conf= np.zeros((nbcat,nbcat))
    for i in range(len(labels)):
            a=labels[i].astype("int")
            b=sortie[i].astype("int")
            conf[a,b]+=1
    Nvp=np.trace(conf)
    N=len(labels)
    txreuexact=(Nvp)/N
    txreupct=round(txreuexact*100,2)
    print('Matrice de confusion : ' +'\n', conf)
    print('Taux de réussite',txreupct,'%')
    return conf

#%% Version non-linéaire avec noyaux

def app_global_MCK(train_data,train_labels,nbcat,rho,epsilon,kern,Pk):
    """
    Parameters
    ----------
    train_data : Array
        données d’entraînements
    train_labels : Array
        labels associés aux données d’entraînements
    nbcat : Int
        le nombre de catégorie d’apprentissage
    rho : Float
        le facteur de régularisation
    epsilon : Float 
        le facteur d’erreur de fin de boucle du gradient conjugué
    kern : TYPE
        DESCRIPTION.
    Pk : List of array
        ligne alétoires de train_data. On la génère avec la fonction liste_point

    Returns
    -------
    W : Array (d+1, nbcat)
        Renvoie la matrice d'apprentissage W. De taille (d+1,nbcat) avec d le nombre de coloenne de la base de donée
    """
    m, d = np.shape(train_data)
    p=len(Pk)
    Dk=np.zeros((m,p))
    B=np.zeros((m,nbcat))
    K = np.zeros((p,p))
    Wko=np.zeros((p,nbcat))
    # Création de la matrice B
    for j in range (nbcat): 
        for i in range(m):
            if train_labels[i]==j:
                B[i,j]=1
    # Création de la matrice Dk 
    for i in range(m):
        for j in range(p):
            x1=train_data[i].reshape(-1,1)
            x1=np.vstack((x1,1))
            x2=Pk[j].reshape(-1,1)
            x2=np.vstack((x2,1))
            Dk[i,j]=kern(x1,x2)
    # Création de la matrice K
    for i in range(p):
        for j in range(p):
            K[i,j]=kern(Pk[i].reshape(-1,1),Pk[j].reshape(-1,1))
    
    # On applique le gradient conjugué comme dans la fonction linéaire,
    # sauf que A et C sont différentes
    A= Dk.T@Dk+rho*K
    C=Dk.T@B
    W ,cpt = GPCmat(A,C,Wko,epsilon)
    return W

#%% Résaux de neurones avec kernels

#%% Liste fonctions noyaux
#%% noyau gaussien

c=0.5
def kerng(x,y):
    return np.exp(-(1/(2*c**2))*np.linalg.norm(x-y)**2)

#%% noyau poly

c=1
# # #c=105 et d=2
def kernp(x,y):
      return 10**(-20)*(c+x.T @ y)**(9)
  
#%% noyau sigmoïde

c=1
gamma=10**(-20)
# # #c=105 et d=2
def kernsig(x,y):
      return np.tanh(c+gamma*x.T @ y)

#%% noyau "rationnal quadratic kernel"

c=10
def kernq(x,y):
      return 1-np.linalg.norm(x-y)**2/(np.linalg.norm(x-y)**2+c)
 
#%% noyau multiquadratique

c=10
def kernmq(x,y):
      return np.sqrt(np.linalg.norm(x-y)**2+c**2)
    
#%% noyau inverse multiquadratique

c=1
def kerninvmq(x,y):
      return 1/np.sqrt(np.linalg.norm(x-y)**2+c**2)
  
#%% chi noyau

def kernchi(x,y):
    S=0
    for i in range(0,len(x)): 
        S=S+2*x[i]*y[i]/(x[i]+y[i])
    return S

#%% chi histogram intersection kernel

def kernchihint(x,y):
    S=0
    for i in range(0,len(x)): 
        S=S+min(x[i],y[i])
    return S

#%% noyau de Cauchy

c=5
def kernc(x,y):
      return 1/(1+(np.linalg.norm(x-y)**2)/c**2)

#%% log kernel (pour dimension paire)

c=2
def kernl(x,y):
      return -np.log(np.linalg.norm(x-y)**2+1)

#%% spline kernel

def kernspl(x,y):
    S=1
    for i in range(0,len(x)): 
        S=S*(1*x[i]*y[i]+x[i]*y[i]*min(x[i],y[i])-((x[i]+y[i])/2)*min(x[i],y[i])**2+(1/3)*min(x[i],y[i])**3)
    return S

#%% liste de points 

def gk(x) : 
    c=1
    return np.log(1+np.exp(c*x))

def gprimk(x) :
    c=1
    return c/(1+np.exp(-c*x))


def liste_points(train_data, labels, nb) :
    """
    Parameters
    ----------
    train_data : Array
        données d’entraînements
    labels : Array
        labels associés aux données d’entraînements
    nb : Int
        Nombre de vecteurs à choisir aléatoirement

    Returns
    -------
    L1 : Array
        Liste de nb vecteurs aléatoirs de train_data
    L2 : Array
        Listes des labels associées aux vecteurs choisies aléatoirement

    """
    var=np.random.randint(0,len(train_data),nb)
    L1=list(train_data[var])
    L2=list(labels[var].astype('int'))
    return L1,L2


def erreurK(train_data, labels, cat, g, W,kern,L1) :
    l,c= l,c=np.shape(train_data)
    E=0
    for i in range(l) : 
        x=train_data[i,:].reshape(-1,1)
        val=evaluationK(g,W,x,kern,L1)
        v=-np.ones((cat,1))
        v[val]=1
        
        u=-np.ones((cat,1))
        u[int(labels[i])]=1
      
        E+=np.linalg.norm(v-u)/np.sqrt(8) 
    return E    

def neuroneformelK(train_data, labels, cat, g, gprim,rho, itermax,kern,L1,error=False) : 
    """
    Parameters
    ----------
    train_data : Array
        données d’entraînements
    labels : Array
        labels associés aux données d’entraînements
    cat : Int
        le nombre de catégorie d’apprentissage
    g : Fonction
    gprim : Fonction
        dérivé de g
    rho : Int ou Float
        le facteur de régularisation
    itermax : Int
        le nombre d'itération maximale
    kern : TYPE
        DESCRIPTION.
    L1 : TYPE
        DESCRIPTION.
    error : Bool
        option pour afficher ou non l'erreur sur chaque itération. Par défaut n'affiche pas

    Returns
    -------
    W : Array (d+1, nbcat)
        Renvoie la matrice d'apprentissage W. De taille (d+1,nbcat) avec d le nombre de coloenne de la base de donée
    """
    l,c=np.shape(train_data)
    P=len(L1)
    W=np.random.rand(P,cat)
   
    W0=np.zeros((P,cat))
    cpt=0
    errmin=100000
    
    while np.linalg.norm(W-W0) >10**(-10) and cpt<itermax : 
        cpt+=1
        indice=np.random.randint(0,l) 
        x=train_data[indice,:].reshape(-1,1)
        xtilde=np.vstack((1, x))
        u=-np.ones((cat,1))
        u[int(labels[indice])]=1
        
        Ktilde=np.zeros((P,1))
        for i in range(P) :  
            xi=L1[i].reshape(-1,1)
            xitilde=np.vstack((1, xi))
            Ktilde[i,0]=kern(xitilde,xtilde)
            
        sigma=W.T@Ktilde 
        #print(g(sigma))
        deltaE=np.kron(Ktilde, (gprim(sigma)*(g(sigma)-u)).T)
        W0=copy.copy(W) 
        W=W-rho*deltaE 
        
        if error==True:
            err=erreurK(train_data, labels, cat, g, W,kern,L1)
            if err<errmin:
                errmin=err
                index=cpt
            print("Itération : {} et erreur {}".format(cpt,err))
            if err<=10 : 
                break
        if error == False:
            print("Itération : {}".format(cpt))
            
    if error == True :
        print("L'erreur minimale est de {}, atteinte au plus tard à l'itération : {}".format(errmin,index))
    return W    

   
#%%  

def evaluationK(g,W,x,kern,L1) :
    xtilde=np.vstack((1, x))
    P=len(L1)
    Ktilde=np.zeros((P,1))
    for i in range(P) :  
            xi=L1[i].reshape(-1,1)
            xitilde=np.vstack((1, xi))
            Ktilde[i,0]=kern(xitilde,xtilde)
    
    sigma=W.T@Ktilde 
    val=np.where(g(sigma)==np.max(g(sigma)))[0][0]
    
    return val


#%%
"""
-------------------2 Un algorithme « alternatif » pour les SVM--------------------
"""
#%%
def x0_update(A,rho,C,d,z0,lambd0):
    return -np.linalg.inv(A+rho*C.T@C)@(C.T@(rho*(z0+d)+lambd0))

def z0_update(C,rho,d,lambd0,x1):
    return -(lambd0/rho)-C@x1-d

def lambd0_update(A,rho,C,d,lambd0,x1,z1):
    return lambd0-rho*(C@x1+z1+d)

def Admm(X,p,q,u,v,rho,epsi,n=2,vue=0):
    #Initialisation rho<epsi
    A=np.hstack([np.eye(n+1,n),np.zeros((n+1,1))])
    d=np.ones((p+q,1))
    E=np.block([[np.ones((p,1))],[-np.ones((q,1))]])
    C=np.block([X,E])
    x0=np.ones((n+1,1))
    z0=np.zeros((p+q,1))
    lambd0=np.zeros((p+q,1))
    
    #Boucle d'itération de l'ADMM
    compteur=0
    Norm=1
    while Norm>epsi and compteur<100:
        
        x1=x0_update(A,rho,C,d,z0,lambd0)
        z1=z0_update(C,rho,d,lambd0,x1)
        lambd1=lambd0_update(A,rho,C,d,lambd0,x1,z1) 
        
        Norm=np.linalg.norm(x1-x0,2)
        compteur+=1
        
        x0=copy.deepcopy(x1)
        z0=copy.deepcopy(z1)
        lambd0=copy.deepcopy(lambd1) 
        
        #Tracé
        if vue==1:
            nbt=100
            w=x1[:2].T[0]
            Ensemble=np.vstack([u,v])
            t=np.linspace(np.min(Ensemble[:,0]),np.max(Ensemble[:,0]),nbt)
            b=(np.min(w@u.T)+np.max(w@v.T))/2
            delta=(np.min(w@ u.T)-np.max(w@v.T))/2
            affichage(Norm,epsi,rho,u,v,delta,b,w,t,nbt,compteur)
            
    print("nombre d'itérations à {}: ".format(epsi),compteur) 
    return x1,lambd1,compteur

# %%
def affichage(Norm,epsi,rho,u,v,delta,b,w,t,nbt,compteur):
    if  Norm>epsi:
        plt.figure(1)
        plt.title("ADMM avec itérations")
        plt.xlabel("{} itérations avec rho={}  epsilon={}".format(compteur+1,rho,epsi))
        #Tracé de l'hyperplan séparateur pour chaque itération
        plt.plot(t,(b*np.ones(nbt)-w[0]*t)/w[1],'--r')    
        
    else :
        plt.figure(1)
        plt.scatter(u[:,0],u[:,1],marker=".",c="black",label="Ensemble 1")
        plt.scatter(v[:,0],v[:,1],marker=".",c="grey",label="Ensemble 2")
        #Tracé de l'hyperplan séparateur
        plt.plot(t,(b*np.ones(nbt)-w[0]*t)/w[1],'b-',label="Hyperplan séparateur") 
        plt.legend() 
        plt.show()
        plt.figure(2)
        plt.title("ADMM avec marge")
        plt.xlabel("{} itérations avec rho={}  epsilon={}".format(compteur,rho,epsi))
        plt.scatter(u[:,0],u[:,1],marker=".",c="black",label="Ensemble 1")
        plt.scatter(v[:,0],v[:,1],marker=".",c="grey",label="Ensemble 2")
        #Tracé de l'hyperplan séparateur
        plt.plot(t,(b*np.ones(nbt)-w[0]*t)/w[1],'b-')   
        #Avec décalage +/-
        plt.plot(t,((b+delta)*np.ones(nbt)-w[0]*t)/w[1],"m-")
        plt.plot(t,((b-delta)*np.ones(nbt)-w[0]*t)/w[1],"y-")   
        plt.legend() 
        plt.show()
    return
# %%
def matrice_conf2(test_imgs,test_labels,nombredetection,w,b) :
    Nvp=0
    Nfn=0
    Nfp=0
    Nvn=0
    N,M=test_imgs.shape
    #Cherchons nos Vrai et Faux 
    for k in range(0,N):  
      #val contient le coéfficient de correspondance au chiffre
      val=w.T@test_imgs[k,:]-b
      #on récupere ici l'index de la valeur max qui est supposé etre la solution
      index=list(val).index(max(val))
      if val[index]>0 and test_labels[k]==index :
         Nvp=Nvp+1
      elif val[index]<=0 and test_labels[k]==index :
         Nfn=Nfn+1
      elif val[index]>0 and test_labels[k]!=index :
         Nfp=Nfp+1
      elif val[index]<=0 and test_labels[k]!=index :   
         Nvn=Nvn+1 
             
    
    
    Mc=np.array([[Nvp, Nfn],[Nfp, Nvn] ]) 
            
    print("Matrice de confusion pour tout w (w0,W1...w9) :\n", Mc)
    
    txreussite=100*(Nvp+Nvn)/N
    sensibilite=Nvp/(Nvp+Nfn)
    print('Taux de réussite : {}%'.format(txreussite))
    print('Sensibilité :', sensibilite)         
    return 

# %%
def u(x):
    noise = np.random.normal(0, 1, x.shape)
    return (x+noise)
def v(x):
    noise = np.random.normal(0, 1, x.shape)
    return 4+(x-noise)