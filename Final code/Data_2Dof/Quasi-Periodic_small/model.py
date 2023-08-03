import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp,array,sqrt
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import scipy.io

# here we store functions used in many files

def linearisedSystem(x0,h,k): # this is Df(x0)
    kx,ky = k,k
    A = np.zeros((4,4))
    A[0,2] = 1
    A[1,3] = 1
    A[2,0] = -h
    A[2,1] = 2*h*x0[1]
    A[3,0] = x0[1]
    A[3,1] = -1  +x0[0]
    A[2,2] = -kx
    A[3,3] = -ky
    return A

def model(h,k,D,G): ## this returns, f, Df(xplus),Df(xminus),xplus,xminus: where xplus/minus are saddle points
    kx,ky = k,k
    f = lambda t,y : array([y[2] ,y[3] ,-h*(y[0]-y[1]**2) - kx*y[2] ,-y[1] + y[0]*y[1] - D(y[3]) + G(t)]) 
    xplus = array([1,1,0,0])
    xminus = array([1,-1,0,0])
    Aplus = linearisedSystem(xplus,h,k)
    Aminus = linearisedSystem(xminus,h,k)
    return f,Aplus, Aminus, xplus, xminus

def eigen(h,k): #eigenvalues and eigen vectors and change of basis matrix
    alpha = 2*h+2*sqrt(8*h + h**2)
    beta = -2*h +2*sqrt(8*h + h**2)
    omega = sqrt(alpha - k**2)/2
    lambda2 = -k/2 
    lambda3 = (-k - sqrt(beta + k**2))/2
    lambda4 = (-k + sqrt(beta + k**2))/2
    unstableplus = array([k/2 + sqrt(k**2 + beta)/2,
                      -(-k*sqrt(h) - k*sqrt(8+h) -sqrt(h)*sqrt(k**2 + beta) - sqrt(8+h)*sqrt(k**2 +beta))/(8*sqrt(h)) ,
                      -h/2 + sqrt(8*h+ h**2)/2,1])
    stableplus = array([k/2 - sqrt(k**2 + beta)/2,-(-k*sqrt(h) - k*sqrt(8+h) +sqrt(h)*sqrt(k**2 + beta) + sqrt(8+h)*sqrt(k**2 +beta))/(8*sqrt(h)) ,-h/2 + sqrt(8*h+ h**2)/2,1])
    center1plus = array([k/2,-(-k*sqrt(h) + k*sqrt(8+h))/(8*sqrt(h)),-h/2 - sqrt(8*h + h**2)/2,1 ])
    center2plus = array([ -omega, -omega*(sqrt(h) - sqrt(8+h) )/(4*sqrt(h)),0,0])
    Pplus = np.vstack((center1plus,center2plus,stableplus,unstableplus)).transpose()
    Pplusinv = np.linalg.inv(Pplus)
    
    center1minus = array([-k/2,-(-k*sqrt(h) + k*sqrt(8+h))/(8*sqrt(h)),h/2 + sqrt(8*h + h**2)/2,1 ])
    center2minus = array([ omega, -omega*(sqrt(h) - sqrt(8+h) )/(4*sqrt(h)),0,0])
    stableminus = array([k/2 + sqrt(k**2 + beta)/2,-(-k*sqrt(h) - k*sqrt(8+h) +sqrt(h)*sqrt(k**2 + beta) + sqrt(8+h)*sqrt(k**2 +beta))/(8*sqrt(h)) ,h/2 - sqrt(8*h+ h**2)/2,1])
    unstableminus = array([-k/2 - sqrt(k**2 + beta)/2,
                      -(-k*sqrt(h) - k*sqrt(8+h) -sqrt(h)*sqrt(k**2 + beta) - sqrt(8+h)*sqrt(k**2 +beta))/(8*sqrt(h)) ,
                      h/2 - sqrt(8*h+ h**2)/2,1])
    Pminus = np.vstack((center1minus,center2minus,stableminus,unstableminus)).transpose()
    Pinvminus = np.linalg.inv(Pminus)
    return Pplus,Pplusinv,Pminus,Pinvminus,lambda2, omega, lambda3, lambda4


def Usolve(y0,Upper): # solves upper triangluar matrix equation Ax = y0
    x = 0*y0
    M = len(y0)
    for i in range(0,M):
        sum = 0
        for j in range(0,i):
            sum +=Upper[M-1-i,M-1-j]*x[M-1-j]
        x[M-1-i] = (y0[M-1-i] - sum)/Upper[M-1-i,M-1-i]
    return x
def Lsolve(y0,Lower):  # solves lower triangluar matrix equation Ax = y0
    x = 0*y0
    M = len(y0)
    for i in range(0,M):
        sum = 0
        for j in range(0,i):
            sum +=Lower[i,j]*x[j]
        x[i] = (y0[i] - sum)/Lower[i,i]
    return x
