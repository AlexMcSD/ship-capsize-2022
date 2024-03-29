import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
from System import MakeSystem


def FindStableTrajectory(f,lminus,lplus,tmin,tmax,N,p,J,Xhyp,A):  # p  is initial displacement along the stable direction [1,lminus]
    t = np.linspace(tmin,tmax,N)
    dt = (tmax-tmin)/N
    Deltatminus = int(np.log(10)/((1+lminus**2)*dt))
    Deltatplus = int((np.log(30)/(dt*(1+lplus**2))))
    imax, imin = min(J + Deltatplus, N-1), max(J - Deltatminus,0) 
#    imax, imin = N-1,max(J - Deltatminus,0) 
    newtmin = t[imin]
    P = np.array([[1,1],[lminus,lplus]])
    Pinv = np.linalg.inv(P)
    newtmax = t[imax]
    N = imax - imin
    XhypJ = Xhyp[J]
    XhypEnd = Xhyp[imax-1]
    X0 = np.zeros((N,2))
    for i in range(0,N):
#        X0[i] = Xhyp[i+imin]+ array([p*exp(-(1+lminus**2)*(t[i+imin] - t[J])),0])  
        X0[i] =p*exp(-(1+lminus**2)*(t[i+imin] - t[J]))*np.array([1,lminus])
    X0 = X0.reshape(2*N) #  len(t) by 2 array
    J = J -imin
    X = X0
    # evolution over one time step dt
    def phi(y0,t0):
        res = solve_ivp (f, [t0,dt+t0], y0 , method = "RK45" , max_step =dt/2 , rtol = 1e-10 )
        #y1 = y0 + dt*f(t0,y0) # Euler method for initial testing
        y = res .y. transpose()
        return  y[-1]
    # fixed point function
    def F(X,N):
        # N number of points taken
        Y = np.zeros((N,2))
        X = X.reshape((N,2))
        for i in range(0, N-1):
#            Y[i] = phi(X[i],t[i]) - X[i+1]
             Y[i] = phi(X[i]+Xhyp[i+imin],t[i+imin]) - X[i+1] - Xhyp[i+1+imin]
        Y[-1][0] =  Pinv.dot(X[J])[0]-p
        Y[-1][1] =  Pinv.dot(X[-1])[1]
        return Y.reshape(2*N)
    # Derivative for linear system
    DF0 = np.zeros((2*N,2*N))
    Dphi = scipy.linalg.expm(dt*A)
    for i in range(0,N-1):
        for j in range(0,N-1):
            DF0[2*i:2*i+2,2*i:2*i+2] = Dphi
            DF0[2*i:2*i+2,2*i+2:2*i+4] = -np.eye(2)
    DF0[2*N-2,2*J:2*J+2] = np.dot(Pinv.transpose(), np.array([1,0]))
    DF0[2*N-1,2*N-2:2*N] = np.dot(Pinv.transpose(), np.array([0,1]))
    P, L, U = scipy.linalg.lu(DF0) # numerical LU decomposition. Save this and integrate into code.
    def Usolve(y0):
        x = 0*y0
        M = len(y0)
        for i in range(0,M):
            sum = 0
            for j in range(0,i):
                sum +=U[M-1-i,M-1-j]*x[M-1-j]
            x[M-1-i] = (y0[M-1-i] - sum)/U[M-1-i,M-1-i]
        return x
    def Lsolve(y0):
        x = 0*y0
        M = len(y0)
        for i in range(0,M):
            sum = 0
            for j in range(0,i):
                sum +=L[i,j]*x[j]
            x[i] = (y0[i] - sum)/L[i,i]
        return x

    def DF0solve(y0):
        x0 = np.transpose(P).dot(y0)
        return Usolve(Lsolve(x0))
    
    i = 0
    Fx = F(X,N)
    normFx = np.linalg.norm(Fx)
    epsF = 1e-6
    epsC = 1e-7
    K = 200
    delta = DF0solve(Fx)
    while  (normFx > epsF or np.linalg.norm(delta) > epsC) and i < K:
        delta = DF0solve(Fx)
        X = X-delta # update sequence
        i += 1 # add counter
        Fx = F(X,N) # Evaluate at new point
        normFx = np.linalg.norm(Fx) # compute norm to check the estimate
    X = X.reshape((N,2)) + Xhyp[imin:imax]
    X0 = X0.reshape((N,2))
 #   print(" after " + str(i) + " iterations an estimate with error of F =" + str(normFx) + " and delta = "+ str(np.linalg.norm(delta))+ "was produced")
    tnew = t[imin:imax]
    return X,tnew, X0,J


def FindUnstableTrajectory(f,lminus,lplus,tmin,tmax,N,q,J,Xhyp,A):
    t = np.linspace(tmin,tmax,N)
    dt = (tmax-tmin)/N
    Deltatminus = int(np.log(10)/((1+lminus**2)*dt))
    Deltatplus = int((np.log(30)/(dt*(1+lplus**2))))
    imax, imin = min(J + Deltatplus, N-1), max(J - Deltatminus,0) 
#    imax, imin = N-1,max(J - Deltatminus,0) 
    newtmin = t[imin]
    P = np.array([[1,1],[lminus,lplus]])
    Pinv = np.linalg.inv(P)
    newtmax = t[imax]
    N = imax - imin
    XhypJ = Xhyp[J]
    XhypEnd = Xhyp[imax-1]
    X0 = np.zeros((N,2))
    for i in range(0,N):
#        X0[i] = Xhyp[i+imin]+ array([p*exp(-(1+lminus**2)*(t[i+imin] - t[J])),0])  
        X0[i] =q*exp(-(1+lminus**2)*(t[i+imin] - t[J]))*np.array([1,lplus])
    X0 = X0.reshape(2*N) #  len(t) by 2 array
    J = J -imin
    X = X0
    # evolution over one time step dt
    def phi(y0,t0):
        res = solve_ivp (f, [t0,dt+t0], y0 , method = "RK45" , max_step =dt/2 , rtol = 1e-10 )
        #y1 = y0 + dt*f(t0,y0) # Euler method for initial testing
        y = res .y. transpose()
        return  y[-1]
    # fixed point function
    def F(X,N):
        # N number of points taken
        Y = np.zeros((N,2))
        X = X.reshape((N,2))
        for i in range(0, N-1):
#            Y[i] = phi(X[i],t[i]) - X[i+1]
             Y[i] = phi(X[i]+Xhyp[i+imin],t[i+imin]) - X[i+1] - Xhyp[i+1+imin]
        Y[-1][0] =  Pinv.dot(X[J])[1]-q  # Boundary condition
        Y[-1][1] =  Pinv.dot(X[0])[0]
        return Y.reshape(2*N)
    # Derivative for linear system
    DF0 = np.zeros((2*N,2*N))
    Dphi = scipy.linalg.expm(dt*A)
    for i in range(0,N-1):
        for j in range(0,N-1):
            DF0[2*i:2*i+2,2*i:2*i+2] = Dphi
            DF0[2*i:2*i+2,2*i+2:2*i+4] = -np.eye(2)
    DF0[2*N-2,2*J:2*J+2] = np.dot(Pinv.transpose(), np.array([0,1]))
    DF0[2*N-1,0:2] = np.dot(Pinv.transpose(), np.array([1,0]))
    P, L, U = scipy.linalg.lu(DF0) # numerical LU decomposition. Save this and integrate into code.
    def Usolve(y0):
        x = 0*y0
        M = len(y0)
        for i in range(0,M):
            sum = 0
            for j in range(0,i):
                sum +=U[M-1-i,M-1-j]*x[M-1-j]
            x[M-1-i] = (y0[M-1-i] - sum)/U[M-1-i,M-1-i]
        return x
    def Lsolve(y0):
        x = 0*y0
        M = len(y0)
        for i in range(0,M):
            sum = 0
            for j in range(0,i):
                sum +=L[i,j]*x[j]
            x[i] = (y0[i] - sum)/L[i,i]
        return x

    def DF0solve(y0):
        x0 = np.transpose(P).dot(y0)
        return Usolve(Lsolve(x0))
    
    i = 0
    Fx = F(X,N)
    normFx = np.linalg.norm(Fx)
    epsF = 1e-6
    epsC = 1e-7
    K = 200
    delta = DF0solve(Fx)
    while  (normFx > epsF or np.linalg.norm(delta) > epsC) and i < K:
        delta = DF0solve(Fx)
        X = X-delta # update sequence
        i += 1 # add counter
        Fx = F(X,N) # Evaluate at new point
        normFx = np.linalg.norm(Fx) # compute norm to check the estimate
    X = X.reshape((N,2)) + Xhyp[imin:imax]
    X0 = X0.reshape((N,2))
 #   print(" after " + str(i) + " iterations an estimate with error of F =" + str(normFx) + " and delta = "+ str(np.linalg.norm(delta))+ "was produced")
    tnew = t[imin:imax]
    return X,tnew, X0,J

