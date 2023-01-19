import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg

def MakeSystem(k,ForcingParameters):
    forcingComponents = int(len(ForcingParameters)/2)
    Amplitudes = ForcingParameters[2*np.arange(0,forcingComponents,1)]
    Frequencies = ForcingParameters[2*np.arange(0,forcingComponents,1)+1]
    G = lambda t : np.sum(Amplitudes*np.cos(Frequencies*t))
    f = lambda t,y : array([-y[0]*lminus**2 + y[1] -tanh(y[0]+y[1])*(cosh(y[0]+y[1])**(-2)) -G(t) , -y[0] + lplus**2 * y[1] +tanh(y[0]+y[1])*(cosh(y[0]+y[1])**(-2)) + G(t)])
    lminus = (-k - (k**2 + 4)**.5)/2
    lplus = (-k + (k**2 + 4)**.5)/2
    return f,lminus,lplus,G

def MakeLinearisedSolution(ForcingParameters,lplus,lminus): # ForcingParameters = (amplitude1, frequency1,....)
    forcingComponents = int(len(ForcingParameters)/2)
    muminus,muplus = 1 + lminus**2, -(1 + lplus**2)
    Amplitudes = 1.0*ForcingParameters[2*np.arange(0,forcingComponents,1)]
    Frequencies = 1.0*ForcingParameters[2*np.arange(0,forcingComponents,1)+1]
    #XQ = lambda t : array([ -A1*((1+ (muminus/w1)**2)**(-1)*(w1**(-1)*sin(w1*t) + (muminus/(w1**2))*cos(w1*t) )) -A2*((1+ (muminus/w2)**2)**(-1)*(w2**(-1)*sin(w2*t) + (muminus/(w2**2))*cos(w2*t) )),
    #                   A1*((1+ (muplus/w1)**2)**(-1)*(w1**(-1)*sin(w1*t) + (muplus/(w1**2))*cos(w1*t) ) )+A2*((1+ (muplus/w2)**2)**(-1)*(w2**(-1)*sin(w2*t) + (muplus/(w2**2))*cos(w2*t) ) )])
    XQ = lambda t : array([ np.sum(-Amplitudes*((1+ (muminus/Frequencies)**2)**(-1)*(Frequencies**(-1)*sin(Frequencies*t) + (muminus/(Frequencies**2))*cos(Frequencies*t)))) , 
                           np.sum(Amplitudes*((1+ (muplus/Frequencies)**2)**(-1)*(Frequencies**(-1)*sin(Frequencies*t) + (muplus/(Frequencies**2))*cos(Frequencies*t) ) ))])
    return XQ
    

def FindHyperbolicTrajectory(f,lminus,lplus,XQ,tmin,tmax,N):
    t = np.linspace(tmin,tmax,N)
    dt = (tmax-tmin)/N
    X0 = np.zeros((N,2))
    for i in range(0,N):
        X0[i] = XQ(t[i])
    X0 = X0.reshape(2*N) #  len(t) by 2 array
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
            Y[i] = phi(X[i],t[i]) - X[i+1]
        Y[-1][0] =  X[0][0]
        Y[-1][1] =  X[-1][1]
        return Y.reshape(2*N)
    # Derivative for linear system
    DF0 = np.zeros((2*N,2*N))
    for i in range(0,2*N-2):
        for j in range(0,2*N):
            if j == i:
                if i % 2 == 0:
                    DF0[i,j] = exp((-1-lminus**2)*dt)
                else:
                    DF0[i,j] = exp((+1+lplus**2)*dt)
            elif j == i + 2:
                DF0[i,j] = -1
    DF0[2*N-2,0] = 1
    DF0[2*N-1,2*N-1] = 1
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
    epsF = 1e-8
    epsC = 1e-8
    K = 200
    


    while  (normFx > epsF or np.linalg.norm(delta) > epsC) and i < K:
        delta = DF0solve(Fx)
        X = X-delta # update sequence
        i += 1 # add counter
        Fx = F(X,N) # Evaluate at new point
        normFx = np.linalg.norm(Fx) # compute norm to check the estimate
    X = X.reshape((N,2))
    X0 = X0.reshape((N,2))
    print(" after " + str(i) + " iterations an estimate with error of F =" + str(normFx) + " and delta = "+ str(np.linalg.norm(delta))+ "was produced")
    return X,t, X0
