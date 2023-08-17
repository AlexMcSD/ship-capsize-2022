# this stores the functions used to find the hyperbolic trajectories
import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp,array,sqrt
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import scipy.io
from model import linearisedSystem,model,eigen,Usolve,Lsolve

def linearHyperbolicCosine(Amp,w,Pplus,Pminus,Pplusinv,Pinvminus,lambda2, omega, lambda3, lambda4):   # Finds the analytical solution for the displacement of the hyperbolic trajectory from the saddle point
    # for the linearised system. Will be used as the intial guess.
    # This function provides this in the case G(t) = Amp*cos(w*t)
    #Pplus,Pminus is the change of basis matrix from the eigenbasis to the standard coordinates.
    A1plus = Amp*Pplusinv.dot(array([0,0,0,1]))[0]
    A2plus = Amp*Pplusinv.dot(array([0,0,0,1]))[1]
    A3plus = Amp*Pplusinv.dot(array([0,0,0,1]))[2]
    A4plus = Amp*Pplusinv.dot(array([0,0,0,1]))[3]
    #need to replace k here with a lambda2
    psi1 = lambda t: .25*((-2*lambda2)*cos((w+omega)*t)+2*(w+omega)*sin((w+omega)*t)  )/(lambda2**2 + (omega + w)**2)  +         .25*( -2*lambda2*cos((omega-w)*t) + 2*(omega-w)*sin((omega-w)*t)  )/(lambda2**2 +(omega-w)**2)
    psi2 = lambda t: .25*((-2*lambda2)*sin((w+omega)*t) - 2*(w+omega)*cos((w+omega)*t)  )/(lambda2**2 + (omega + w)**2)    +         .25*(-2*lambda2*sin((omega-w)*t) - 2*(omega-w)*cos((omega-w)*t)  )/(lambda2**2 + (omega - w)**2) 
    XQplus = lambda t : Pplus.dot(array([A1plus*cos(omega*t)*psi1(t) + A2plus*cos(omega*t)*psi2(t) + A1plus*sin(omega*t)*psi2(t) - A2plus*sin(omega*t)*psi1(t),
                                A1plus*sin(omega*t)*psi1(t) + A2plus*sin(omega*t)*psi2(t) - A1plus*cos(omega*t)*psi2(t) + A2plus*cos(omega*t)*psi1(t),
                                A3plus*(sin(w*t)/w - lambda3 *cos(w*t)/w**2)/(1+(lambda3/w)**2),
                                A4plus*(sin(w*t)/w - lambda4 *cos(w*t)/w**2)/(1+(lambda4/w)**2)]))
    
    A1minus = Amp*Pinvminus.dot(array([0,0,0,1]))[0]
    A2minus = Amp*Pinvminus.dot(array([0,0,0,1]))[1]
    A3minus = Amp*Pinvminus.dot(array([0,0,0,1]))[2]
    A4minus = Amp*Pinvminus.dot(array([0,0,0,1]))[3]
    XQminus = lambda t : Pminus.dot(array([A1minus*cos(omega*t)*psi1(t) + A2minus*cos(omega*t)*psi2(t) + A1minus*sin(omega*t)*psi2(t) - A2minus*sin(omega*t)*psi1(t),
                                A1minus*sin(omega*t)*psi1(t) + A2minus*sin(omega*t)*psi2(t) - A1minus*cos(omega*t)*psi2(t) + A2minus*cos(omega*t)*psi1(t),
                                A3minus*(sin(w*t)/w - lambda3 *cos(w*t)/w**2)/(1+(lambda3/w)**2),
                                A4minus*(sin(w*t)/w - lambda4 *cos(w*t)/w**2)/(1+(lambda4/w)**2)]))
    return XQplus,XQminus

def makeBVP(x0,f,A,Pinv,N,tmin,tmax):
    # this returns the BVP to be solved, F(X0) = 0, along with DF^(-1)
    # x0,A,Pinv all depend only on the hyprebolic trajectory being considered, i.e. either the left or right side.
    t = np.linspace(tmin,tmax,N)
    dt = (tmax-tmin)/N
    def phi(y0,t0): # evolves forward by dt
        res = solve_ivp (f, [t0,dt+t0], y0 , method = "RK45" , max_step =dt/2 , rtol = 1e-12 )
        y = res .y. transpose()
        return  y[-1]    
# fixed point function
    def F(X,N):
        # N number of points taken, X is diplacement from x0
        Y = np.zeros((N,4))
        X = X.reshape((N,4))
        for i in range(0, N-1):
            Y[i] = phi(X[i] + x0,t[i]) - X[i+1] - x0   # ensures this is a trajectory.
        # Boundary conditions (stable/unstable coordinates are zero to keep bounded ( need displacement from equilibrium)
        Y[-1][1] = Pinv.dot(X[0])[0]
        Y[-1][1] = Pinv.dot(X[0])[1]
        Y[-1][2] = Pinv.dot(X[0])[2]
        Y[-1][3] = Pinv.dot(X[-1])[3] # unstable
        return Y.reshape(4*N)
    Dphi = scipy.linalg.expm(dt*A) # matrix exponential for linear flow displacement from x0
    # Derivative for linear system
    # Amend
    DF0 = np.zeros((4*N,4*N))
    for i in range(0,N-1):
        # find linear derivative
        DF0[4*i:4*i + 4, 4*i:4*i + 4] = Dphi
        DF0[4*i:4*i +4,4*i+4:4*i+8] = -np.eye(4)
    DF0[4*N-4,0:4] = np.dot(Pinv.transpose(),array([1,0,0,0]))
    DF0[4*N-3,0:4] = np.dot(Pinv.transpose(),array([0,1,0,0]))
    DF0[4*N-2,0:4] = np.dot(Pinv.transpose(),array([0,0,1,0]))
    DF0[4*N-1,4*N-4:] = np.dot(Pinv.transpose(),array([0,0,0,1]))
    Permute, L, U = scipy.linalg.lu(DF0) # numerical LU decomposition. This is computed once to maximise efficiency
    def DF0solve(y0):   # we only store the solution function to DF*h = y.
        x0 = np.transpose(Permute).dot(y0)
        return Usolve(Lsolve(x0,L),U)
    return phi,F,DF0,DF0solve
def findTrajectory(X0,epsF,epsC,K,F,DF0,DF0solve,N):  # This finds X which satisfies:
    # norm(F(X)) < epsF, by a Newton method with convervence criteria from epsC.
    # X0 is the intial guess.
    i = 0
    Fx = F(X0,N)
    X = X0.reshape(4*N)
    normFx = np.linalg.norm(Fx)
    delta = DF0solve(Fx)
    foundTrajectory = False
    while  (normFx > epsF or np.linalg.norm(delta) > epsC) and i < K:
 #       print(normFx)
        delta = DF0solve(Fx)
        X = X-delta # update sequence
        i += 1 # add counter
        Fx = F(X,N) # Evaluate at new poin#t
        normFx = np.linalg.norm(Fx) # compute norm to check the estimate
        if normFx > 100:
            break
    if (normFx < epsF and np.linalg.norm(delta) < epsC):
        foundTrajectory = True
    X = X.reshape((N,4))
    X0 = X0.reshape((N,4))
    print(" after " + str(i) + " iterations an estimate with error of F =" + str(normFx) + " and delta = "+ str(np.linalg.norm(delta))+ "was produced")
    return X,foundTrajectory
def ForcingType(ForcingParameters,forcingtype):    
    #forcingtype = 'periodic','quasi-periodic' supported where they are linear combinations of cosine waves.
    # ForcingParameters = (Amplitude,frequency) or (Amplitude1,frequency1,Amplitude2,frequency2 )
    if forcingtype == 'periodic':
        Amp,w= ForcingParameters
        def findHyperbolicTrajectory(N,T,h,k,D,G):
            f,Aplus, Aminus, xplus, xminus = model(h,k,D,G)    # the vector field, matrices for linearised systems at equilibria, and equilibria
            Pplus,Pplusinv,Pminus,Pinvminus,lambda2, omega, lambda3, lambda4  =  eigen(h,k)  # the change of basis matrices, so Pplus.dot(array([1,0,0,0])) is the eigenvector along centre direction for postivie capsis
            tmin, tmax = -T, T
            t = np.linspace(tmin,tmax,N)
            dt = (tmax-tmin)/N
            XQ1plus,XQ1minus =  linearHyperbolicCosine(Amp,w,Pplus,Pminus,Pplusinv,Pinvminus,lambda2, omega, lambda3, lambda4)
            X0plus = XQ1plus(t).transpose() #  len(t) by 4 array
            X0minus =  XQ1minus(t).transpose()
            phi,Fplus,DF0plus,DF0plussolve = makeBVP(xplus,f,Aplus,Pplusinv,N,tmin,tmax)
            phi,Fminus,DF0minus,DF0minussolve = makeBVP(xminus,f,Aminus,Pinvminus,N,tmin,tmax)
            epsF = 1e-7
            epsC = 1e-8
            K = 200
            Xhyperbolicplus,foundTrajectoryPlus = findTrajectory(X0plus,epsF,epsC,K,Fplus,DF0plus,DF0plussolve,N)
            Xhyperbolicminus,foundTrajectoryMinus = findTrajectory(X0minus,epsF,epsC,K,Fminus,DF0minus,DF0minussolve,N)
            return Xhyperbolicplus+xplus ,Xhyperbolicminus +xminus, t, X0plus,X0minus,foundTrajectoryPlus,foundTrajectoryMinus
    elif forcingtype == 'quasi-periodic':
        if len(ForcingParameters) % 3 != 0:
            print("Define forcing properly")
        n = len(ForcingParameters)//3
        Amp = np.zeros(n)
        w = np.zeros(n)
        eps = np.zeros(n)
        for i in range(0,n):
            Amp[i] = ForcingParameters[3*i]
            w[i] = ForcingParameters[3*i+1]
            eps[i] = ForcingParameters[3*i+2]

        def findHyperbolicTrajectory(N,T,h,k,D,G):
            f,Aplus, Aminus, xplus, xminus = model(h,k,D,G)    # the vector field, matrices for linearised systems at equilibria, and equilibria
            Pplus,Pplusinv,Pminus,Pinvminus,lambda2, omega, lambda3, lambda4  =  eigen(h,k)  # the change of basis matrices, so Pplus.dot(array([1,0,0,0])) is the eigenvector along centre direction for postivie capsis
            tmin, tmax = -T, T
            t = np.linspace(tmin,tmax,N)
            X0plus = np.zeros((N,4))
            X0minus = np.zeros((N,4))
            for i in range(0,n):
                XQ1plus,XQ1minus =  linearHyperbolicCosine(Amp[i],w[i],Pplus,Pminus,Pplusinv,Pinvminus,lambda2, omega, lambda3, lambda4)
                X0plus += XQ1plus(t + eps[i]/w[i]).transpose()#  len(t) by 4 array
                X0minus +=  XQ1minus(t+ eps[i]/w[i]).transpose() 
            phi,Fplus,DF0plus,DF0plussolve = makeBVP(xplus,f,Aplus,Pplusinv,N,tmin,tmax)
            phi,Fminus,DF0minus,DF0minussolve = makeBVP(xminus,f,Aminus,Pinvminus,N,tmin,tmax)
            epsF = 1e-7
            epsC = 1e-8
            K = 200
            Xhyperbolicplus,foundTrajectoryPlus = findTrajectory(X0plus,epsF,epsC,K,Fplus,DF0plus,DF0plussolve,N)
            Xhyperbolicminus,foundTrajectoryMinus = findTrajectory(X0minus,epsF,epsC,K,Fminus,DF0minus,DF0minussolve,N)
            return Xhyperbolicplus + xplus ,Xhyperbolicminus + xminus, t, X0plus,X0minus,foundTrajectoryPlus,foundTrajectoryMinus
    else:
        print("Clarify forcing function type and parameters")
    return findHyperbolicTrajectory