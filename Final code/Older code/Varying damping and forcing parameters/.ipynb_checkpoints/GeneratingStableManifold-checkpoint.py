# This stores the function to find points on the stable manifold
import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp,array,sqrt
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import scipy.io
from model import linearisedSystem,model,eigen,Usolve,Lsolve
def FindStablePoints(CapsizeSide,h,k,D,G,T,N,Xhyperbolicplus,Xhyperbolicminus,t0indices,R,M,Q1,Q2,Q3,Deltatminus,Deltatplus): # This is the function called to find a sample of points on the stable manifold, the side of which determined by Capsizeside
    thyperbolic = np.linspace(-T,T,N)
    f,Aplus, Aminus, xplus, xminus = model(h,k,D,G)  # set up model
    dt = 2*T/N
    Pplus,Pplusinv,Pminus,Pinvminus,lambda2, omega, lambda3, lambda4  =  eigen(h,k)
    if CapsizeSide == "positive":  # 
        Xhyperbolic = Xhyperbolicplus
        A = Aplus # this is Df(xplus)
        P = Pplus # Eigenvectors of Aplus
        Pinv =  Pplusinv # inverese of vhange of basis matrice
    elif CapsizeSide == "negative":
        Xhyperbolic = Xhyperbolicminus
        A = Aminus
        P = Pminus
        Pinv =  Pinvminus
    Dphi = scipy.linalg.expm(dt*A)
    def defineBVP(t0index,imax, imin):#  This sets up the BVP used to find the trajectory
        l = t0index - imin
        def g1(X,q1,q2,q3):  # This is the BVP, that the solution must solve,
            # X is displacement of trajectory from the hyperbolic trajectory
            # q1, q2, q3 stable coordinates that we force the solution to satisfy at t = t0
            Y = np.zeros((imax - imin+1,4)) 
            X = X.reshape((imax - imin+1  ,4)) 
            for i in range(0, imax -imin-1):
                Y[i] = phi(X[i]+Xhyperbolic[i + imin],thyperbolic[i+imin]) - X[i+1] - Xhyperbolic[i+imin+1]
            #boundary condtions
            Y[-1][0] = np.dot(Pinv,X[l])[0] - q1 
            Y[-1][1] = np.dot(Pinv,X[l])[1] - q2
            Y[-1][2] = np.dot(Pinv,X[l])[2] - q3
            Y[-1][3] =  np.dot(Pinv,X[-1])[3]
            return Y.reshape(4*(imax - imin+1))

        def phi(y0,t0): # evolve the system forward by dt
            res = solve_ivp (f, [t0,dt+t0], y0 , method = "RK45" , max_step =dt/2 , rtol = 1e-12 )
                #y1 = y0 + dt*f(t0,y0) # Euler method for initial testing
            y = res .y. transpose()
            return  y[-1] 

        # For linearised system we 0have that
        Dg = np.zeros((4*(imax+1 - imin),4*(imax+1 - imin)))  # We set the approximate derviative to g1
        for i in range(0,(imax - imin)):
                # find linear derivative
            Dg[4*i:4*i + 4, 4*i:4*i + 4] = Dphi
            Dg[4*i:4*i +4,4*i+4:4*i+8] = -np.eye(4)
        Dg[4*(imax+1 - imin)-4,4*(l):4*(l)+4] = np.dot(Pinv.transpose(),array([1,0,0,0]))
        Dg[4*(imax+1 - imin)-3,4*(l):4*(l)+4] = np.dot(Pinv.transpose(),array([0,1,0,0]))
        Dg[4*(imax+1 - imin)-2,4*(l):4*(l)+4] = np.dot(Pinv.transpose(),array([0,0,1,0]))
        Dg[4*(imax+1 - imin)-1,4*(imax+1 - imin)-4:] = np.dot(Pinv.transpose(),array([0,0,0,1]))
        GP , GL, GU = scipy.linalg.lu(Dg) 
        def DG0solve(y0):
            x = np.transpose(GP).dot(y0)
            return Usolve(Lsolve(x,GL),GU)
        return g1, DG0solve
    
    def findStablePoint(q1,q2,q3,t0index): # finds the point on the stable manifold at t=thyperbolic[t0index] with g1(X,q1,q2,q3) = 0
        imax, imin = min(t0index + Deltatplus, N-1), max(t0index - Deltatminus,0) # Ensuring that the proceedure does not p=blow up by integrating over too long a time interval
        g1, DG0solve = defineBVP(t0index,imax, imin)
        # analytical solution to linear problem
        Xstart = lambda t : np.dot(P,np.array([exp(lambda2*t)*(q1*cos(omega*t) - q2 * sin(omega*t)) ,exp(lambda2*t)*(q1*sin(omega*t) + q2*cos(omega*t)) , t-t,t-t ])) 
        X = Xstart(thyperbolic[imin:imax+1]-thyperbolic[t0index]).transpose().reshape(4*(imax - imin + 1))
        i = 0
        Gx =  g1(X,q1,q2,q3)
        normGx = np.linalg.norm(Gx) # Initial error to Gx = 0
        epsG = 1e-6
        epsC = 1e-6
        delta = DG0solve(Gx) 
        K = 250
        foundTrajectory = False
        while  (normGx > epsG or np.linalg.norm(delta) > epsC) and i < K: # Newton method proceedure
            delta = DG0solve(Gx)
            X = X-delta # update sequence
            i += 1 # add counter
            Gx = g1(X,q1,q2,q3)
            normGx = np.linalg.norm(Gx) # compute norm to check the estimate
            if normGx > 100:
                break
        X = X.reshape((imax+1-imin,4)) + Xhyperbolic[imin:imax+1] # find the trajectory

        if (normGx < epsG and np.linalg.norm(delta) < epsC):
            foundTrajectory  = True

        #print(" after " + str(i) + " iterations an estimate with error of G =" + str(normGx) + " and delta = "+ str(np.linalg.norm(delta))+ "was produced")
        return X[t0index - imin],foundTrajectory # Outputs the point of the stable maniold at t=t0
    
    
    # We now compute the points for various values of t0, q1,q2,q3
    # The pointsSM store the q1,q2,q3 values, and will be used to parameterise the stable manifold
    #valx stores the corresponding value in x coordinate ....
    pointsSM = array([0,0,0]) 
    valxS = Xhyperbolic[t0indices[0],0]
    valyS = Xhyperbolic[t0indices[0],1]
    valvxS = Xhyperbolic[t0indices[0],2]
    valvyS =Xhyperbolic[t0indices[0],3]
    timekey = np.zeros(len(t0indices)+1)
    unfoundTrajectoriesS = 0
    counter = 0
    numberofpoints = 1
    # storing hyperbolic trajectory values.
    for l in range(0,len(t0indices)):
        if l > 0: # Storing the hyperbolic trajectory for q1,q2,q3 = 0,0,0
            X = Xhyperbolic[t0indices[l]]
            pointsSM = np.vstack((pointsSM,array([0,0,0])))
            valxS = np.hstack((valxS,X[0]))
            valyS = np.hstack((valyS,X[1]))
            valvxS = np.hstack((valvxS,X[2]))
            valvyS = np.hstack((valvyS,X[3]))
            numberofpoints +=1
        for i in range(0, len(Q1)):
            for j in range(0,len(Q2)):
                    for m in range(0,len(Q3)):
                        X,foundTrajectory= findStablePoint(Q1[i],Q2[j],Q3[m],t0indices[l]) # finding the stable point
                        counter +=1
                        if counter%((len(t0indices)*M**3)//4) ==0:
                            print(str(100*counter/(len(t0indices)*M**3))+ "% trajectories calculated")
                            print(" Trajectories not found: " +str(100*unfoundTrajectoriesS/counter) + "%")
 
                        if foundTrajectory: # storing the values of each coodinate and the corresponding values
                            pointsSM = np.vstack((pointsSM,array([Q1[i],Q2[j],Q3[m]])))
                            valxS = np.hstack((valxS,X[0]))
                            valyS = np.hstack((valyS,X[1]))
                            valvxS = np.hstack((valvxS,X[2]))
                            valvyS = np.hstack((valvyS,X[3]))
                            numberofpoints +=1
                        else:
                            unfoundTrajectoriesS +=1
        timekey[l+1] = numberofpoints   # This ensures we can extract the stablemanifold for t = thyperbolic[t0indices[l]], by considering the values/points at  [ timekey[l].timekey[l+1]]
        
    print("Stable manifold algorithm failed to find: " + str(unfoundTrajectoriesS) + " out of " + str(M**3*len(t0indices)))
    return pointsSM, valxS,valyS,valvxS,valvyS,timekey