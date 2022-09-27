import numpy as np
from numpy import cos, sin, cosh, sinh, tanh, array,pi, exp,array,sqrt
from numpy.linalg import norm,solve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg
import scipy.io
from model import linearisedSystem,model,eigen,Usolve,Lsolve
from StableManifoldImmersion import getInterpolatedImmersionS
from numpy import nan
from scipy.interpolate import RBFInterpolator

Xhyperbolicplus= np.load('Xhyperbolicplus.npy')
Xhyperbolicminus= np.load('Xhyperbolicminus.npy')
pointsSMplus= np.load("pointsSMplus.npy")
valxSplus = np.load("valxSplus.npy")
valySplus= np.load("valySplus.npy")
valvxSplus= np.load("valvxSplus.npy")
valvySplus= np.load("valvySplus.npy")
pointsSMminus = np.load("pointsSMminus.npy")
valxSminus = np.load("valxSminus.npy")
valySminus= np.load("valySminus.npy")
valvxSminus= np.load("valvxSminus.npy")
valvySminus= np.load("valvySminus.npy")

    #Data used to contruct the interpolation of the immersion
timekeyminus  = np.load("timekeyminus.npy")
timekeyplus = np.load("timekeyplus.npy")
Rplus = np.load("Rplus.npy")[0]
Rminus = np.load("Rminus.npy")[0]
    # we create the parameterisation of the stable manifolds
iotaplus, Diotaplus,xSMplus,ySMplus,vxSMplus,vySMplus = getInterpolatedImmersionS("positive",'rbf')  
iotaminus, Diotaminus,xSMminus,ySMminus,vxSMminus,vySMminus = getInterpolatedImmersionS("negative",'rbf')
tindicesSplus = np.load("tindicesSplus.npy")
tindicesSminus = np.load("tindicesSminus.npy")
tindices = tindicesSplus.astype(int)
t0index = tindices[len(tindices)//2]

def MakeFindIntegrityMeasure(h,k,D,G,T,N,Rplus,Rminus):
    thyperbolic = np.linspace(-T,T,N)
    f,Aplus, Aminus, xplus, xminus = model(h,k,D,G)
    dt = 2*T/N
    Pplus,Pplusinv,Pminus,Pinvminus,lambda2, omega, lambda3, lambda4  =  eigen(h,k)
    t0  = thyperbolic[t0index]
    def MakeDividingManifold(CapsizeSide): # this function creates the dividing manifolds for each side of capsize
        if CapsizeSide == "positive": 
            Xhyperbolic = Xhyperbolicplus
            P = Pplus #Eigenbasis matrix
            Pinv =  Pplusinv
            Xhyperbolic1 = Xhyperbolicplus[:,0]
            Xhyperbolic2 = Xhyperbolicplus[:,1]
            Xhyperbolic3 = Xhyperbolicplus[:,2]
            Xhyperbolic4 = Xhyperbolicplus[:,3]
            iota = iotaplus
            Diota = Diotaplus
        elif CapsizeSide == "negative":
            Xhyperbolic = Xhyperbolicminus
            P = Pminus # Eigenbasis matrix
            Pinv =  Pinvminus
            Xhyperbolic1 = Xhyperbolicminus[:,0]
            Xhyperbolic2 = Xhyperbolicminus[:,1]
            Xhyperbolic3 = Xhyperbolicminus[:,2]
            Xhyperbolic4 = Xhyperbolicminus[:,3]
            iota = iotaminus
            Diota = Diotaminus
        # We interpolate the hyperbolic trajectories.
        interpXhyperbolic1 = scipy.interpolate.UnivariateSpline(thyperbolic,Xhyperbolic1,k=3)
        interpXhyperbolic2 = scipy.interpolate.UnivariateSpline(thyperbolic,Xhyperbolic2,k=3)
        interpXhyperbolic3 = scipy.interpolate.UnivariateSpline(thyperbolic,Xhyperbolic3,k=3)
        interpXhyperbolic4 = scipy.interpolate.UnivariateSpline(thyperbolic,Xhyperbolic4,k=3)
        Xhyperbolic = lambda t: array([interpXhyperbolic1(t),interpXhyperbolic2(t),interpXhyperbolic3(t),interpXhyperbolic4(t)]) # We interpolate the hyperbolic trajectory, to get a smooth function
        def dividingManifold(y,t):    # The dividing manifold are the points (y,t) with dividingManifold(y,t) = 0.
            if np.linalg.norm(Diota(array([0,0,0]),t)) < 10000:
                e1 = array([1,0,0])
                e2 = array([0,1,0])
                e3 = array([0,0,1])
                stable = (Diota(array([0,0,0]),t).dot(e3))/np.linalg.norm((Diota(array([0,0,0]),t).dot(e3)))
                centre1 = (Diota(array([0,0,0]),t).dot(e1))/np.linalg.norm((Diota(array([0,0,0]),t).dot(e1)))
                centre2 = (Diota(array([0,0,0]),t).dot(e2))/np.linalg.norm((Diota(array([0,0,0]),t).dot(e2)))
                A = np.vstack((centre1,centre2,stable + P.dot(array([0,0,0,1])))).transpose() # The direction vectors of the dividing manifold at time $t$
                DividingPlane = scipy.linalg.orth(A)
                u1,u2,u3 = DividingPlane.dot(array([1,0,0])),DividingPlane.dot(array([0,1,0])),DividingPlane.dot(array([0,0,1]))
                normal = (stable  - u1.dot(stable)*u1 - u2.dot(stable)*u2 - u3.dot(stable)*u3)/np.linalg.norm(stable  - u1.dot(stable)*u1 - u2.dot(stable)*u2 - u3.dot(stable)*u3) # normal to the hyperplane
                d = normal.dot(Xhyperbolic(t))
                return normal.dot(y) - d # This is zero when (y,t) lies on the dividing manifold 
            else:
                return nan
        return dividingManifold

    def MakefindDistanceSM(CapsizeSide): # makes the funciton to measure distance between a point and stable manifold
        if CapsizeSide == "positive":
            iota = iotaplus
            Diota = Diotaplus
        elif CapsizeSide == "negative":
            iota = iotaminus
            Diota = iotaminus
        def findDistanceSM(y0,t0):
            # this computes the distance between y0 and the stable manifold at $t = t_0$.
            dist = lambda q : np.linalg.norm(iota(q,t0) - y0)**2 # distance of a point on the SM and y0
            Ddist = lambda q: 2*Diota(q,t0).transpose().dot((iota(q,t0) -y0)[0])
            q0 = array([0,0,0])
            eps = 1e-3
            minimise = scipy.optimize.minimize(dist,q0, method='Powell' ) # we minimise the distance function
            if minimise.success:
                q = minimise.x
                d = np.linalg.norm(iota(q,t0) - y0)
                return q,d, minimise.success # We output a point of minimal distance, the distance and whether the minimisation process was a success
            else:
                q = minimise.x
                if dist(q) < eps:
                    d = dist(q)
                    return q,d**0.5, True
                else:
                    print("Error for y0 = " + str(y0) + ": " + minimise.message)
                return q0,-1,False
        return findDistanceSM

    def TimetoCapsize(y0,dividingManifold,CapsizeSide,capsizeregion,findDistanceSM):
        if abs( np.sign(dividingManifold(y0,t0))- capsizeregion) >1:
            iscapsize, notTooCloseToSM =  isCapsizepoint(y0,t0,findDistanceSM,dividingManifold,CapsizeSide,capsizeregion)
            if iscapsize:
                return 0
            else:
                return nan
        else:
            def event(t,y):
                return dividingManifold(y,t)
            event.terminal = True
            res = solve_ivp (f, [t0,100+t0], y0 , method = "RK45" , events=[event],  max_step =0.05 , rtol = 1e-12 )
            y = res .y. transpose()
            T =res.t.transpose()
            if CapsizeSide == 'positive':
                if T[-1]<40 and y[-1,1] > 0:  # test for positive roll angle only
                    return T[-1] # -t0
                else:
                    return nan
            elif CapsizeSide == 'negative':
                if y[-1,1] < 0:
                    return T[-1] # -t0
                else:
                    return nan    
  
    findDistanceSMminus = MakefindDistanceSM('negative')
    dividingManifoldminus = MakeDividingManifold('negative')
    strongstablepointminus =  Xhyperbolicminus[t0index] -0.1*Pminus.dot(array([0,0,1,0]))
    capsizeregionminus  = dividingManifoldminus(strongstablepointminus,t0)
    findDistanceSMplus = MakefindDistanceSM('positive')
    dividingManifoldplus = MakeDividingManifold('positive')
    strongstablepointplus =  Xhyperbolicplus[t0index] +0.1*Pplus.dot(array([0,0,1,0]))
    capsizeregionplus  = np.sign(dividingManifoldplus(strongstablepointplus,t0))
    H = lambda y: (.25*y[2]**2)/h + .5*y[3]**2 + .5*(y[1]**2 +.5*y[0]**2 - y[0]*y[1]**2)
    def U(y0):
        return H(y0) <.25 and abs(y0[1]) <= 1
    
    q1Splus = np.linspace(-Rplus+0.1,Rplus-0.2,25)
    q2Splus =np.linspace(-Rplus+0.1,Rplus-0.2,25)
    q3Splus =np.linspace(-Rplus+0.1,Rplus-0.2,25)
    t0 = thyperbolic[t0index]
    #t = thyperbolic[t0indices[0]:t0indices[-1]+1]
    #tindices = np.arange(t0indices[0],t0indices[-1],2)
    xplus = np.zeros((len(q1Splus),len(q2Splus),len(q3Splus)))
    yplus = np.zeros((len(q1Splus),len(q2Splus),len(q3Splus)))
    vxplus = np.zeros((len(q1Splus),len(q2Splus),len(q3Splus)))
    vyplus = np.zeros((len(q1Splus),len(q2Splus),len(q3Splus)))
    startPoints = True
    for l in range(0, len(q1Splus)):
        for j in range(0,len(q2Splus)):
            for m in range(0,len(q3Splus)):
                xplus[l,j,m] = xSMplus(array([q1Splus[l],q2Splus[j],q3Splus[m]]),t0 )    # store values of the stable manifold at t = t0
                yplus[l,j,m] = ySMplus(array([q1Splus[l],q2Splus[j],q3Splus[m]]),t0)  
                vxplus[l,j,m] = vxSMplus(array([q1Splus[l],q2Splus[j],q3Splus[m]]),t0)
                vyplus[l,j,m] = vySMplus(array([q1Splus[l],q2Splus[j],q3Splus[m]]),t0)
                if startPoints:
                    pointsplus = array( [xplus[l,j,m],yplus[l,j,m],vxplus[l,j,m]]) #store x,y,vx values
                    vyvalsplus =array([vyplus[l,j,m ]]) # store vy values
                    startPoints = False
                else:
                    pointsplus = np.vstack((pointsplus,array([xplus[l,j,m],yplus[l,j,m],vxplus[l,j,m]])))  #store x,y,vx values
                    vyvalsplus = np.hstack((vyvalsplus,array([vyplus[l,j,m ]])))  # store vy values
    vyvalsgraphplus = RBFInterpolator(pointsplus,vyvalsplus)                      # stores stable manifold as a graph     

    q1Sminus = np.linspace(-Rminus+0.1,Rminus-0.2,25)
    q2Sminus =np.linspace(-Rminus+0.1,Rminus-0.2,25)
    q3Sminus =np.linspace(-Rminus+0.1,Rminus-0.2,25)
    t0 = thyperbolic[t0index]
    #t = thyperbolic[t0indices[0]:t0indices[-1]+1]
    #tindices = np.arange(t0indices[0],t0indices[-1],2)
    xminus = np.zeros((len(q1Sminus),len(q2Sminus),len(q3Sminus)))
    yminus = np.zeros((len(q1Sminus),len(q2Sminus),len(q3Sminus)))
    vxminus = np.zeros((len(q1Sminus),len(q2Sminus),len(q3Sminus)))
    vyminus = np.zeros((len(q1Sminus),len(q2Sminus),len(q3Sminus)))
    startPoints = True
    for l in range(0, len(q1Sminus)):
        for j in range(0,len(q2Sminus)):
            for m in range(0,len(q3Sminus)):
                xminus[l,j,m] = xSMminus(array([q1Sminus[l],q2Sminus[j],q3Sminus[m]]),t0 )    # store values of the stable manifold at t = t0
                yminus[l,j,m] = ySMminus(array([q1Sminus[l],q2Sminus[j],q3Sminus[m]]),t0)  
                vxminus[l,j,m] = vxSMminus(array([q1Sminus[l],q2Sminus[j],q3Sminus[m]]),t0)
                vyminus[l,j,m] = vySMminus(array([q1Sminus[l],q2Sminus[j],q3Sminus[m]]),t0)
                if startPoints:
                    pointsminus = array( [xminus[l,j,m],yminus[l,j,m],vxminus[l,j,m]]) #store x,y,vx values
                    vyvalsminus =array([vyminus[l,j,m ]])  # store vy values
                    startPoints = False
                else:
                    pointsminus = np.vstack((pointsminus,array([xminus[l,j,m],yminus[l,j,m],vxminus[l,j,m]])))  #store x,y,vx values
                    vyvalsminus = np.hstack((vyvalsminus,array([vyminus[l,j,m ]])))  # store vy values
    vyvalsgraphminus = RBFInterpolator(pointsminus,vyvalsminus)             # stores stable manifold as a graph              

    def isCapsizepoint(y0,t0,findDistanceSM,dividingManifold,CapsizeSide,capsizeregion): # function to determine if a point is a capsize point or not
        if CapsizeSide == 'positive': # allocates side to check
            vyvalsgraph = vyvalsgraphplus 
        if CapsizeSide == 'negative':
            vyvalsgraph = vyvalsgraphminus
        SMintersection = findDistanceSM(y0,t0) # check intial distance
        distSM   = SMintersection[1]
        notTooCloseToSM = (distSM > 0.05) # check to make sure it is significantly far away from stable manifold
        point = y0[0:3] # the x,y,vx values
        if CapsizeSide == 'positive':
            if y0[3] < vyvalsgraph(array([point])): # checking the side of stable manifold
                return False,notTooCloseToSM
            else:
                return True, notTooCloseToSM
        elif CapsizeSide == 'negative':
            if y0[3] > vyvalsgraph(array([point])): # checking the side of stable manifold
                return False,notTooCloseToSM
            else:
                return True, notTooCloseToSM

    def findIntegrityMeasure(Npoints):
        print("Beginning Simulations for Integrity Measure")
        # need to esti`mate maximum values to find the volume
        findDistanceSMminus = MakefindDistanceSM('negative')
        dividingManifoldminus = MakeDividingManifold('negative')
        strongstablepointminus =  Xhyperbolicminus[t0index] -0.1*Pminus.dot(array([0,0,1,0]))
        capsizeregionminus  = dividingManifoldminus(strongstablepointminus,t0)
        findDistanceSMplus = MakefindDistanceSM('positive')
        dividingManifoldplus = MakeDividingManifold('positive')
        strongstablepointplus =  Xhyperbolicplus[t0index] +0.1*Pplus.dot(array([0,0,1,0]))
        capsizeregionplus  = np.sign(dividingManifoldplus(strongstablepointplus,t0))
        H = lambda y: (.25*y[2]**2)/h + .5*y[3]**2 + .5*(y[1]**2 +.5*y[0]**2 - y[0]*y[1]**2)
        def U(y0):
            return H(y0) <.25 and abs(y0[1]) <= 1
        X =  np.random.uniform(-1,1,Npoints)  # choose a bounding rectangle
        Y = np.random.uniform(-1,1,Npoints)
        #Y = array([valy0])
        VX = np.random.uniform(-sqrt(h),sqrt(h),Npoints)
        VY = np.random.uniform(-sqrt(2),sqrt(2),Npoints)
        totalinU = 0
        total = 0
        capsizes = 0
        tooclose = 0
        safepoints = 0
        for i in range(0,len(X)):
            for j in range(0,len(Y)):
                for l in range(0,len(VX)):
                    for m in range(0,len(VY)):
                        y0 = array([X[i],Y[j],VX[l],VY[m]])
                        total +=1
                        if total % (Npoints**4//10) ==0:
                            print(str(100*total/(len(X)*len(Y)*len(VX)*len(VY))) + " % completed")
                        if U(y0):
                            totalinU +=1
                            confirmedCapsize = False
                            for CapsizeSide in ('positive','negative'):
                                if confirmedCapsize == False:
                                    if CapsizeSide == 'positive':
                                        findDistanceSM = findDistanceSMplus
                                        dividingManifold = dividingManifoldplus
                                        capsizeregion  = capsizeregionplus
                                    elif CapsizeSide == 'negative':
                                        findDistanceSM = findDistanceSMminus
                                        dividingManifold = dividingManifoldminus
                                        capsizeregion  = capsizeregionminus
                                 #   print(findDistanceSM(y0,t0))
                                    iscapsize,nottoclose = isCapsizepoint(y0,t0,findDistanceSM,dividingManifold,CapsizeSide,capsizeregion)
                                    if nottoclose:
                                        if iscapsize == True:
                                            capsizes +=1
                                            T =TimetoCapsize(y0,dividingManifold,CapsizeSide,capsizeregion,findDistanceSM)
                                            if capsizes == 1:
                                                capsizepoints = y0.copy().astype(np.float64)
                                                capsizeTimes = array([T])
                                            else:
                                                capsizepoints = np.vstack((capsizepoints,y0.copy().astype(np.float64)))
                                                capsizeTimes = np.vstack((capsizeTimes,array([T])))
                                            confirmedCapsize = True
                                        else:
                                            safepoints +=1
                                            if safepoints == 1:
                                                noncapsizepoints = y0.copy().astype(np.float64)
                                            else:
                                                noncapsizepoints = np.vstack((noncapsizepoints,y0.copy().astype(np.float64)))
                                    else:
                                        tooclose +=1
        VolU = sqrt(2*h)*np.pi*163/210 
        return VolU*(1-capsizes/totalinU)
    return findIntegrityMeasure