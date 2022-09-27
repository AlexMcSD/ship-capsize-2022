# This is where we store the function to make a parameterisation for the stable manifold given the data
import numpy as np
from numpy.linalg import norm,solve
from numpy import array
import scipy.linalg
import scipy.io
import scipy.interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator
# importing the data
pointsSMplus= np.load("pointsSMplus.npy")
valxSplus = np.load("valxSplus.npy")
valySplus= np.load("valySplus.npy")
valvxSplus= np.load("valvxSplus.npy")
valvySplus= np.load("valvySplus.npy")
timekeyplus = np.load("timekeyplus.npy")
pointsSMminus = np.load("pointsSMminus.npy")
valxSminus = np.load("valxSminus.npy")
valySminus= np.load("valySminus.npy")
valvxSminus= np.load("valvxSminus.npy")
valvySminus= np.load("valvySminus.npy")
timekeyminus  = np.load("timekeyminus.npy")
thyperbolic = np.load("thyperbolic.npy")
tindicesSplus = np.load("tindicesSplus.npy")
tindicesSminus = np.load("tindicesSminus.npy")
from numpy import nan
def getInterpolatedImmersionS(CapsizeSide, method ="rbf"): # CapsizeSide determines whih side we consider, # method can be RBF interpolation of Linear interpolation
    # rbf is normally better as it is smooth and is globablly defined.
    if CapsizeSide == 'positive':  # we set the specific points/values to be used depending on the capsize side
        pointsSM = pointsSMplus
        valxS = valxSplus
        valyS = valySplus
        valvxS = valvxSplus
        valvyS = valvySplus
        timekeys = timekeyplus.astype(int)
        t0indices = tindicesSplus
    elif CapsizeSide == 'negative':
        pointsSM = pointsSMminus
        valxS = valxSminus
        valyS = valySminus
        valvxS = valvxSminus
        valvyS = valvySminus
        timekeys = timekeyminus.astype(int)
        t0indices = tindicesSminus
    else:
        print("Define CapsizeSide")
    if method == 'linear':
        # for the linear interplation, we interpolate the map given by [q1,q2,q3,t] -> [valx,valy,valvx,valvy] 
        newpoints = np.zeros((pointsSM.shape[0],4))
        for i in range(0,len(t0indices)):
            t = thyperbolic[t0indices[i]]
            for j in range(timekeys[i],timekeys[i+1]):
                newpoints[j] = np.hstack((pointsSM[j],array([t])))
        interpxS =LinearNDInterpolator(newpoints,valxS) 
        interpyS =LinearNDInterpolator(newpoints,valyS) 
        interpvxS =LinearNDInterpolator(newpoints,valvxS) 
        interpvyS =LinearNDInterpolator(newpoints,valvyS) 
        def xSM(q,t): # we return it this way to make it easier to fix time.
            return interpxS(np.hstack((q,array([t]))))
        def ySM(q,t):
            return interpyS(np.hstack((q,array([t]))))
        def vxSM(q,t):
            return interpvxS(np.hstack((q,array([t]))))
        def vySM(q,t):
            return interpvyS(np.hstack((q,array([t]))))
    if method == 'rbf': # for rbf we do rbf interpolation on each time slice, and then linearly interpolate bewtween time slices.
        # to this we create a dictionary of RBFInterpolators corrspoinding for each time slice
        points = pointsSM[:timekeys[1]]
        interpxS = {thyperbolic[t0indices[0]]:RBFInterpolator(points,valxS[:timekeys[1]])  }
        interpyS = {  thyperbolic[t0indices[0]]: RBFInterpolator(points,valyS[:timekeys[1]] )}
        interpvxS = {  thyperbolic[t0indices[0]]: RBFInterpolator(points,valvxS[:timekeys[1]]) }
        interpvyS = {  thyperbolic[t0indices[0]]: RBFInterpolator(points,valvyS[:timekeys[1]]) }
        
        for l in range(1,len(t0indices)):
            points = pointsSM[timekeys[l]:timekeys[l+1]]
            for i in range(0,points.shape[0]-1):
                point = points[i]
                for j in range(i+1,points.shape[0]):
                    if np.linalg.norm(points[j] - point) < 1e-8:
                        print("error")
            interpxS[ thyperbolic[t0indices[l]]] = RBFInterpolator(points,valxS[timekeys[l]:timekeys[l+1]])
            interpyS[ thyperbolic[t0indices[l]]] = RBFInterpolator(points,valyS[timekeys[l]:timekeys[l+1]])
            interpvxS [ thyperbolic[t0indices[l]]]= RBFInterpolator(points,valvxS[timekeys[l]:timekeys[l+1]])
            interpvyS[ thyperbolic[t0indices[l]]] = RBFInterpolator(points,valvyS[timekeys[l]:timekeys[l+1]])     
            
        def xSM(q,t): # we search through the sample times,then output the linear interpolation of the RBF interpolations for the smapled time slices above and below the time t.
            i = 0
            for l in range(0,len(t0indices)-1):
                i+=1
                if thyperbolic[t0indices[l]] == t:
                    #l = len(t0indices)
                    return  interpxS[thyperbolic[t0indices[l]]](array([q]))[0]
                if thyperbolic[t0indices[l]] < t and thyperbolic[t0indices[l+1]]> t:
                    t0 = thyperbolic[t0indices[l]] 
                    t1 = thyperbolic[t0indices[l+1]]
                    break
            if i< len(t0indices) - 2:
                SM1 = interpxS[ thyperbolic[t0indices[l]]](array([q]))[0]
                SM2 = interpxS[ thyperbolic[t0indices[l+1]]](array([q]))[0]
                return SM1 + ((t - t)/(t1-t))*(SM2-SM1)
            else: 
                return nan
        def ySM(q,t):  # we search through the sample times,then output the linear interpolation of the RBF interpolations for the smapled time slices above and below the time t.
            i = 0
            for l in range(0,len(t0indices)-1):
                i+=1
                if thyperbolic[t0indices[l]] == t:
                    return  interpyS[thyperbolic[t0indices[l]]](array([q]))[0]
                if thyperbolic[t0indices[l]] < t and thyperbolic[t0indices[l+1]]> t:
                    t0 = thyperbolic[t0indices[l]] 
                    t1 = thyperbolic[t0indices[l+1]]
                    break
            if i< len(t0indices) - 2:
                SM1 = interpyS[ thyperbolic[t0indices[l]]](array([q]))[0]
                SM2 = interpyS[ thyperbolic[t0indices[l+1]]](array([q]))[0]
                return SM1 + ((t - t)/(t1-t))*(SM2-SM1)
            else: 
                return nan
        def vxSM(q,t):  # we search through the sample times,then output the linear interpolation of the RBF interpolations for the smapled time slices above and below the time t.
            i = 0
            for l in range(0,len(t0indices)-1):
                i+=1
                if thyperbolic[t0indices[l]] == t:
                    return  interpvxS[thyperbolic[t0indices[l]]](array([q]))[0]
                if thyperbolic[t0indices[l]] < t and thyperbolic[t0indices[l+1]]> t:
                    t0 = thyperbolic[t0indices[l]] 
                    t1 = thyperbolic[t0indices[l+1]]
                    break
            if i< len(t0indices) - 2:
                SM1 = interpvxS[ thyperbolic[t0indices[l]]](array([q]))[0]
                SM2 = interpvxS[ thyperbolic[t0indices[l+1]]](array([q]))[0]
                return SM1 + ((t - t)/(t1-t))*(SM2-SM1)
            else: 
                return nan
        def vySM(q,t):  # we search through the sample times,then output the linear interpolation of the RBF interpolations for the smapled time slices above and below the time t.
            i = 0
            for l in range(0,len(t0indices)-1):
                i+=1
                if thyperbolic[t0indices[l]] == t:
                    return  interpvyS[thyperbolic[t0indices[l]]](array([q]))[0]
                if thyperbolic[t0indices[l]] < t and thyperbolic[t0indices[l+1]]> t:
                    t0 = thyperbolic[t0indices[l]] 
                    t1 = thyperbolic[t0indices[l+1]]
                    break
            if i< len(t0indices) - 2:
                SM1 = interpvyS[ thyperbolic[t0indices[l]]](array([q]))[0]
                SM2 = interpvyS[ thyperbolic[t0indices[l+1]]](array([q]))[0]
                return SM1 + ((t - t)/(t1-t))*(SM2-SM1)
            else: 
                return nan
    delta= 1e-10 # used to differentiate iota
    e1,e2,e3 = array([1,0,0]),array([0,1,0]),array([0,0,1])
    iota = lambda q,t : array([xSM(q,t),ySM(q,t),vxSM(q,t),vySM(q,t) ]) # this is the parameterisation of the stable manifold.
    Diota = lambda q,t : np.vstack(( (iota(q+delta*e1,t) - iota(q,t))/delta ,  (iota(q+delta*e2,t) - iota(q,t))/delta, (iota(q+delta*e3,t) - iota(q,t))/delta )).transpose() # derivative of immrsion
    return iota, Diota,xSM,ySM,vxSM,vySM