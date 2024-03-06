import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d 


linePoints = 400
randPoints = 200

m = 1
c = 0
v = 0.1 
permVariation = 0.1
iterations    = 100000 
xPoints = np.random.random( linePoints )
yPoints = xPoints*m + c + (1-2*np.random.random( linePoints ))*v

m = -1
c = 1
v = 0.1 

xPoints = np.concatenate( (xPoints, np.random.random( linePoints )) )  
yPoints = np.concatenate( (yPoints, xPoints[-linePoints:]*m + c + (1-2*np.random.random( linePoints ))*v) )  

xPoints = np.concatenate( (xPoints, np.random.random( randPoints )) )
yPoints = np.concatenate( (yPoints, np.random.random( randPoints )) )


def calcRVal( inpM, inpC ):
    rvals = ((yPoints-(inpC+inpM*xPoints))/np.sqrt(inpM**2 + 1))**2
    avgRSQ = np.mean( rvals )

    return avgRSQ, rvals

def get2Points( len ):
    point1 = int(np.random.random()*len)
    point2 = int(np.random.random()*(len-1))

    if ( point2 >= point1 ):
        return point1, point2+1
    return point1, point2



totalError = np.zeros( xPoints.size )
useError =  np.zeros( xPoints.size )

bestM = 0
bestC = 0
bestE = 999999999999999

for i in range(0, iterations):
    index1, index2 = get2Points( xPoints.size )

    gM = (yPoints[index1]-yPoints[index2])/(xPoints[index1]-xPoints[index2])
    gC = yPoints[index1] - xPoints[index1]*gM

    avError, indError = calcRVal( gM, gC )
    
    totalError += indError
    useError[index1] = (useError[index1]*10 + avError)/11
    useError[index2] = (useError[index2]*10 + avError)/11

    if ( avError < bestE ):
        bestM = gM
        bestC = gC
        bestE = avError  

totalError /= iterations

fM, fC = np.polyfit(xPoints, yPoints, deg=1) 
print("1m=",fM,"\tc=",fC) 
plt.plot( [0,1], [fC,fC+fM], "b" ) 
plt.plot( xPoints, yPoints, "rx" ) 

plt.figure(2) 
inlierMask = calcRVal( bestM, bestC )[1] < permVariation
inlierMask = useError < permVariation 

fM, fC = np.polyfit(xPoints[inlierMask], yPoints[inlierMask], deg=1) 
print("2m=",fM,"\tc=",fC) 
plt.plot( [0,1], [fC,fC+fM], "b" ) 
plt.plot( xPoints[inlierMask], yPoints[inlierMask], "rx" ) 
plt.plot( xPoints[inlierMask!=True], yPoints[inlierMask!=True], "yx" ) 

plt.figure(3)
plt.plot( totalError )

plt.figure(4)
plt.plot( useError )
plt.show()


