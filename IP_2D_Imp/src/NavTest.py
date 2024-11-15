import matplotlib.pyplot as plot 

from Navigator import Navigator

from time import sleep as delay

import random

import livePlotter as lp 

maxTime = 4001
timeStep = 0.05
 
wheel_seperation = 0.135 * 2
wheel_diameter   = 0.075*2

navigator = Navigator( wheel_diameter, wheel_seperation )
TARGET_Xs = [0, 2, 0]
TARGET_Ys = [2, 2, 0]
for i in range(0, len(TARGET_Xs)):
    navigator.addTarget(TARGET_Xs[i], TARGET_Ys[i])  

velLwh = 0
velRwh = 0

x = []
y = []
alpha = []

realX = []
realY = []
realAlpha = []

velL = []
velR = []

t = []

errorFrac = 0.0
bias = 0.0

def errorMult():
    return ((random.random()-0.5)*errorFrac + 1 )

shownPrev = False

dispWind = lp.PlotWindow( 3,6 )
roboDisp = lp.RobotDisplay(0,0,3,3, dispWind, -5, 5, -5, 5)   

def showChartLive():
    roboDisp.parseData( navigator.currentPose, [[],[]] ) 
    
    dispWind.render()
    

for i in range(0, int(maxTime/timeStep) ): 
    delay(0.01) 
    
    lEncoderChange = velLwh*timeStep 
    rEncoderChange = velRwh*timeStep  

    navigator.updatePosition( lEncoderChange*(errorMult() + bias), rEncoderChange*errorMult() ) 

    velLwh, velRwh = navigator.desiredTargetMVels(  )
    
    velL.append( velLwh )
    velR.append( velRwh )
    
    x.append( navigator.currentPose.x )
    y.append( navigator.currentPose.y )
    alpha.append( navigator.currentPose.yaw )
    
    """realX.append( actualPose.worldPose.x )
    realY.append( actualPose.worldPose.y )
    realAlpha.append( actualPose.worldPose.yaw )"""

    t.append( i * timeStep )
    
    showChartLive()

plot.figure(1234)
plot.plot( x, y, "b-", label="detec pos" )
plot.plot( realX, realY, "r--", label="real pos" )
plot.plot( TARGET_Xs, TARGET_Ys, "gx"  )
plot.legend() 

plot.figure(12342)
plot.plot( t, realX, "r--", label="real x" )
plot.plot( t, realY, "y--", label="real y" )
#plot.plot( t, realAlpha, "b--", label="real alpha" )
plot.plot( t, x, "r", label="detec x" )
plot.plot( t, y, "y", label="detec y" )
#plot.plot( t, alpha, "b--", label="real alpha" )
plot.legend() 

"""plot.figure(123242) 
plot.plot( t, realAlpha, "b--", label="real alpha" )
plot.plot( t, alpha, "b--", label="detec alpha" )
plot.legend() 

plot.figure(623242) 
plot.plot( t, velL, "b--", label="velL" )
plot.plot( t, velR, "r--", label="velR" )
plot.legend()"""
plot.show()




