import matplotlib.pyplot as plot 

import testing as ControllerMaths

import random


maxTime = 112
timeStep = 0.05
 
wheel_seperation = 0.135 * 2
wheel_diameter   = 0.075*2

navigator = ControllerMaths.Navigator( wheel_diameter, wheel_seperation )
TARGET = [1,1]
navigator.addTarget(TARGET[0], TARGET[1])

actualPose = ControllerMaths.PositionTracker( ControllerMaths.WheelEncoderCalculator( wheel_diameter, wheel_seperation ) )

velLwh = 0
velRwh = 0

x = []
y = []
alpha = []

realX = []
realY = []
realAlpha = []

t = []

errorFrac = 0.0
bias = 0.0

def errorMult():
    return ((random.random()-0.5)*errorFrac + 1 )

for i in range(0, int(maxTime/timeStep) ): 
    lEncoderChange = velLwh*timeStep 
    rEncoderChange = velRwh*timeStep  

    navigator.updatePosition( lEncoderChange*(errorMult() + bias), rEncoderChange*errorMult() )
    actualPose.updateWorldPose( lEncoderChange, rEncoderChange )

    velLwh, velRwh = navigator.desiredTargetMVels(  )
    
    x.append( navigator.posTracker.worldPose.x )
    y.append( navigator.posTracker.worldPose.y )
    alpha.append( navigator.posTracker.worldPose.yaw )
    
    realX.append( actualPose.worldPose.x )
    realY.append( actualPose.worldPose.y )
    realAlpha.append( actualPose.worldPose.yaw )

    t.append( i * timeStep )

plot.figure(1234)
plot.plot( x, y, "b-", label="detec pos" )
plot.plot( realX, realY, "r--", label="real pos" )
plot.plot( TARGET[0], TARGET[1], "gx"  )
plot.legend() 

plot.figure(12342)
plot.plot( t, realX, "r--", label="real x" )
plot.plot( t, realY, "y--", label="real y" )
#plot.plot( t, realAlpha, "b--", label="real alpha" )
plot.plot( t, x, "r", label="detec x" )
plot.plot( t, y, "y", label="detec y" )
#plot.plot( t, alpha, "b--", label="real alpha" )
plot.legend() 

plot.figure(123242) 
plot.plot( t, realAlpha, "b--", label="real alpha" )
plot.plot( t, alpha, "b--", label="detec alpha" )
plot.legend()
plot.show()




