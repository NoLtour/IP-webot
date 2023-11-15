
from controller import Robot, DistanceSensor, Motor

if __name__ == "__main__":
    # create the Robot instance.
    robot = Robot()
    
    
    # init motors
    motorL = robot.getDevice("left wheel motor")
    motorR = robot.getDevice("right wheel motor")
    
    motorL.setPosition(float('inf'))
    motorL.setVelocity(0.0)
    motorR.setPosition(float('inf'))
    motorR.setVelocity(0.0)
    
    # get the time step of the current world.
    TIME_STEP = 32
    
    # initialize devices
    ps = []
    psNames = [
        'ps0', 'ps1', 'ps2', 'ps3',
        'ps4', 'ps5', 'ps6', 'ps7'
    ] 
    
    for i in range(8):
        ps.append(robot.getDevice(psNames[i]))
        ps[i].enable(TIME_STEP) # sens update freq
         
    rSens = ps[0]
    lSens = ps[7]
    
    maxSpeed = 6
    
    def avSensors(numbs):
        sum = 0
        for i in numbs:
            sensVal = ps[i].getValue()/4096
            
            if ( sensVal < 0.007 ):
                sensVal = 0
            else:
                sensVal = sensVal*5 + 0.05
                
            sum += min(1, sensVal)
            
        return sum/len(numbs)
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(TIME_STEP) != -1: 
        lAv = avSensors( [7, 6, 5, 4] )
        rAv = avSensors( [0, 1, 2, 3] )
    
        motorR.setVelocity( maxSpeed*(1-2*lAv) )
        motorL.setVelocity( maxSpeed*(1-2*rAv) )
             
        print( "it should work right?", lAv, rAv )
        
    
    # Enter here exit cleanup code.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    