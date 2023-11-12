"""firstController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, DistanceSensor, Motor

if __name__ == "__main__":
    # create the Robot instance.
    robot = Robot()
    
    
    # init motors
    motorL = robot.getMotor("motorL")
    motorR = robot.getMotor("motorR")
    
    motorL.setPosition(float('inf'))
    motorL.setVelocity(0.0)
    motorR.setPosition(float('inf'))
    motorR.setVelocity(0.0)
    
    # get the time step of the current world.
    timestep = 64
    
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
    
    
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        if ( rSens.getValue() != 0 ):
            motorL.setVelocity( 0 )
        else:
            motorL.setVelocity( maxSpeed )
            
        if ( lSens.getValue() != 0 ):
            motorR.setVelocity( 0 )
        else:
            motorR.setVelocity( maxSpeed ) 
        
        
    
    # Enter here exit cleanup code.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    