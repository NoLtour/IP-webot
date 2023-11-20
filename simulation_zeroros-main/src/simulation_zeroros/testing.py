import numpy as np

globalLoggs = []

def fixRads( inpRads ):
    """ mod's input between -pi and +pi """
    if ( inpRads < 0 ):
        return np.pi-(np.pi-inpRads)%(np.pi*2)

    return (np.pi+inpRads)%(np.pi*2)-np.pi

class CartesianPose:
    def __init__(this, x, y, z,  roll, pitch, yaw) -> None:
        """stuff

        Args: 
            x (float): x
            y (float): y
            z (float): z
            roll (float): roll
            pitch (float): pitch
            yaw (float): yaw
        """
        
        this.x = x
        this.y = y
        this.z = z
        
        this.roll = roll
        this.pitch = pitch
        this.yaw = yaw
     
    @staticmethod 
    def fromPolar2D( R, theta, z=0 ):
        """Constructs a CartesianPose from a set of polar coordinates"""
        
        return CartesianPose(
            R * np.cos( theta ),
            R * np.sin( theta ),
            z,
            0,
            0,
            0
        )
    
    @staticmethod 
    def fromNPVecSet( locationVector, rotationVector ):
        pass
    
    def getNPPosVec(this):
        """returns this pose's position component as a numpy column vector"""
        return np.matrix( [ [this.x], [this.y], [this.z] ] )
        
    def getNPRotVec(this):
        """returns this pose's rotation component as a numpy column vector"""
        return np.matrix( [ [this.roll], [this.pitch], [this.yaw] ] )
    
    def copy(this):
        return CartesianPose(this.x,this.y,this.z,this.roll,this.pitch,this.yaw)
    
    def addPose(this, inputPose ):
        """Add's input pose to this pose, doesn't perform any rotations just add's values"""
        this.x += inputPose.x
        this.y += inputPose.y
        this.z += inputPose.z
        
        this.roll += inputPose.roll
        this.pitch += inputPose.pitch
        this.yaw += inputPose.yaw
        
        
    def rotatePose(this, roll, pitch, yaw):
        """Rotates this pose using the inputs given"""
        
        cosRoll = np.cos( roll )
        sinRoll = np.sin( roll )
        
        cosPitch = np.cos( pitch )
        sinPitch = np.sin( pitch )
        
        cosYaw = np.cos( yaw )
        sinYaw = np.sin( yaw )
        
        rotMatrix = np.matrix([
            [cosYaw * cosPitch, -sinYaw *cosRoll+ cosYaw*sinPitch*sinRoll, sinYaw*sinRoll + cosYaw*cosRoll*sinPitch],
            [sinYaw*cosPitch, cosYaw*cosRoll + sinPitch*sinYaw*sinRoll, -cosYaw *sinRoll + sinYaw*sinRoll*cosPitch],
            [ -sinPitch , cosPitch*sinRoll , cosPitch*cosRoll ]
        ])
        
        this.x, this.y, this.z = np.matmul(  rotMatrix , this.getNPPosVec()).flat
        this.roll, this.pitch, this.yaw = np.matmul(rotMatrix ,  this.getNPRotVec()).flat
        
          
class WheelEncoderCalculator: 
    def __init__(this, wheelDiameter, wheelSeperation):
        this.wheelDiameter = wheelDiameter
        this.wheelSeperation = wheelSeperation 
        
    def calculateProgression( this, leftRotation, rightRotation ):
        """Returns a CartesianPose for the x, y and yaw result of wheel rotation"""
        
        leftMove = leftRotation * this.wheelDiameter/2
        
        if ( leftRotation == rightRotation ):
            return CartesianPose(
            leftMove,
            0,
            0,
            0,
            0,
            0
        )
        
        rightMove = rightRotation   * this.wheelDiameter/2 
        
        turningCirc = this.wheelSeperation * (  2*rightMove/(rightMove-leftMove) - 1 )
        
        yaw = (this.wheelDiameter/(4*this.wheelSeperation))*(rightRotation-leftRotation) #rightMove/((turningCirc - this.wheelSeperation))
        
        return CartesianPose(
            turningCirc*np.sin( yaw ),
            turningCirc*(1-np.cos( yaw )),
            0,
            0,
            0,
            yaw
        )
    
    def invertProgressionCalc(this, targetAlpha, targetForward):
        """ This uses a linear approximation to create an inverse function, returning wheel angles for inputs (left angle, right angle) """
        K = 2*this.wheelSeperation/this.wheelDiameter
        
        return K*( targetAlpha + targetForward ), K*( - targetAlpha + targetForward )


class PositionTracker:
    def __init__(this, wheelEncoderCalculator) -> None:
        this.wheelCalc = wheelEncoderCalculator
        
        this.worldPose = CartesianPose(0,0,0,0,0,0)
    
    def updateWorldPose(this, leftRotation, rightRotation):
        # Intially in local coordiantes
        movementPose = this.wheelCalc.calculateProgression(leftRotation, rightRotation)
        
        # Rotated into world coordiantes
        movementPose.rotatePose( this.worldPose.roll, this.worldPose.pitch, this.worldPose.yaw )
        
        # Added onto this world pose
        this.worldPose.addPose( movementPose )
        
    
        
class Navigator:
    MAX_TARGET_ANGLE_DEVIATION = np.deg2rad( 4 )
    TARGET_HIT_DISTANCE = 0.005
    
    def __init__(this, wheelDiameter, wheelSeperation):
        this.targetNodes = []
        this.currentTarget = CartesianPose( 0, 0, 0, 0, 0, 0 )
        
        this.wheelEncoderCalculator = WheelEncoderCalculator( wheelDiameter, wheelSeperation)
        this.posTracker = PositionTracker( this.wheelEncoderCalculator )    
        
        this.nextTarget()
    
    def addTarget(this, x, y ):
        """ Adds a new target, only consider's target x and y since movement's constrained to 2D plane """
        this.targetNodes.append(
            CartesianPose( x, y, 0, 0, 0, 0 )
        )
        
    def nextTarget(this):
        if ( len(this.targetNodes) == 0 ):
            this.currentTarget = -1
        
        elif( this.currentTarget == -1 ):
            this.currentTarget = this.targetNodes[0]
            
        else:
            this.targetNodes.pop(0)
            this.currentTarget = -1
            this.nextTarget()
            return
            
        
    def updatePosition(this, leftAngleChange, rightAngleChange):
        """ Updates the robot's position based of wheel rotation, equivilent to X*_k = R_BI F(u_k) """
        this.posTracker.updateWorldPose( leftAngleChange, rightAngleChange )
 
    FWD_R_SPEED = 1
    BWD_R_SPEED = -1
    FORWARD_SPEED = 1

    QD_A_1 = (4*(FWD_R_SPEED-BWD_R_SPEED))
    QD_B_1 = (-(1.5*QD_A_1 + 4*(FORWARD_SPEED-BWD_R_SPEED-QD_A_1/8)))
    QD_C_1 = (-(3*QD_A_1/4 + QD_B_1))
    QD_D_1 = (BWD_R_SPEED)

    QD_A_2 = (-QD_A_1)
    QD_B_2 = (-(1.5*QD_A_2 + 4*(FORWARD_SPEED-FWD_R_SPEED-QD_A_2/8)))
    QD_C_2 = (-(3*QD_A_2/4 + QD_B_2))
    QD_D_2 = (FWD_R_SPEED)
    
    ANGLE_SR_GAIN = 10
    
    def splitWheelVelocity(this, targetAngle):
        speedRatio= max(min(0.5 + Navigator.ANGLE_SR_GAIN*targetAngle/(np.pi*2), 1), 0)

        sr2 = speedRatio*speedRatio;
        sr3 = sr2*speedRatio;

        rightSpeed =  ( Navigator.QD_A_1*sr3 + Navigator.QD_B_1*sr2 + Navigator.QD_C_1*speedRatio + Navigator.QD_D_1 )  
        leftSpeed  =  ( Navigator.QD_A_2*sr3 + Navigator.QD_B_2*sr2 + Navigator.QD_C_2*speedRatio + Navigator.QD_D_2 ) 

        return leftSpeed, rightSpeed
    
    def desiredTargetMVels(this):
        """ Return's desired motor velocity for reaching target """
        
        if ( this.currentTarget == -1 ):
            if ( len(this.targetNodes) == 0 ):
                return 0, 0
            
            this.currentTarget = this.targetNodes[0]
                
        targetDx = this.currentTarget.x - this.posTracker.worldPose.x
        targetDy = this.currentTarget.y - this.posTracker.worldPose.y
        
        if ( abs(targetDx) < Navigator.TARGET_HIT_DISTANCE and abs(targetDy) < Navigator.TARGET_HIT_DISTANCE ):
            this.nextTarget()
            return this.desiredTargetMVels()
        
        targetDYaw = fixRads( np.arctan2( targetDy, targetDx ) - this.posTracker.worldPose.yaw )
        
        #                                                                                         | lazy approximation|
        #rightTarget, leftTarget = this.wheelEncoderCalculator.invertProgressionCalc( targetDYaw, abs(targetDx + targetDy) )

        return this.splitWheelVelocity( targetDYaw )
        


        
    


 








