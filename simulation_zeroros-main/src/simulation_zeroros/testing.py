import numpy as np

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
        rightMove = rightRotation   * this.wheelDiameter/2 
        
        turningCirc = this.wheelSeperation * (  2*leftMove/(leftMove-rightMove) - 1 )
        
        yaw = rightMove/((turningCirc - this.wheelSeperation))
        
        return CartesianPose(
            turningCirc*np.sin( yaw ),
            turningCirc*(1-np.cos( yaw )),
            0,
            0,
            0,
            yaw
        )
    


class PositionTracker:
    def __init__(this) -> None:
        this.wheelCalc = WheelEncoderCalculator(9.92, 17.8)
        
        this.worldPose = CartesianPose(0,0,0,0,0,0)
    
    def updateWorldPose(this, leftRotation, rightRotation):
        # Intially in local coordiantes
        movementPose = this.wheelCalc.calculateProgression(leftRotation, rightRotation)
        
        # Rotated into world coordiantes
        movementPose.rotatePose( this.worldPose.roll, this.worldPose.pitch, this.worldPose.yaw )
        
        # Added onto this world pose
        this.worldPose.addPose( movementPose )
        
        
        

 
 
pt = PositionTracker()

pt.updateWorldPose(4, 6)

print("")

pt.updateWorldPose(4, 6)

print("")






