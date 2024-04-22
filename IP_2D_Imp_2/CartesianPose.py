
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
    def zero(   ):
        """Constructs a CartesianPose with values zero"""
        
        return CartesianPose(
            0,
            0,
            0,
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
        
        return this
    
    def asNumpy(this):
        return np.array((this.x, this.y, this.yaw))
    
    def subtractPose(this, inputPose ):
        """Subtracts input pose to this pose, doesn't perform any rotations just add's values"""
        this.x -= inputPose.x
        this.y -= inputPose.y
        this.z -= inputPose.z
        
        this.roll -= inputPose.roll
        this.pitch -= inputPose.pitch
        this.yaw -= inputPose.yaw

        return this
    
    def divide(this, value ):
        """Divides this pose by input pose"""
        this.x /= value
        this.y /= value
        this.z /= value
        
        this.roll /= value
        this.pitch /= value
        this.yaw /= value
    
    def upscale(this, scaleFactor ):
        """Performs an upscale by scaleFactor, only effects position"""
        this.x *= scaleFactor
        this.y *= scaleFactor
        this.z *= scaleFactor 

    def forceInt(this):
        this.x = int( this.x )
        this.y = int( this.y )
        this.z = int( this.z )
          
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