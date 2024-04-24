
import numpy as np
import matplotlib.pyplot as plt

"""def gaussian_kernel(size, sigma=1):
     
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)"""

def acuteAngle( theta1:np.ndarray, theta2:np.ndarray ):
    theta1 = (np.pi*14 + theta1)%(np.pi*2)
    theta2 = (np.pi*14 + theta2)%(np.pi*2)
    
    return min( abs(theta1-theta2), 2*np.pi-abs(theta1-theta2) )

def rotationMatrix(theta):
    """ Generate a 2x2 rotation matrix for a given angle theta (in radians). """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

def solidCircle(diameter):
    return np.fromfunction(
        lambda x, y: np.where( diameter**2//4 < (x-diameter//2)**2+(y-diameter//2)**2, 1, 0 )
        , (diameter, diameter))

def gaussianKernel(sigma, cutoff=0.02):
    """Generates a Gaussian kernel."""
    diam = 1+2*int(np.sqrt((-2*sigma**2) * np.log(cutoff/sigma)))

    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(diam-1)/2)**2 + (y-(diam-1)/2)**2)/(2*sigma**2)),
        (diam, diam)
    )
    return kernel / np.sum(kernel)
 
def generate1DGuassianDerivative( sigma, cutoff=0.02 ):
    radius = int(np.sqrt((-2*sigma**2) * np.log(cutoff/sigma)))

    x      = np.arange( -radius, radius+1, 1 )
    kernel = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2)/(2*sigma**2))
    
    kernel /= np.abs(kernel)
    kernel[radius] = 0
    
    kernel = kernel/np.sum(kernel)
    
    grad = np.where( x<0, -np.abs(kernel), np.abs(kernel)) #np.gradient( kernel )
    
    dx = np.column_stack( grad )
    dy = np.transpose( dx )
    
    return dx, dy



fplotN = 0
def fancyPlot( inp ):
    global fplotN
    
    #inp = np.where( inp>0, 1, np.where( inp<0, -1, 0 ) )
    
    plt.figure(5000+fplotN)
    fplotN += 1
    y, x = np.meshgrid(np.arange(inp.shape[1]), np.arange(inp.shape[0]))
    plt.imshow( np.ones(inp.shape), cmap="gray", origin="lower", vmin=0, vmax=1 )
    cb = plt.imshow( np.where(inp==0, np.inf, inp), origin="lower" )
    #plt.axis("off")
    #plt.colorbar(cb)
 
def generateAngleArray(diameter): 
    """ generates a 2D array of set diameter  """
    indices = np.arange(diameter)
     
    center = diameter // 2 
    x, y = np.meshgrid(indices, indices)
     
    angle_array = np.arctan2(center - y, center - x)  
    return angle_array

def convolveWithEdgeWrap(input_array, kernel):
    # Calculate the amount of padding needed
    pad_width = kernel.shape[0] // 2
    # Pad the input array
    padded_array = np.pad(input_array, pad_width, mode='wrap')
    # Perform convolution
    convolved = np.convolve(padded_array, kernel, mode='valid')
    return convolved