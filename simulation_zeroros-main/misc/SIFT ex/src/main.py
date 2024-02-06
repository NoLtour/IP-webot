
"""
    Resource doc
     
    1) https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
    2) https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
     
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from skimage.measure import block_reduce

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_kernel2( sigma ):
    """Generates a Gaussian kernel."""
    size = round(np.sqrt(sigma))*2+3
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# At -1 it does thingy, returns [pos], [I]
def findMaxima( inpArray:np.ndarray, threshHold = -1, maskSize = 3 ):
    filterDims = (maskSize,)*inpArray.ndim
    localMaximums = np.where( (inpArray == maximum_filter(inpArray, size=filterDims, mode='constant')) | (inpArray == minimum_filter(inpArray, size=filterDims, mode='constant')) )
    localIntensities = laplace( inpArray )[ localMaximums ]
    
    if ( threshHold == -1 ):
        threshHold = np.max( np.abs(localIntensities) ) * 0.1
        
    mask = np.where( np.abs(localIntensities) > threshHold )
            
    return np.array(localMaximums).transpose()[ mask ], localIntensities[mask]


def rgbConvolve( input_image:np.ndarray, kernal, normaliseChannels=False ):
    """if ( normaliseChannels ):
        tmp1 = convolve2d( input_image.transpose()[0], kernal, mode="same" )
        tmp2 = convolve2d( input_image.transpose()[1], kernal, mode="same" )
        tmp3 = convolve2d( input_image.transpose()[2], kernal, mode="same" )
        
        return (
            tmp1/np.max(tmp1)+
            tmp2/np.max(tmp2)+
            tmp3/np.max(tmp3)
        ).transpose()"""
    if ( normaliseChannels ):
        tmp1 = convolve2d( input_image.transpose()[0], kernal, mode="same" )
        tmp2 = convolve2d( input_image.transpose()[1], kernal, mode="same" )
        tmp3 = convolve2d( input_image.transpose()[2], kernal, mode="same" )
        
        return (
            np.maximum(np.maximum(tmp1, tmp2), tmp3)
        ).transpose()
    else:
        return (
            convolve2d( input_image.transpose()[0], kernal, mode="same" )+
            convolve2d( input_image.transpose()[1], kernal, mode="same" )+
            convolve2d( input_image.transpose()[2], kernal, mode="same" )
        ).transpose()

class SIFT_thing:

    def __init__(this):
        ""

    # TODO find the sampling optimal numbers
    def findDifferenceOfGuassian(this, baseImage, k_sigma=np.sqrt(2), per_octave=4, max_octaves=5 ): 
        oupsDOG = []

        for octave in range(0, max_octaves):
            oupsDOG.append([])

            init_sigma = k_sigma**( -1 + octave*(per_octave+1)/2 )
            
            dsImage = this.downSample( baseImage, 2**(octave) )
            
            prevOutput = rgbConvolve( dsImage, gaussian_kernel2( init_sigma ) ) # TODO, unsure if image 0 is blurred or not?
            
            for n in range(0, per_octave):
                sigma = init_sigma * (k_sigma**(n+1))
                
                cOutput = rgbConvolve( dsImage,  gaussian_kernel2( sigma ) )
                oupsDOG[octave].append( prevOutput - cOutput )
                prevOutput = cOutput
                
                
        return oupsDOG
        
    def tttt(this, oupsDOG):
        
        octavePiles = []
        for octaveLayers in oupsDOG:
            octavePile = np.array( octaveLayers )
            octavePiles.append( octavePile )
            
            peaks = findMaxima( octavePile )
             
             
        
        
        
        return

    # Input format is image, identical to after Image.open(...)
    def downSample(this, inpImage:np.ndarray, scaleFactor:float):
        
        return np.stack( [
            zoom( inpImage[:,:,0], 1/scaleFactor ),
            zoom( inpImage[:,:,1], 1/scaleFactor ),
            zoom( inpImage[:,:,2], 1/scaleFactor ),
            zoom( inpImage[:,:,3], 1/scaleFactor )            
        ], axis=-1 )

# Load the image using PIL (Python Imaging Library) 
image_path = 'C:\\IP-webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-05_16.20.47.png'
image_path = 'C:\\IP-webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-02_16.54.43.png'
image_path = 'C:\\IP-webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\sampleImage.png' 
img = Image.open(image_path)
 
# Convert the image to a numpy array
img_array = np.array(img)

def showSimp():
    plt.imshow( img_array )
    #plt.imshow( img_array.transpose()[0].transpose() )
    plt.figure(542) 
    
    tmp =  (gaussian_kernel( 19, 15 ) ) 
     
    oup = rgbConvolve( img_array, (gaussian_kernel( 19, 15 )), True ) - rgbConvolve( img_array, (gaussian_kernel( 9, 7 )), True )
    plt.imshow( convolve2d( img_array.transpose()[0], tmp, mode="same" ).transpose() )
    plt.imshow( np.abs(oup), cmap='gray' )
    plt.figure(7562)   
    
    #oup = convolve2d( img_array.transpose()[0], (laplace(tmp)), mode="same" ).transpose() - convolve2d( img_array.transpose()[0], (laplace(gaussian_kernel( 11, 7 ) )), mode="same" ).transpose()
    
    oup = rgbConvolve( img_array, laplace(gaussian_kernel( 19, 15 )), True ) - rgbConvolve( img_array, laplace(gaussian_kernel( 9, 7 )), True )
    
    plt.imshow( np.abs(oup), cmap='gray' )

def otherTest():
    ST = SIFT_thing()
    img_array2 = ST.downSample( img_array, 1 )

    # Display the image using matplotlib
    plt.imshow( img_array2 ) 
    oupsDOG = ST.findDifferenceOfGuassian( img_array2 )
    
    ST.tttt( oupsDOG )

    plt.figure(2) 

    plt.imshow( oupsDOG[0][0], cmap='gray' )
    plt.figure(3)  

    plt.imshow( oupsDOG[1][0], cmap='gray' )
    plt.figure(4)   

    plt.imshow( oupsDOG[2][0], cmap='gray' )
    plt.figure(5)   

    plt.imshow( oupsDOG[3][0], cmap='gray' )
    
 
def peaking():
    guassian = np.array([0.05,0.15,0.6,0.15,0.05])
    
    ST = SIFT_thing()
    yVals = np.convolve( ST.downSample( img_array, 6 ).transpose()[0][140], guassian )
    xVals = np.arange( 0, yVals.size, 1 )
    
    plt.plot( yVals, "b-" )
    
    gradVals = np.gradient( yVals )
    peaksXs = np.where( (gradVals[:-1] * gradVals[1:])<0 )[0]
    
    plt.plot( peaksXs, np.zeros(peaksXs.shape), "rx" )
    plt.plot( gradVals, "r--" )  
 
    
    
        
    
    

def peaking2():
    guassian = gaussian_kernel( 3, 0.1 ) 
    """plt.figure(4)
    plt.imshow( img_array  )  
    return"""
    
    ST = SIFT_thing()
    hMap = convolve2d( ST.downSample( img_array, 4 ).transpose()[0].transpose(), guassian, mode="same" )
    
    
    yGrad, xGrad = np.gradient( hMap )  

    # Find local maxima using maximum_filter
    local_maxima, intenisy = findMaxima( hMap )
    
    plt.imshow( hMap, cmap='Reds' )  
    
    for point in local_maxima:
        plt.plot(point[1], point[0], 'bx' )
    
    plt.figure(4)
    plt.imshow( np.abs(laplace( hMap ))  )
    
    plt.figure( 5 )
     
    occ, ppp = np.unique((intenisy), return_counts=True)
    plt.bar(occ, ppp)
    
    """yGradPeaks = convolve2d(np.abs(convolve2d( yGrad, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), mode="same" )), guassian, mode="same" )
    xGradPeaks = convolve2d(np.abs(convolve2d( xGrad, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).transpose(), mode="same" )), guassian, mode="same" )
    
    plt.figure(4)
    plt.imshow( yGradPeaks*xGradPeaks , cmap='Reds')
    plt.figure(5)
    plt.imshow( yGradPeaks , cmap='Blues')
    plt.figure(6)
    plt.imshow( xGradPeaks , cmap='Blues')"""
    
    """plt.figure(4)
    plt.imshow( yGrad )
    plt.figure(5)
    plt.imshow(  convolve2d(np.abs(convolve2d( yGrad, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), mode="same" )), guassian, mode="same" ) )
    
    plt.figure(6)
    plt.imshow( xGrad )
    plt.figure(7)
    plt.imshow(  convolve2d(np.abs(convolve2d( xGrad, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).transpose(), mode="same" )), guassian, mode="same" ) )"""
    
    
    """plt.plot( peaksXs, np.zeros(peaksXs.shape), "rx" )
    plt.plot( gradVals, "r--" )  """
    
    
    
 

#tmp = np.array( img_array.transpose() )

"""plt.imshow(img_array.transpose()[0].transpose(), cmap='Reds')
plt.subplot(211) 
plt.imshow(img_array.transpose()[1].transpose(), cmap='Greens')
plt.subplot(212) 
plt.imshow(img_array.transpose()[2].transpose(), cmap='Blues')"""

#plt.imshow(this.gData, cmap='gray', interpolation='none', origin='lower', extent=[0, xMax, 0, yMax])

#otherTest()
peaking2()

plt.show()
