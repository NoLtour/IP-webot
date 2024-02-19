
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
from scipy.linalg import eigvals 

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

def guassianCornerDist( inpArray:np.ndarray, kernal=gaussian_kernel(7, 3)  ):
    Iy, Ix = np.gradient( inpArray ) 
    
    IxIx = convolve2d( np.square( Ix ), kernal, mode="same" )
    IxIy = convolve2d( 2 * Ix * Iy, kernal, mode="same" )
    IyIy = convolve2d( np.square( Iy ), kernal, mode="same" )
      
    #return eigvals( np.stack((IxIx, IxIy, IxIy, IyIy), axis=-1).reshape((*IxIx.shape, 2, 2)) )
    #return eigvals(np.array([[IxIx, IxIy], [IxIy, IyIy]]))
    
    ApC = IxIx + IyIy
    sqBAC = np.sqrt( np.square(IxIy) + np.square( IxIx - IyIy ) )
    
    lambda_1 = 0.5 * ( ApC + sqBAC )
    lambda_2 = 0.5 * ( ApC - sqBAC )
    
    Rval = lambda_1 * lambda_2 - 0.05*np.square( lambda_1 + lambda_2 )
    
    return Rval, lambda_1, lambda_2
 
def gcsImage( inpArray:np.ndarray, K=10, kernal=gaussian_kernel(7, 3) ):
    outputs = np.zeros( inpArray[:,:,0].shape )
    
    for i in range(0, 3):
        l1, l2 = guassianCornerDist( inpArray[:,:,i], kernal=kernal )
        outputs = np.logical_or((outputs) , (l1/l2>K))
        
    return outputs
    
    

# Produces a histogram with the centre shifted to be inline with the exponentially weighted "centre of mass"
def extractOrientations( inpArray:np.ndarray, pointXs:np.ndarray, pointYs:np.ndarray, radius:int, oRes=22 ):
    # Fix ranges to fit
    yMins = np.maximum( pointYs-radius, 0 )  
    yMaxs = np.minimum( pointYs+radius, inpArray.shape[0]-1 )
    xMins = np.maximum( pointXs-radius, 0 )  
    xMaxs = np.minimum( pointXs+radius, inpArray.shape[1]-1 )
    
    outputs = []
    #guassian = gaussian_kernel2( radius*2 + 1 )
    
    angleArrange = np.arange( 0, oRes, 1 )*(2*np.pi/oRes)
    
    vectorX = np.cos( angleArrange )
    vectorY = np.sin( angleArrange )
    
    for yMin, yMax, xMin, xMax, i in zip(yMins, yMaxs, xMins, xMaxs, range(0, xMins.size)):
        windDy, windDx = np.gradient( inpArray[ yMin:yMax, xMin:xMax ] )
        
        windDy = windDy.flatten()
        windDx = windDx.flatten()
        
        gain = np.sqrt(np.square(windDx) + np.square(windDy)) #* guassian
          
        # extract angles within the specified window after normalising about the primary direction
        angles = np.mod(np.arctan2( windDy, windDx ) + 2*np.pi, 2*np.pi)
         
        # extract occurances of angles after converting them into the specified resolution
        #types, freqs = np.unique( (angles*oRes/(2*np.pi)).astype(int), return_index=True )
        
        nAngles = (angles*oRes/(2*np.pi)).astype(int)
        
        # insert the frequency of occurances into the output array adjusted by gain
        angleDist = np.zeros( oRes )
        np.add.at( angleDist, nAngles, gain )
        
        netX = np.sum( vectorX*angleDist )
        netY = np.sum( vectorY*angleDist )
        
        avrgAngle = np.arctan2( netY**3, netX**3 )
        angleDist = np.roll( angleDist, - int( avrgAngle*(oRes*0.5/np.pi)) )
        """expAngleDist = ( angleDist )**4
        centralPoint = np.sum(expAngleDist * np.arange(0, oRes))/np.sum(expAngleDist)
        centralPoint = int( centralPoint + 0.5 ) FIX TO EITHER USE VECTORS OR MAX
        angleDist = np.roll( angleDist, -centralPoint )"""
        
        outputs.append( angleDist )
    
    return np.array(outputs)
    
    

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
        downIMG = []

        for octave in range(0, max_octaves):
            oupsDOG.append([])

            init_sigma = k_sigma**( -1 + octave*(per_octave+1)/2 )
            
            dsImage = this.downSample( baseImage, 2**(octave) )
            downIMG.append( dsImage )
            
            prevOutput = rgbConvolve( dsImage, gaussian_kernel2( init_sigma ) ) # TODO, unsure if image 0 is blurred or not?
            
            for n in range(0, per_octave):
                sigma = init_sigma * (k_sigma**(n+1))
                
                cOutput = rgbConvolve( dsImage,  gaussian_kernel2( sigma ) )
                oupsDOG[octave].append( prevOutput - cOutput )
                prevOutput = cOutput
                
                
        return oupsDOG, downIMG
    
    def extractFeatures(this, baseImage, k_sigma=np.sqrt(2), per_octave=4, max_octaves=5):
        
        oupsDOG, downIMGs = this.findDifferenceOfGuassian( baseImage, k_sigma, per_octave, max_octaves )
        
        features = []
        
        for octave, downIMG in zip(oupsDOG, downIMGs):
            scale = baseImage.shape[0]/downIMG.shape[0]
            
            rDiscriminator = gcsImage( baseImage )
            
            fLay = np.zeros((0, 2))
            
            for cDOG in octave:
                maximaPos, maximaVal = findMaxima( cDOG, -1, 9 )
                
                appliedDiscriminator = rDiscriminator[ maximaPos[:, 0], maximaPos[:, 1] ]
                
                layerfeatures = maximaPos[ appliedDiscriminator ]
                
                fLay = np.append( fLay, layerfeatures*scale, axis=0 )
            features.append( fLay )
                
                
                
                
        
        return features
        
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
image_path = 'C:\\IP webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-05_16.20.47.png'
image_path = 'C:\\IP webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\CaptureBB.png'
#image_path = 'C:\\IP-webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-02_16.56.42.png'
 
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

def circlePlot( ax, points, size ):
    for i in range(len(points)):
        center = (points[i][1], points[i][0])  # Matplotlib uses (y, x) instead of (x, y)
        circle = plt.Circle(center, size, color='r', fill=False)  # You can customize the color and other properties
        ax.add_patch(circle)

def otherTest():
    ST = SIFT_thing()
    img_array2 = ST.downSample( img_array, 4.5 )

    
    fig, ax = plt.subplots()
    
    # Display the image using matplotlib
    plt.imshow( img_array2 ) 
    features = ST.extractFeatures( img_array2 )
    
    circlePlot( ax, features[0], 4 )
    circlePlot( ax, features[1], 8 )
    circlePlot( ax, features[2], 16 )
    circlePlot( ax, features[3], 32 )
    circlePlot( ax, features[4], 64 )
    
    
    """plt.plot(features[0][:,1], features[0][:,0], 'bx' )
    plt.plot(features[1][:,1], features[1][:,0], 'rx' )
    plt.plot(features[2][:,1], features[2][:,0], 'gx' )
    plt.plot(features[3][:,1], features[3][:,0], 'yx' )"""
    
 
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
 
   
def corning(): 
    ST = SIFT_thing() 
     
    lLLL = np.array(guassianCornerDist( img_array[:,:,0] )) + np.array(guassianCornerDist( img_array[:,:,1] )) + np.array(guassianCornerDist( img_array[:,:,2] ))
    l1 = lLLL[0]
    l2 = lLLL[1]
    
    Rval = l1/l2
    
    Rval = Rval>10
    Rval = gcsImage( img_array )
    
    #Rval = guassianCornerDist( img_array.transpose()[0].transpose() )[0] + guassianCornerDist( img_array.transpose()[1].transpose() )[0] + guassianCornerDist( img_array.transpose()[2].transpose() )[0]
    
    #localMaximums = np.where( (Rval == maximum_filter(Rval, size=(5,5), mode='constant')) & (Rval > np.max(Rval)*0.1) )
    
    plt.imshow( img_array, cmap="gray" )
    #plt.plot( localMaximums[1], localMaximums[0], "rx" )
    
    plt.figure(6)
    plt.imshow( Rval  )
    plt.figure(3)
    plt.imshow( l1, cmap="Reds" )
    plt.figure(4)
    plt.imshow( l2, cmap="Blues" ) 
    """v, u = np.gradient( GSImg )
    
    plt.figure(3)
    plt.imshow( l1, cmap="Reds" )
    plt.figure(4)
    plt.imshow( l2, cmap="Blues" ) 
    plt.figure(6)
    plt.imshow( Rval ) """
   
def angling():   
    ST = SIFT_thing()
    GSImg = ST.downSample( img_array, 8 ).transpose()[0].transpose() 
    
    outputs = extractOrientations( GSImg, np.array([203, 40]).astype(int), np.array([38, 60]).astype(int), 3 )
    
    plt.imshow( GSImg )
    plt.figure(4)
    plt.plot( outputs[0] )


def peaking2():
    guassian = gaussian_kernel( 3, 0.1 ) 
    """plt.figure(4)
    plt.imshow( img_array  )  
    return"""
    
    ST = SIFT_thing()
    hMap = convolve2d( ST.downSample( img_array, 2 ).transpose()[0].transpose(), guassian, mode="same" ) 
    yGrad, xGrad = np.gradient( hMap )  

    # Find local maxima using maximum_filter
    local_maxima, intenisy = findMaxima( hMap )
    
    # eigenvalues
    Rval, l1, l2 = guassianCornerDist( hMap, gaussian_kernel(9, 5) )
    
    plt.imshow( hMap, cmap='Reds' )  
    
    lowRFilter = Rval[local_maxima[:,0],local_maxima[:,1]]>(333)
    local_maxima = local_maxima[lowRFilter]
    
    for point in local_maxima:
        plt.plot(point[1], point[0], 'bx' )
    
    plt.figure(4)
    #plt.imshow( np.abs(laplace( hMap ))  )
    plt.imshow( Rval  )
    
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

#corning()
#otherTest()
peaking2()

plt.show()
