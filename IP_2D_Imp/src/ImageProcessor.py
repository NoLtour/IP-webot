
import numpy as np

from ProbabilityGrid import ProbabilityGrid 
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom


class ImageProcessor:
    
    @staticmethod
    def gaussian_kernel(size, sigma=1):
        """Generates a Gaussian kernel."""
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    @staticmethod
    def guassianCornerDist( wallArray:np.ndarray, kernal=gaussian_kernel(9, 4)  ):   
        """ Extracts the eigenvalues """ 

        Iy, Ix = np.gradient( wallArray ) 
        
        IxIx = convolve2d( np.square( Ix ), kernal, mode="same" ) 
        IxIy =  convolve2d( 2 * Ix * Iy , kernal, mode="same" )   
        IyIy =  convolve2d(np.square( Iy ) , kernal, mode="same" )
        
        #return eigvals( np.stack((IxIx, IxIy, IxIy, IyIy), axis=-1).reshape((*IxIx.shape, 2, 2)) )
        #return eigvals(np.array([[IxIx, IxIy], [IxIy, IyIy]]))
        
        ApC = IxIx + IyIy
        sqBAC = np.sqrt( np.square(IxIy) + np.square( IxIx - IyIy ) )
        
        lambda_1 = 0.5 * ( ApC + sqBAC )
        lambda_2 = 0.5 * ( ApC - sqBAC ) 
        
        Rval = lambda_1 * lambda_2 - 0.05*np.square( lambda_1 + lambda_2 ) 
        
        return lambda_1, lambda_2, Rval

    @staticmethod
    def tmp( image: np.ndarray ):
        # Extract zero layer for masking

        # Extract negative layer normalising to hold values of -1
        # Extract positive layer normalising to hold values of 1

        # For each octave: DOG, extract extrema, 

        # 
        ""

    @staticmethod
    def estimateFeatures( inpGrid:ProbabilityGrid, estimatedWidth:float, sharpnessMult=2.5 ):
        """ Uses a model of the environment to partially fill in missing data """
        pixelWidth = estimatedWidth*inpGrid.cellRes
        kern = ImageProcessor.gaussian_kernel( int(pixelWidth)*2+1, pixelWidth )
        kern /= np.max(kern)
        
        oup = np.maximum(convolve2d( inpGrid.positiveData, kern, mode="same" ) - inpGrid.negativeData*pixelWidth*sharpnessMult, 0)
        return np.minimum( oup/np.max(oup)+inpGrid.positiveData, 1 ) 
    
    @staticmethod
    def findMaxima( inpArray:np.ndarray, maskSize = 3, threshHold = -1 ):
        filterDims = (maskSize,)*inpArray.ndim
        localMaximums = np.where( (inpArray == maximum_filter(inpArray, size=filterDims, mode='constant'))  ) #  | (inpArray == minimum_filter(inpArray, size=filterDims, mode='constant')) 
        localIntensities = inpArray[ localMaximums ]
    
        if ( threshHold == -1 ):
            threshHold = np.max( np.abs(localIntensities) ) * 0.1
            
        mask = np.where( np.abs(localIntensities) > threshHold )
                
        return np.array(localMaximums).transpose()[ mask ], localIntensities[mask]

    @staticmethod
    def extractOrientations( inpArray:np.ndarray, pointXs:np.ndarray, pointYs:np.ndarray, radius:int, oRes=12 ):
        """ Produces a histogram with the centre shifted to be inline with the exponentially weighted "centre of mass" """

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
        
        # Iterates through each search window
        for yMin, yMax, xMin, xMax, i in zip(yMins, yMaxs, xMins, xMaxs, range(0, xMins.size)):
            if ( yMax-yMin > 2 and xMax-xMin > 2 ):

                windDy, windDx = np.gradient( inpArray[ yMin:yMax, xMin:xMax ] )
                
                windDy = windDy.flatten()
                windDx = windDx.flatten()
                
                gain = np.sqrt(np.square(windDx) + np.square(windDy))
                
                # extract angles within the specified window after normalising about the primary direction
                angles = np.mod(np.arctan2( windDy, windDx ) + 2*np.pi, 2*np.pi)
                
                # extract occurances of angles after converting them into the specified resolution
                #types, freqs = np.unique( (angles*oRes/(2*np.pi)).astype(int), return_index=True )
                
                nAngles = (angles*oRes/(2*np.pi)).astype(int)
                
                # insert the frequency of occurances into the output array adjusted by gain
                angleDist = np.zeros( oRes )
                np.add.at( angleDist, nAngles, gain )
                
                netX = np.sum( vectorX*(angleDist**2) )
                netY = np.sum( vectorY*(angleDist**2) )
                # Finds the square weighted average vectors angle
                avrgAngle = np.arctan2( netY, netX )
                #avrgAngle = np.sum(( angleDist==np.max(angleDist) ) * np.arctan2( vectorY, vectorX ))
                
                # Shifts the angle distribution array such that the average angle lies at index zero
                angleDist = np.roll( angleDist, - int( avrgAngle*(oRes*0.5/np.pi) - 0.5) )
                
                outputs.append( angleDist )
        
        return np.array(outputs)
 
        
        
    
    
        
    














