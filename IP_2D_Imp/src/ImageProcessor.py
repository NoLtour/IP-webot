
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
    def estimateFeatures( inpGrid:ProbabilityGrid, estimatedWidth, sharpnessMult=2.5 ):
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
    def extractIntrest( inpGrid:ProbabilityGrid  ):
        """  """
        
        
        
    
    
        
    














