import numpy as np
import datetime
from pathlib import Path

from scipy.signal import convolve2d   
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from matplotlib import pyplot as plt 
 

from Navigator import Navigator, CartesianPose
from Mapper import Mapper
from ProbabilityGrid import exportScanFrames, importScanFrames
from ImageProcessor import ImageProcessor 

from time import sleep
import livePlotter as lp
 
class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = 25
 
lpWindow = lp.PlotWindow(5, 15)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
 
class MC_PROP:
    GRID_RESOLUTION = 40
    MAX_FRAMES_MERGE = 7
    MAX_INTER_FRAME_ANGLE =  np.rad2deg(20)

    # Image estimation
    IE_OBJECT_SIZE = 0.25
    IE_SHARPNESS   = 2.6

    # Corner detector
    CORN_PEAK_SEARCH_DIAMETER = 3
    CORN_PEAK_MIN_VALUE       = 0.15/1000
    CORN_DET_KERN             = ImageProcessor.gaussian_kernel(7, 2)
    # ... guassian kernal info


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
 
def test1( inpArray:np.ndarray ):
    plt.figure(20)

    pos = [ 122, 166 ] # frame len()==3 [ 53, 238 ] [ 55, 182 ]

    plt.imshow( inpArray[ pos[1]-7:pos[1]+8, pos[0]-7:pos[0]+8 ] )#
    #plt.plot( pos[0], pos[1], "rx" )

    outputs = extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 12 )

    plt.figure(4)
    plt.plot( outputs[0] )

    outputs = extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 50 )

    plt.figure(5)
    plt.plot( outputs[0] )

    outputs = extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 200 )

    plt.figure(6)
    plt.plot( outputs[0] )

    plt.show()

print("importing test data...")
allScansRaw = importScanFrames( "testRunData" )
print("imported test data")
mapper = Mapper( None, MC_PROP.GRID_RESOLUTION )

prevScan = 0

for cRawScan in allScansRaw:
    mapper.pushScanFrame( cRawScan, MC_PROP.MAX_FRAMES_MERGE, MC_PROP.MAX_INTER_FRAME_ANGLE )
    
    if ( len( mapper.allScans ) > 0 ):
        scan = mapper.allScans[-1]

        if ( np.max( scan.positiveData ) != 0 and scan!=prevScan ):
            pMap = ImageProcessor.estimateFeatures( scan, MC_PROP.IE_OBJECT_SIZE, MC_PROP.IE_SHARPNESS ) - scan.negativeData/2
            
            lambda_1, lambda_2, Rval = ImageProcessor.guassianCornerDist( pMap, MC_PROP.CORN_DET_KERN )
            #rend = lambda_1/(lambda_2+0.00000000000001)
            
            maxPos, vals = ImageProcessor.findMaxima( Rval, MC_PROP.CORN_PEAK_SEARCH_DIAMETER, MC_PROP.CORN_PEAK_MIN_VALUE )  

            #if ( len( mapper.allScans ) > 3 ): test1( pMap )
            
            gridDisp.parseData( pMap, maxPos[:,1], maxPos[:,0]  )
            gridDisp2.parseData( Rval*1000, maxPos[:,1], maxPos[:,0]  )

            prevScan = scan

            lpWindow.render()

            print("pushed ",scan.height*scan.width)
            #sleep(0.1)


""