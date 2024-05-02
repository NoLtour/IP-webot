from RawScanFrame import RawScanFrame

import matplotlib.pyplot as plt
import livePlotter as lp
from IPConfig import IPConfig

import numpy as np
from scipy.ndimage import rotate
from itertools import product

from Chunk import Chunk
 
from scipy.interpolate import RBFInterpolator 
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy import stats
import random

from CommonLib import fancyPlot
from CartesianPose import CartesianPose

import PPLib as pl

lpWindow = pl.lpWindow
gridDisp      = pl.gridDisp
gridDisp2     = pl.gridDisp2
config = pl.config

print("importing...")
#allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST2_rawInpData-2" )
allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST1_LinearMatchedData-4" )
print("imported")

#pl.plotPathError( allScanData )
# Noise step
np.random.seed(3115)


if (True):
    allPoseChanges = [ (0,0) ]
    cccc = []
        
    cumulativeError = CartesianPose.zero()
    for i in range(0,len(allScanData)): 
        cScan:RawScanFrame = allScanData[i]
        
        if (i!=0):
            pScan:RawScanFrame = allScanData[i-1]
            
            interFrameChange = cScan.pose.copy()
            interFrameChange.subtractPose( pScan.pose ) 
        
            sepLength = np.sqrt( interFrameChange.x**2 + interFrameChange.y**2 )
            beta = np.arctan2( interFrameChange.y, interFrameChange.x )
            newBeta = beta - cScan.pose.yaw
            localOffset = np.array((sepLength*np.cos( newBeta ), sepLength*np.sin( newBeta ), interFrameChange.yaw ))
            
            noiseScale = 0.4
            staticDriftAngle = 0.002
            
            scaleFactor = 1.5
            
            allPoseChanges.append( (scaleFactor*(sepLength*(1+(np.random.random()-0.5)*noiseScale*2) ), scaleFactor*(staticDriftAngle+interFrameChange.yaw)*(1+(np.random.random()-0.5)*noiseScale*2)) )
            
            cccc.append( interFrameChange.x**2 + interFrameChange.y**2 )
            ""
        
        cScan.index = i 
    
    
    cccc2 = []
    
    tXp = 0
    tYp = 0
    tAp = 0
    
    for i in range(0,len(allScanData)): 
        cScan:RawScanFrame = allScanData[i]
        
        sepChange, angChange = allPoseChanges[i]
        
        tXp += sepChange*np.cos( tAp )
        tYp += sepChange*np.sin( tAp )
        tAp += angChange
        
        cScan.pose = CartesianPose( tXp, tYp, 0,0,0, tAp )
        
        cScan.index = i 

    
    # plt.figure(5)
    # plt.plot( cccc2 )
    # plt.show(block=False)
    ""
    
    
config = IPConfig() 

 
"""
    processedData3 -> chunks test1 high quality
    
"""


#pl.descriptorTest( pl.getChunk( allScanData, 250 ), pl.getChunk( allScanData, 280 ) )
pl.descriptorTestExt( pl.getChunk( allScanData, 0 ) )
#pl.autoTunePlot( allScanData, 10 )
pl.meanFeaturelessAutoTune( allScanData, 5 )

pl.featurelessFullTest( pl.getChunk( allScanData, 1 ) )
 

""





