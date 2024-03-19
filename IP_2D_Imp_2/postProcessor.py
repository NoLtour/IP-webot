from RawScanFrame import RawScanFrame

import matplotlib.pyplot as plt
import livePlotter as lp
from IPConfig import IPConfig

import numpy as np
from scipy.ndimage import rotate

from Chunk import Chunk

print("importing...")
allScanData = RawScanFrame.importScanFrames( "cleanDataBackup" )
print("imported")


config = IPConfig()

class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = config.GRID_RESOLUTION

lpWindow = lp.PlotWindow(5, 15)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
 

rawStack = []
chunkStack = []

for i in range( 0, len(allScanData) ):
    cRawScan:RawScanFrame = allScanData[i]

    rawStack.append( cRawScan )
    
    midScan = rawStack[int(len(rawStack)/2)]
    if ( abs(cRawScan.pose.yaw - midScan.pose.yaw) > config.MAX_INTER_FRAME_ANGLE or len(rawStack) > config.MAX_FRAMES_MERGE ):
        # Frame merge 
        nChunk = Chunk.initFromRawScans( rawStack[0:-1], config, 0 )
        rawStack = [ rawStack[-1] ] 
        nChunk.constructProbabilityGrid() 
 
        gridDisp2.parseData( nChunk.cachedProbabilityGrid.mapEstimate ) 

        gridDisp.parseData( nChunk.cachedProbabilityGrid.copyRotated( np.deg2rad( 20 ) ).mapEstimate  ) 
        lpWindow.render()

        chunkStack.append( nChunk )
    
    if ( len( chunkStack ) > 5 ):
        parentChunk = Chunk.initEmpty( config )

        parentChunk.addChunks( chunkStack )
        chunkStack = []






""









