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

import PPLib as pl

lpWindow = pl.lpWindow
gridDisp      = pl.gridDisp
gridDisp2     = pl.gridDisp2
config = pl.config

print("importing...")
#allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST2_rawInpData-2" )
allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST1_rawInpData" )
print("imported")

# Noise step
np.random.seed(3115)
for i in range(0,len(allScanData)):
    cScan = allScanData[i]
    
    cScan.index = i 
    ""

config = IPConfig() 

 
"""
    processedData3 -> chunks test1 high quality
    
"""

#pl.autoTunePlot( allScanData, 10 )
pl.meanFeaturelessAutoTune( allScanData, 5 )

pl.featurelessFullTest( pl.getChunk( allScanData, 1 ) )
 

""





