import numpy as np
import datetime
from pathlib import Path

from scipy.signal import convolve2d
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Twist, Odometry, Vector3
from zeroros.datalogger import DataLogger
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom

from simulation_zeroros.console import Console

from Navigator import Navigator, CartesianPose

from ProbabilityGrid import exportScanFrames, importScanFrames
from ImageProcessor import ImageProcessor 

from matplotlib import pyplot as plt 

import timeit
import numpy as np

vals = np.array([ 
    [0, 3, 6], 
    [1, 4, 7], 
    [2, 5, 8], 
 ])

print( np.sum( vals, axis=0 ) )


plt.imshow( vals, origin="lower" )
plt.show()