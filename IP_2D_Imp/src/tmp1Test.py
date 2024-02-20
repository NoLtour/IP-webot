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

l1 = 20
l2 = 5

hV = np.random.random(l1)

hV2 = np.mean( hV.reshape( -1, int(l1/l2) ), axis=-1 )

plt.plot( hV )

plt.figure(2)
plt.plot( hV2 )

plt.show()

"""plt.imshow( ImageProcessor.gaussian_kernel(7, 2) )
plt.show()"""
