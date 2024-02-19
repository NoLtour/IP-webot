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

class Test:

    x = 20
    y = 30
    z = None
    f: float = None

    def __init__(this):
        this.x = 21
        this.y = None

tmp = Test()
 

"""plt.imshow( ImageProcessor.gaussian_kernel(7, 2) )
plt.show()"""
