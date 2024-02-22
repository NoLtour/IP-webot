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

# Example arrays of vectors
array1 = np.array([[1, 2], [3, 4]])  # Example array 1
array2 = np.array([[5, 6], [7, 8], [9, 10]])  # Example array 2

# Get the dimensions of the input arrays
m, n = array1.shape
p, q = array2.shape

# Create 2D arrays repeating the vectors along the x and y axes
repeated_array1 = np.repeat(array1, p, axis=0)
repeated_array2 = np.repeat(array2, n, axis=1)

# Sum the two arrays element-wise to get the final array containing all possible vector combinations
final_array = repeated_array1 + repeated_array2

print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)
print("\nRepeated Array 1:")
print(repeated_array1)
print("\nRepeated Array 2:")
print(repeated_array2)
print("\nFinal Array:")
print(final_array)