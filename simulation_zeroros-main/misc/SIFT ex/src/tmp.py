

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from skimage.measure import block_reduce
from scipy.linalg import eigvals



y = np.arange( 0, 30, 0.1 )
x = y.copy()

plt.plot( x, y, "b-" )
plt.plot( x, np.mod(y + 2* np.pi, 2* np.pi), "r--" )


plt.show()











