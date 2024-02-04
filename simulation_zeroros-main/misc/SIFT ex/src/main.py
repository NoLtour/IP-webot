
"""
    Resource doc
     
    1) https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
    2) https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
     
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)



class SIFT_thing:

    def __init__(this):
        ""

    # TODO find the sampling optimal numbers
    def generateScales(this, baseImage, k_sigma=np.sqrt(2), per_octave=5, max_octaves=2 ):
        
        outputs = {}

        for octave in range(1, 1+max_octaves):
            outputs[octave] = []

            dsImage = this.downSample( baseImage, octave )

            for n in range(0, per_octave):
                sigma = k_sigma * (2**n)

                kernal = gaussian_kernel( round(1+sigma/1.5), sigma )

                outputs[octave].append( convolve2d( dsImage,  kernal, mode="same" ) )
        

    # Input format is image, identical to after Image.open(...)
    def downSample(this, inpImage:np.ndarray, scaleFactor:float):
        imageHeight = inpImage.shape[0]
        imageWidth  = inpImage.shape[1]
        
        sampleHeight = (scaleFactor * np.arange( 0, int(imageHeight/scaleFactor), 1 )).astype(int)
        sampleWidth  = (scaleFactor * np.arange( 0, int(imageWidth/scaleFactor ), 1  )).astype(int)  

        return inpImage[np.ix_(sampleHeight, sampleWidth)]

# Load the image using PIL (Python Imaging Library)
image_path = 'C:\\IP webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-02_16.55.03.png'
img = Image.open(image_path)

# Convert the image to a numpy array
img_array = np.array(img)

#tmp = np.array( img_array.transpose() )

ST = SIFT_thing()

# Display the image using matplotlib
plt.imshow( ST.downSample( img_array, 10 ) )

plt.figure(2) 

plt.imshow( ST.downSample( img_array, 20 ) )
 
"""plt.imshow(img_array.transpose()[0].transpose(), cmap='Reds')
plt.subplot(211) 
plt.imshow(img_array.transpose()[1].transpose(), cmap='Greens')
plt.subplot(212) 
plt.imshow(img_array.transpose()[2].transpose(), cmap='Blues')"""

#plt.imshow(this.gData, cmap='gray', interpolation='none', origin='lower', extent=[0, xMax, 0, yMax])

plt.show()
