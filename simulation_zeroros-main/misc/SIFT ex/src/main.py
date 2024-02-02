
"""
    Resource doc
     
    1) https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
    2) https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b
     
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL (Python Imaging Library)
image_path = 'C:\\IP-webot\\simulation_zeroros-main\\misc\\SIFT ex\\mcImages\\2024-02-02_16.55.03.png'
img = Image.open(image_path)

# Convert the image to a numpy array
img_array = np.array(img)

tmp = np.array( img_array.transpose()[0] )

# Display the image using matplotlib
plt.imshow(img_array)

plt.figure(2) 
plt.imshow(img_array.transpose()[0].transpose(), cmap='Reds')
plt.figure(3) 
plt.imshow(img_array.transpose()[1].transpose(), cmap='Greens')
plt.figure(4) 
plt.imshow(img_array.transpose()[2].transpose(), cmap='Blues')

#plt.imshow(this.gData, cmap='gray', interpolation='none', origin='lower', extent=[0, xMax, 0, yMax])

plt.show()
