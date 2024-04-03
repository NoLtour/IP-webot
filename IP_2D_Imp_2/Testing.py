



import numpy as np

import matplotlib.pyplot as plt
 

def generate_angle_array(n):
    # Generate indices for the array
    indices = np.arange(n)
    
    # Compute the center point
    center = n // 2
    
    # Generate a grid of x and y coordinates
    x, y = np.meshgrid(indices, indices)
    
    # Compute the angle between each point and the center
    angle_array = np.arctan2(center - y, center - x)
    
    return angle_array

# Example: Generate a 5x5 array
n = 105
angle_array = generate_angle_array(n)
print(angle_array)

plt.imshow( angle_array )
plt.show()

""




