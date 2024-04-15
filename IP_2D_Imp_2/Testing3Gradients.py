import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def generate_circle_image(radius, function, resolution=13, size=13):
    # Generate a 2D array to represent the image
    image = np.zeros((size, size))

    # Generate thickness values for the circle's perimeter using the provided function
    theta = np.linspace(0, 2*np.pi, resolution)
    x_circle = radius * np.cos(theta) + resolution // 2
    y_circle = radius * np.sin(theta) + resolution // 2
    thickness_values = function(theta)

    thickness_values = np.roll(thickness_values, int(0.5 * len(thickness_values)))

    # Iterate over the perimeter of the circle and draw filled circles with varying radii
    for i in range(resolution):
        x = int(round(x_circle[i]))
        y = int(round(y_circle[i]))
        thickness = int(round(thickness_values[i]))
        cv2.circle(image, (x, y), max(0,thickness), 1, -1)

    return image

# Define a function to generate thickness values for the circle's perimeter
def thickness_function(theta):
    return np.sin(theta*2)  + np.cos(theta*6)*1.6  + np.sin(theta*0.2)*0.8  + 0.9  # Example function, adjust as needed

# Define circle radius
radius = 2

# Generate circle image using the provided function
circle_image = generate_circle_image(radius, thickness_function)

def convolve_with_edge_wrap(input_array, kernel):
    # Calculate the amount of padding needed
    pad_width = kernel.shape[0] // 2
    # Pad the input array
    padded_array = np.pad(input_array, pad_width, mode='wrap')
    # Perform convolution
    convolved = np.convolve(padded_array, kernel, mode='valid')
    return convolved

def findOrientations( inpImage:np.ndarray, oRes:int ):
    x_coords, y_coords = np.meshgrid(np.arange(inpImage.shape[1]), np.arange(inpImage.shape[0]))

    x_coords = x_coords - (inpImage.shape[1]-1)/2
    y_coords = y_coords - (inpImage.shape[0]-1)/2
 
    gaussian_array = np.array([
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
        [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
        [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]
    ])
    
    kern = np.array((0.054,0.242,0.398,0.242,0.054))

    intrestMask = (x_coords**2 + y_coords**2) < ((inpImage.shape[0]/2)**2)
    intrestMask[ int(inpImage.shape[1]/2),int(inpImage.shape[0]/2) ] = 0 

    dy, dx = np.gradient( convolve2d( inpImage, gaussian_array, mode="same" ) )
 
    magnitudes = np.sqrt(dy**2 + dx**2) 

    angles = np.mod(np.arctan2( dy, dx ) + 2*np.pi, 2*np.pi) 
    nAngles = (angles*oRes/(2*np.pi)).astype(int)

    outputs = np.zeros( (oRes) )
    np.add.at( outputs, nAngles, magnitudes )  
    outputs = convolve_with_edge_wrap( outputs, kern )

    avrgAngle = np.arctan2( np.sum(dy**3), np.sum(dx**3) )
    avrgAngle = np.pi*2+avrgAngle if avrgAngle<0 else avrgAngle
    avrgIndex = int(0.5+oRes*avrgAngle/(np.pi*2))

    alignedOutputs = np.roll( outputs, -avrgIndex )

    plt.figure(4)
    plt.imshow(convolve2d( inpImage, gaussian_array, mode="same" ), cmap='gray')
    plt.title("Circle with varying perimeter thickness")
    plt.colorbar(label='Thickness')   
    plt.figure(8)
    plt.plot( outputs, "b" )
    plt.plot( alignedOutputs, "r" )
    plt.plot( 0,0,"ro" )
    plt.plot( avrgIndex,0,"bx" )
    plt.figure(9)
    plt.imshow(dy )
    plt.figure(10)
    plt.imshow(dx )
    plt.figure(11)
    plt.imshow(angles )

    plt.show(block=False)

    ""

findOrientations( circle_image, 24 )

# Display the constructed circle image
plt.imshow(circle_image, cmap='gray')
plt.title("Circle with varying perimeter thickness")
plt.colorbar(label='Thickness')
plt.show()

