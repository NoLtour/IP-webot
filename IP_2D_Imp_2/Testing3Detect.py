import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_circle_image(radius, function, resolution=7, size=7):
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
    return np.sin(theta*2)  + np.cos(theta*6)*0.2  + np.sin(theta*0.2)*0.3  + 0.3  # Example function, adjust as needed

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

    kern = np.array((0.054,0.242,0.398,0.242,0.054))

    intrestMask = (x_coords**2 + y_coords**2) < ((inpImage.shape[0]/2)**2)
    intrestMask[ int(inpImage.shape[1]/2),int(inpImage.shape[0]/2) ] = 0

    angleSet = np.arange( 0, np.pi*2, np.pi*2/oRes )

    angles = np.arctan2( y_coords, x_coords )+np.pi
    angles = (angles*oRes/(2*np.pi) + 0.5 ).astype(int)%oRes

    scaleFactors = np.zeros( (oRes) )
    np.add.at( scaleFactors, angles, intrestMask )

    outputs = np.zeros( (oRes) )
    np.add.at( outputs, angles, inpImage )
    outputs = outputs/scaleFactors

    outputs = convolve_with_edge_wrap( outputs, kern )

    xVecFlat = np.cos( angleSet )
    yVecFlat = np.sin( angleSet )

    avrgAngle = np.arctan2( np.sum(yVecFlat*(outputs**3)), np.sum(xVecFlat*(outputs**3)) )
    avrgAngle = np.pi*2+avrgAngle if avrgAngle<0 else avrgAngle
    avrgIndex = int(0.5+oRes*avrgAngle/(np.pi*2))

    alignedOutputs = np.roll( outputs, -avrgIndex )

    plt.imshow(circle_image, cmap='gray')
    plt.title("Circle with varying perimeter thickness")
    plt.colorbar(label='Thickness') 
    plt.figure(3)
    plt.imshow( x_coords*intrestMask )
    plt.figure(4)
    plt.imshow( y_coords*intrestMask )
    plt.figure(5)
    plt.imshow( angles*intrestMask )
    plt.figure(7)
    plt.plot( outputs, "b" )
    plt.plot( alignedOutputs, "r" )
    plt.plot( 0,0,"ro" )
    plt.plot( avrgIndex,0,"bx" )

    plt.show(block=False)

    ""

findOrientations( circle_image, 24 )

# Display the constructed circle image
plt.imshow(circle_image, cmap='gray')
plt.title("Circle with varying perimeter thickness")
plt.colorbar(label='Thickness')
plt.show()

