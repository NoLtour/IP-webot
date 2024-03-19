from scipy.ndimage import rotate
from RawScanFrame import RawScanFrame 
import matplotlib.pyplot as plt
from ProbabilityGrid import ProbabilityGrid
import numpy as np

size = 15

# Create an empty image
image = np.zeros((size, size))

# Define the vertices of the triangle
vertices = np.array([[1, 1], [size-2, 1], [size//2, size-2]])

# Define the lines forming the edges of the triangle
lines = [[vertices[i], vertices[(i+1)%3]] for i in range(3)]

# Define a function to draw a line
def draw_line(image, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        image[y1, x1] = 1
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

# Draw the boundary lines of the triangle
for line in lines:
    draw_line(image, line[0], line[1])

image[int(size/2),int(size/2)] = 1

# Display the image
plt.imshow(image, cmap='gray', origin='lower') 
plt.figure(2)

rImage = np.maximum( rotate( image, 10, mode="constant", cval=0, reshape=True ), -90)

plt.imshow(rImage, cmap='gray', origin='lower')
plt.show()





