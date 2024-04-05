import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize

# Sample data
np.random.seed(351)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
values = (x*2*np.pi+78)**4 + (y*2*np.pi-78)**1.3 + z*5 -9

# Interpolation
points = np.column_stack((x, y, z))
interpolator = LinearNDInterpolator(points, values)

# Define a function that evaluates the interpolated data
def interpolated_function(coords):
    return interpolator(coords)

# Define initial guess for the minimum
initial_guess = [0.5, 0.5, 0.5]

# Use scipy.optimize.minimize to find the minimum
result = minimize(interpolator, initial_guess)

# The result will contain the minimum point and value
min_point = result.x
min_value = result.fun

print("Minimum point:", min_point)
print("Minimum value:", min_value)

""