
 
import numpy as np
import matplotlib.pyplot as plt
from simpleicp import PointCloud, SimpleICP

POINTS = 16

mainCloudPoints = np.random.random( (POINTS, 2) )

mainCloudPoints = mainCloudPoints[np.argsort(mainCloudPoints[:, 0])]

theta = np.pi / 4  # Rotate by 45 degrees

# Define the 2x2 rotation matrix
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
translation = np.array( [ np.random.random(), np.random.random() ] )

# Apply the rotation matrix to the points
transformedPoints = np.dot(mainCloudPoints, rotation_matrix) + translation

mainCloudPoints = np.array([mainCloudPoints.transpose()[0], mainCloudPoints.transpose()[1],  np.ones( mainCloudPoints.shape[0] ) ] )
transformedPoints = np.array([transformedPoints.transpose()[0], transformedPoints.transpose()[1],  np.ones( transformedPoints.shape[0] ) ] )

pc_fix = PointCloud(mainCloudPoints.transpose(), columns=["x", "y", "z"])
pc_mov = PointCloud(transformedPoints.transpose(), columns=["x", "y", "z"])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

plt.figure(34) 
plt.plot( mainCloudPoints[0], mainCloudPoints[1], "bx--" )
plt.plot( transformedPoints[0], transformedPoints[1], "rx--" )

plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the plot
plt.show()