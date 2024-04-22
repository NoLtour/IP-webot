from GraphLib import GraphSLAM2D
import numpy as np

graph_slam = GraphSLAM2D(verbose=True)

# Add fixed pose ID #0
pose1 = graph_slam.add_fixed_pose()
 
pose2 = graph_slam.add_pose( pose1, np.array(( 1.05, 0, np.deg2rad(80) )) )
  
pose3 = graph_slam.add_pose( pose2, np.array(( 1.05, 0, np.deg2rad(80) )) )
  
pose4 = graph_slam.add_pose( pose3, np.array(( 1.05, 0, np.deg2rad(80) )) )
  
pose5 = graph_slam.add_pose( pose4, np.array(( 1.05, 0, np.deg2rad(80) )) )


#plot_slam2d( graph_slam.optimizer, "3" ).show()

# Optimize
graph_slam.optimize(10, verbose=True)

graph_slam.plot()
# Optimize
#graph_slam.relate_pose( np.array(( 0.95, 0.1, np.deg2rad(90) )), pose4, pose1 )
graph_slam.relate_pose( pose1, pose3, np.array(( 1, 1, np.deg2rad(180) ))  )
graph_slam.relate_pose( pose2, pose4, np.array(( 1, 1, np.deg2rad(180) ))  )
#graph_slam.relate_pose( np.array(( 1, 1, 0 )), rootPose, midP )
graph_slam.optimize(10, verbose=True)

graph_slam.plot()
 
graph_slam.relate_pose( pose5, pose1, np.array(( 0, 0, 0 )) )
#graph_slam.relate_pose( np.array(( 1, 1, 0 )), rootPose, midP )
graph_slam.optimize(10, verbose=True)

for cPose in [pose1,pose2,pose3,pose4,pose5]:
    print( cPose, " pos: ", graph_slam.vertex_pose( cPose ) )

graph_slam.plot()
""