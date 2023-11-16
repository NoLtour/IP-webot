# Simulation with ZeroROS and Webots

This repository contains the base code to set a simulation environment using webots,
python and ZeroROS as its middleware.

## Structure
 - The subfolder `webots` contains a world file and a controller written in python to 
simulate a differential drive robot with a 2D lidar.

 - The subfolder `simulation_zeroros` contains the base python code to interact with the
simulated robot.

 - Information is shared via ZeroROS topics. These are the available ones:
    - Topic `/lidar` ([sensor_msgs.LaserScan](https://github.com/miquelmassot/zeroros/blob/main/src/zeroros/messages/sensor_msgs.py#L7)): Each lidar scan is published.
    - Topic `/odom` ([nav_msgs.Odometry](https://github.com/miquelmassot/zeroros/blob/main/src/zeroros/messages/nav_msgs.py#L8)): Simulation ground-truth position.
    - Topic `/cmd_vel` ([geometry_msgs.Twist](https://github.com/miquelmassot/zeroros/blob/main/src/zeroros/messages/geometry_msgs.py#L46)): Input velocities to the robot.

## How to run

### Webots

1. Install webots from [here](https://cyberbotics.com/#download).
2. Open the world file `webots/worlds/simulation_zeroros.wbt` with webots.
3. Run the simulation.

### Python

1. Install python3 and pip3.
2. Open a terminal in the root folder of this repository and install the package: `pip install -U --user .`.
3. Run the python script: `python controller.py`.

You should see the robot moving in the simulation and the lidar data in the terminal.
When stopping the code, two log files will be generated, a console log and a data log containing all the data published in the topics.