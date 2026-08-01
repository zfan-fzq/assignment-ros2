# Assignment 3 TurtleBot3 Gazebo Package

This package adapts the ROS 2 Humble assignment instructions to the installed Ubuntu 24.04 + ROS 2 Jazzy environment.

It uses Gazebo Sim through `ros_gz_sim`, TurtleBot3 Waffle Pi, a custom soccer-field world with construction cones, and a Python OpenCV node that subscribes to `/camera/image_raw` and publishes `geometry_msgs/msg/TwistStamped` on `/cmd_vel`.

Generative AI note: this starter package was created with assistance from OpenAI Codex. Cite that assistance in the source comments and assignment report as required by the assignment PDF.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select assignment3_tb3
source install/setup.bash
```

## Run

```bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch assignment3_tb3 new_world.launch.py
```

The navigation node intentionally does not subscribe to `/scan`; it only uses the camera image.
