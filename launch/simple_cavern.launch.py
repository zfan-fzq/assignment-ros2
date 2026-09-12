import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    os.environ["TURTLEBOT3_MODEL"] = "waffle_pi"
    pkg_share = get_package_share_directory("assignment3_tb3")
    tb3_share = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros_share = get_package_share_directory("gazebo_ros")
    world = os.path.join(pkg_share, "worlds", "simple_cavern.world")

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world}.items(),
    )
    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, "launch", "gzclient.launch.py")),
    )
    robot_state = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_share, "launch", "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(tb3_share, "launch", "spawn_turtlebot3.launch.py")),
        launch_arguments={"x_pose": "-12.0", "y_pose": "0.0"}.items(),
    )
    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
        SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
        SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3"),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", os.path.join(pkg_share, "models")),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", os.path.join(tb3_share, "models")),
        gz_server, gz_client, robot_state, spawn,
    ])
