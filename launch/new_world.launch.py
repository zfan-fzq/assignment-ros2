import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    os.environ["TURTLEBOT3_MODEL"] = "waffle_pi"

    pkg_share = get_package_share_directory("assignment3_tb3")
    tb3_share = get_package_share_directory("turtlebot3_gazebo")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    gui = LaunchConfiguration("gui", default="true")
    x_pose = LaunchConfiguration("x_pose", default="-2.0")
    y_pose = LaunchConfiguration("y_pose", default="-1.0")
    linear_speed = LaunchConfiguration("linear_speed", default="0.30")
    slow_speed = LaunchConfiguration("slow_speed", default="0.15")
    turn_speed = LaunchConfiguration("turn_speed", default="0.95")

    world = os.path.join(pkg_share, "worlds", "new_world.world")
    tb3_launch_dir = os.path.join(tb3_share, "launch")

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r -s -v2 ", world], "on_exit_shutdown": "true"}.items(),
    )

    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-g -v2 "}.items(),
        condition=IfCondition(gui),
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
    )

    vision_nav = TimerAction(
        period=7.0,
        actions=[
            Node(
                package="assignment3_tb3",
                executable="vision_nav",
                name="vision_nav",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "linear_speed": linear_speed,
                        "slow_speed": slow_speed,
                        "turn_speed": turn_speed,
                    }
                ],
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("linear_speed", default_value="0.30"),
            DeclareLaunchArgument("slow_speed", default_value="0.15"),
            DeclareLaunchArgument("turn_speed", default_value="0.95"),
            SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3"),
            AppendEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                os.path.join(pkg_share, "models"),
            ),
            AppendEnvironmentVariable(
                "GZ_SIM_RESOURCE_PATH",
                os.path.join(tb3_share, "models"),
            ),
            gz_server,
            gz_client,
            spawn_turtlebot,
            robot_state_publisher,
            vision_nav,
        ]
    )
