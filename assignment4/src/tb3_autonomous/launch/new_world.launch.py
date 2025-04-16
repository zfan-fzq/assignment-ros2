import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ——————————————————————————————————————————————————————
    # 1) Make sure TURTLEBOT3_MODEL is set (burger, waffle, etc.)
    # ——————————————————————————————————————————————————————
    set_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value='waffle'   # ← change to "burger" or "waffle_pi" as needed
    )

    # ——————————————————————————————————————————————————————
    # 2) Package directories & launch‐time configs
    # ——————————————————————————————————————————————————————
    pkg_tb3_auton = get_package_share_directory('tb3_autonomous')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gz    = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'launch'
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose       = LaunchConfiguration('x_pose',      default='0.0')
    y_pose       = LaunchConfiguration('y_pose',      default='2.0')

    # your custom world with markers at the corners
    world = os.path.join(pkg_tb3_auton, 'worlds', 'marker_world.world')

    # ——————————————————————————————————————————————————————
    # 3) Gazebo server & client
    # ——————————————————————————————————————————————————————
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # ——————————————————————————————————————————————————————
    # 4) Robot state publisher (URDF → /tf)
    # ——————————————————————————————————————————————————————
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gz, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # ——————————————————————————————————————————————————————
    # 5) Delay the TB3 spawn until Gazebo is ready
    #    (so /spawn_entity service exists)
    # ——————————————————————————————————————————————————————
    spawn_tb3 = TimerAction(
        period=3.0,  # seconds
        actions=[ IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_tb3_gz, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': x_pose,
                'y_pose': y_pose
            }.items()
        ) ]
    )

    # ——————————————————————————————————————————————————————
    # 6) Static map→odom (identity, until SLAM overwrites it)
    # ——————————————————————————————————————————————————————
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        output='screen',
        arguments=['0','0','0','0','0','0','map','odom']
    )

    # ——————————————————————————————————————————————————————
    # 7) slam_toolbox (after a bit of time for Gazebo to settle)
    # ——————————————————————————————————————————————————————
    slam_toolbox = TimerAction(
        period=4.0,
        actions=[ Node(
            package='slam_toolbox',
            executable='sync_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ) ]
    )

    # ——————————————————————————————————————————————————————
    # 8) Your vision_nav (ArucoNavigator) node
    # ——————————————————————————————————————————————————————
    vision_nav = TimerAction(
        period=5.0,
        actions=[ Node(
            package='tb3_autonomous',
            executable='vision_nav',
            name='tb3_vision_navigator',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ) ]
    )

    # ——————————————————————————————————————————————————————
    # 9) Assemble launch description
    # ——————————————————————————————————————————————————————
    ld = LaunchDescription([
        set_model,
        gzserver,
        gzclient,
        rsp,
        spawn_tb3,
        static_map_to_odom,
        slam_toolbox,
        vision_nav,
    ])
    return ld
