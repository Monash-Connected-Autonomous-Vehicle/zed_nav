from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Static TF: base_link to velodyne (lidar)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.2', '0', '0.15', '0', '0', '0', 'base_link', 'velodyne']
        ),
        
        # Static TF: base_link to zed_camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.1', '0', '0.1', '0', '0', '0', 'base_link', 'zed_camera']
        ),
        
        # Odometry from ZED (using zed-ros2-wrapper)
        Node(
            package='zed_odom',
            executable='zed_odom',
            parameters=[{'publish_tf': True, 'publish_map_tf': False}],
            remappings=[('odom', 'zed/odom')]
        )
    ])