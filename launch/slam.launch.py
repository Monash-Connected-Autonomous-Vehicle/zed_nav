import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('zed_nav')
    
    return LaunchDescription([
        # Convert Velodyne pointcloud to laserscan
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            parameters=[os.path.join(pkg_dir, 'config', 'pointcloud_to_laserscan.yaml')],
            remappings=[
                ('/cloud_in', '/velodyne_voxel_filtered'),
                ('/scan', '/scan_filtered')
            ]
        ),
        
        # SLAM Toolbox node
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[os.path.join(pkg_dir, 'config', 'slam_config.yaml')]
        )
    ])