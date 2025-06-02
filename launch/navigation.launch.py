from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                'launch', 'tf_setup.launch.py'
            )])
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                'launch', 'perception.launch.py'
            )])
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                '/opt/ros/humble/share/nav2_bringup/launch/navigation_launch.py'
            ]),
            launch_arguments={
                'params_file': 'config/nav2_params.yaml',
                'use_sim_time': 'false'
            }.items()
        )
    ])