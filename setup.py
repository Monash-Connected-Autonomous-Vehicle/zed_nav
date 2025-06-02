# zed_nav/setup.py
from setuptools import setup, find_packages
import os
from glob import glob

package_name = "zed_nav"

setup(
    name=package_name,
    version="0.1.0",
    # Pick up zed_nav and every sub-package that has an __init__.py
    packages=find_packages(include=[package_name, f"{package_name}.*"],
                           exclude=["test"]),
    # No standalone modules outside the package directory
    py_modules=[],

    data_files=[
        # make ROS 2 discover the package
        ("share/ament_index/resource_index/packages",
         [os.path.join("resource", package_name)]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # (optional) install launch files if you add them later
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.py")),
    ],

    # PyPI-only dependencies; ROS packages go into package.xml
    install_requires=[
        "setuptools",
        "opencv-python",
        "numpy",
    ],
    zip_safe=True,
    maintainer="tao",
    maintainer_email="yide.tao@monash.edu",
    description="ZED-2i navigation with lane detection",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_commander = zed_nav.camera_commander:main",
        ],
    },
)
