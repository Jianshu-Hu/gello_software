from setuptools import find_packages, setup
import glob

package_name = "franka_lerobot_data_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "pyzmq"],
    zip_safe=True,
    maintainer="Franka Robotics GmbH",
    maintainer_email="support@franka.de",
    description="Streams synchronized robot and camera data for LeRobot recording.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lerobot_data_bridge = franka_lerobot_data_bridge.lerobot_data_bridge:main",
        ],
    },
)
