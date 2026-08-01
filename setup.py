from glob import glob
from setuptools import find_packages, setup

package_name = "assignment3_tb3"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
        (
            f"share/{package_name}/models/robocup_2009_spl_field",
            glob("models/robocup_2009_spl_field/*.sdf")
            + glob("models/robocup_2009_spl_field/*.config"),
        ),
        (
            f"share/{package_name}/models/robocup_2009_spl_field/materials/textures",
            glob("models/robocup_2009_spl_field/materials/textures/*"),
        ),
        (
            f"share/{package_name}/models/construction_cone",
            glob("models/construction_cone/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fzq64",
    maintainer_email="student@example.com",
    description="Camera-only TurtleBot3 navigation in a custom Gazebo world for ROS 2 Jazzy.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_nav = assignment3_tb3.vision_nav:main",
        ],
    },
)
