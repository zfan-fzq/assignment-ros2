from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tb3_autonomous'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
            # 安装 launch 文件到 share/tb3_autonomous/launch
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # 安装 xacro 文件到 share/tb3_autonomous/xacro
        ('share/' + package_name + '/xacro', glob('xacro/*.xacro')),
        # 安装 world 文件到 share/tb3_autonomous/worlds
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        # 安装 package.xml
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zfan',
    maintainer_email='zfan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'vision_nav = tb3_autonomous.navigation_node:main',
        ],
    },
)
