from setuptools import setup, find_packages

setup(
    name='lane_detection',
    version='1.0',
    packages=find_packages(include=['lane_detection', 'lane_detection.*']),
    install_requires=['opencv-python==4.8.1.78'],
)
