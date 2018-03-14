from setuptools import setup, find_packages

import vehicle_detection

setup(
    name='vehicle_detection',
    version=vehicle_detection.__version__,
    packages=find_packages(),
    long_description=open('README.md').read(),
    license=open('LICENSE').read(),
    entry_points={'console_scripts' : ['run_vehicle_detection = vehicle_detection.core:run']}
    )