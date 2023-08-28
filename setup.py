from setuptools import setup, find_packages
import sys

setup(name='jsac',
      packages=[package for package in find_packages()
                if package.startswith('jsac')],
      description='Jax based real-time Reinforcement Learning for Vision-Based Robotics Utilizing Local and Remote Computers',
      author='Fahim Shahriar',
      url='https://github.com/fahimfss/JSAC',
      author_email='fshahri1@ualberta.ca',
      version='1.0.0')