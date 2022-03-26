from setuptools import setup

setup(
    name='pricer7405',
    version='0.0.1',
    packages=['pricer'],
    author_email='cbtchn@gmail.com',
    entry_points={
      'console_scripts': ['pricer=pricer.bin:cli']
    }
)