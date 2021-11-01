import os
import setuptools
from distutils.core import Extension

VERSION = os.environ.get("RISKMODELS_VERSION", "1.0.0")

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='riskmodels',  

     version=VERSION,

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = setuptools.find_namespace_packages(include=['riskmodels.*']),

     description="Extreme value models for applications in energy procurement",

     license = "MIT",

     setup_requires = ['cffi>=1.0.0'],

     install_requires=[
        'pydantic>=1.8.2',
        'emcee>=3.1.0',
        'scipy>=1.7.1',
        'numpy>=1.21.2',
        'matplotlib>=3.4.3',
        'pandas',
        'statsmodels>=0.12',
        'cffi>=1.0.0',
        'tqdm'
    ],

     long_description=long_description,

     long_description_content_type="text/markdown",

     cffi_modules=[
       "riskmodels/c_build/build_timedependence.py:ffibuilder",
       "riskmodels/c_build/build_univarmargins.py:ffibuilder",
       "riskmodels/c_build/build_bivarmargins.py:ffibuilder"],

     url="https://bitbucket.com/nestorsag/phd",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ]

 )