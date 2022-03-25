import os
from pathlib import Path
import re
import setuptools
from distutils.core import Extension
from setuptools import setup, find_namespace_packages, Extension

PROJECT_NAME = "riskmodels"

def get_project_property(prop, project):
    init_file = Path(PROJECT_NAME) / '__init__.py'
    value = re.search(f'(^|\n){prop} *= *"v?([^\n]+)"', open(init_file).read())
    return value.group(2)

with open("README.md", "r") as fh:

    long_description = fh.read()

    setup(

     name=PROJECT_NAME,  

     version=get_project_property('__version__', PROJECT_NAME),

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = find_namespace_packages(include=['riskmodels', 'riskmodels.*']),

     python_requires='>=3.7',

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