import setuptools
from distutils.core import Extension

LATEST = "1.0.0"

# with open("README.md", "r") as fh:

#     long_description = fh.read()

setuptools.setup(

     name='riskmodels',  

     version=LATEST,

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = setuptools.find_namespace_packages(include=['riskmodels.*']),

     description="Models for risk modelling in power capacity planning",

     license = "MIT",

     install_requires=[
        'pydantic>=1.8.2',
        'emcee>=3.1.0',
        'scipy>=1.7.1',
        'numpy>=1.21.2',
        'matplotlib>=3.4.3',
        'pandas'
    ],

     #long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://bitbucket.com/nestorsag/phd",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ]

 )