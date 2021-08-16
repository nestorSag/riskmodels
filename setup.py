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
        'pymc3>=3.11',
        'pydantic>=1.6'
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