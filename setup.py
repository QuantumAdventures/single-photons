from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

import os


PROJECT_DIR = os.path.dirname(__file__)

setup(
     name='single_photons',  
     version='0.0.1',
     author='Quantum Adventures',
     author_email='laboratorio.quantica.cetuc@gmail.com',
     description='short description here',
     long_description=long_description,
     long_description_content_type='text/markdown',
     license=open(os.path.join(PROJECT_DIR, 'LICENSE')).read(),
     packages=find_packages(),
     url='',
     classifiers=[
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
     ],
    install_requires=[
        'numpy',
        'pytest',
        'matplotlib',
        'seaborn'
    ]
)