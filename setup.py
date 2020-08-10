#! /usr/bin/env python

from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='disaggregator',
    version='0.0.1',
    author='DemandRegioTeam',
    author_email='fgotzens@fz-juelich.de',
    description='A tool for disaggregating electrical and thermal demand in high resolution.',
    namespace_package=['disaggregator'],
    long_description=read('README.md'),
    packages=find_packages(),
    package_dir={'disaggregator': 'disaggregator'},
 #   extras_require={
 #         'dev': ['nose', 'sphinx', 'sphinx_rtd_theme', 'requests']},
    install_requires=[
        'pandas >= 0.17.0',
        'PyYAML',
        'requests',
        'geopandas',
        'xarray',
        'matplotlib',
        'holidays'
    ])
