#!/usr/bin/env python
'''
Copyright 2014 Nedim Srndic, University of Tuebingen

This file is part of Mimicus.

Mimicus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Mimicus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mimicus.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################
setup.py

Created on March 11, 2014.
'''

import multiprocessing # To fix a bug when running tests
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

def readme():
    with open('README.rst') as f:
        return f.read()

class MyInstall(install):
    '''
    A class for running custom post-install code.
    '''
    def run(self):
        '''
        Runs after installation. 
        '''
        install.run(self)
        from mimicus import config

class MyDevelop(develop):
    '''
    A class for running custom post-install code in develop-mode.
    '''
    def run(self):
        '''
        Runs after develop-mode installation. 
        '''
        develop.run(self)
        from mimicus import config

setup(name='mimicus',
      version='1.0',
      description='A library for adversarial classifier evasion',
      url='https://github.com/srndic/mimicus',
      download_url='https://github.com/srndic/mimicus/tarball/master',
      author='Nedim Srndic, Pavel Laskov',
      author_email='nedim.srndic@uni-tuebingen.de',
      license='GPLv3',
      packages=find_packages(),
      install_requires=['matplotlib >= 1.1.1rc', 
                        'numpy >= 1.6.1', 
                        'scikit_learn >= 0.13.1', 
                        'scipy >= 0.9.0'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      cmdclass={'install': MyInstall,
                'develop': MyDevelop})
