#!/usr/bin/env python
'''
Copyright 2013, 2014 Nedim Srndic, University of Tuebingen

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
datasets_test.py

Created on May 23, 2013
'''

import os
import unittest

from mimicus import config
from mimicus.tools import datasets

class datasets_Test(unittest.TestCase):
    '''
    Tests for the mimicus.tools.datasets module.
    '''
    def setUp(self):
        pass
    
    def test_standardize_csv(self):
        datasets.standardize_csv(config.get('datasets_test', 'csv_in'), config.get('datasets_test', 'csv_temp'))
        os.remove(config.get('datasets_test', 'csv_temp'))

if __name__ == "__main__":
    unittest.main()