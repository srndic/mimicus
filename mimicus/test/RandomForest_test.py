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
RandomForest_test.py

Created on May 17, 2013
'''

import os
import unittest

import numpy

from mimicus import config
from mimicus.classifiers import RandomForest
from mimicus.tools import datasets

class RandomForest_Test(unittest.TestCase):
    '''
    Tests for the RandomForest class.
    '''
    
    def setUp(self):
        self.rf = RandomForest()
    
    def test_constructor(self):
        _ = RandomForest()
    
    def test_decision_function(self):
        X, y, _ = datasets.csv2numpy(config.get('RandomForest_test', 'traindata'))
        self.rf.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('RandomForest_test', 'noveldata'))
        self.assertTrue(len(self.rf.decision_function(newX)) == 20)
    
    def test_fit(self):
        X, y, _ = datasets.csv2numpy(config.get('RandomForest_test', 'traindata'))
        self.rf.fit(X, y)
    
    def test_predict(self):
        X, y, _ = datasets.csv2numpy(config.get('RandomForest_test', 'traindata'))
        self.rf.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('RandomForest_test', 'noveldata'))
        self.assertTrue(len(self.rf.predict(newX)) == 20)
    
    def test_save_model(self):
        X, y, _ = datasets.csv2numpy(config.get('RandomForest_test', 'traindata'))
        self.rf.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('RandomForest_test', 'noveldata'))
        self.assertTrue(len(self.rf.predict(newX)) == 20)
        self.rf.save_model(config.get('RandomForest_test', 'modelfile'))
        os.remove(config.get('RandomForest_test', 'modelfile'))
    
    def test_load_model(self):
        X, y, _ = datasets.csv2numpy(config.get('RandomForest_test', 'traindata'))
        self.rf.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('RandomForest_test', 'noveldata'))
        prediction = self.rf.predict(newX)
        self.rf.save_model(config.get('RandomForest_test', 'modelfile'))
        newrf = RandomForest()
        newrf.load_model(config.get('RandomForest_test', 'modelfile'))
        self.assertTrue(numpy.array_equal(prediction, newrf.predict(newX)))
        os.remove(config.get('RandomForest_test', 'modelfile'))

if __name__ == "__main__":
    unittest.main()