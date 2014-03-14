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
sklearn_SVC_test.py

Created on May 23, 2013.
'''

import os
import unittest

import numpy

from mimicus import config
from mimicus.classifiers.sklearn_SVC import sklearn_SVC
from mimicus.tools import datasets

class sklearn_SVC_Test(unittest.TestCase):
    '''
    Tests for the sklearn_SVC class.
    '''
    def setUp(self):
        self.svc = sklearn_SVC()
    
    def test_constructor(self):
        _ = sklearn_SVC()
        _ = sklearn_SVC()
        _ = sklearn_SVC()
    
    def test_fit(self):
        X, y, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'traindata'))
        self.svc.fit(X, y)
    
    def test_predict(self):
        X, y, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'traindata'))
        self.svc.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'noveldata'))
        self.assertTrue(len(self.svc.predict(newX)) == 20)
    
    def test_decision_function(self):
        X, y, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'traindata'))
        self.svc.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'noveldata'))
        self.assertTrue(len(self.svc.decision_function(newX)) == 20)
    
    def test_save_model(self):
        X, y, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'traindata'))
        self.svc.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'noveldata'))
        self.assertTrue(len(self.svc.predict(newX)) == 20)
        self.svc.save_model(config.get('sklearn_SVC_test', 'modelfile'))
        os.remove(config.get('sklearn_SVC_test', 'modelfile'))
    
    def test_load_model(self):
        X, y, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'traindata'))
        self.svc.fit(X, y)
        newX, _, _ = datasets.csv2numpy(config.get('sklearn_SVC_test', 'noveldata'))
        prediction = self.svc.predict(newX)
        self.svc.save_model(config.get('sklearn_SVC_test', 'modelfile'))
        newsvc = sklearn_SVC()
        newsvc.load_model(config.get('sklearn_SVC_test', 'modelfile'))
        self.assertTrue(numpy.array_equal(prediction, newsvc.predict(newX)))
        os.remove(config.get('sklearn_SVC_test', 'modelfile'))
    

if __name__ == "__main__":
    unittest.main()