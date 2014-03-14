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
sklearn_SVC.py

Implements the sklearn_SVC class.

Created on May 23, 2013.
'''

import pickle

from sklearn.svm import SVC

class sklearn_SVC(SVC):
    '''
    A class representing the Support Vector Machine classifier as implemented 
    by scikit-learn. 
    '''

    def __init__(self, 
                 C=10, # Found using grid search
                 kernel='rbf', 
                 degree=3, 
                 gamma=0.01, # Found using grid search
                 coef0=0.0, 
                 shrinking=True, 
                 probability=False, 
                 tol=0.001, 
                 cache_size=200, 
                 class_weight=None, 
                 verbose=False, 
                 max_iter=-1):
        '''
        Constructor
        '''
        super(sklearn_SVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                       coef0=coef0, shrinking=shrinking, probability=probability, 
                       tol=tol, cache_size=cache_size, class_weight=class_weight, 
                       verbose=verbose, max_iter=max_iter)
    
    def save_model(self, modelfile):
        '''
        Saves a trained SVM model into the specified file. 
        
        modelfile - name of the file where the model should be saved.
        '''
        pickle.dump(self.__dict__, open(modelfile, 'wb+'))
    
    def load_model(self, modelfile):
        '''
        Loads a trained SVM model from the specified file. 
        
        modelfile - name of the file where the model is saved.
        '''
#         self.svc = pickle.load(open(modelfile, 'rb'))
        self.__dict__.update(pickle.load(open(modelfile, 'rb')))
