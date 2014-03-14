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
RandomForest.py

Implements the RandomForest class. If the rpy2 interface to R or 
the R randomForest package are unavailable, uses the scikit_learn 
implementation instead. 

Created on March 7, 2014.
'''

'''
Concrete implementation will be decided based on the availability 
of the rpy2 Python package, the R programming language and its 
randomForest package. If they are not available, scikit_learn 
implementation is used instead. 
'''
RandomForest = 0
try:
    from .R_randomForest import R_randomForest
    RandomForest = R_randomForest
except:
    import sys
    sys.stderr.write('R randomForest implementation not found, '\
                     'falling back to scikit_learn.\n')
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    class sklearn_RF(RandomForestClassifier):
        '''
        A customized version of the scikit_learn Random Forest.
        '''
        
        def __init__(self,
                     n_estimators=1000, # Used by PDFrate
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     max_features=43, # Used by PDFrate
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=-1, # Run in parallel
                     random_state=None,
                     verbose=0):
            '''
            Constructor.
            '''
            super(sklearn_RF, 
                  self).__init__(n_estimators = n_estimators,
                                 criterion = criterion,
                                 max_depth = max_depth,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 oob_score = oob_score,
                                 n_jobs = n_jobs,
                                 random_state = random_state,
                                 verbose = verbose)
        
        def decision_function(self, X):
            '''
            Returns the class probability of the given samples.
            '''
            return self.predict_proba(X)[:, [1]]
        
        def save_model(self, model_file):
            '''
            Saves a trained model into the specified file. 
            
            model_file - name of the file where the model should be saved.
            '''
            pickle.dump(self.__dict__, open(model_file, 'wb+'))
        
        def load_model(self, model_file):
            '''
            Loads a trained model from the specified file. 
            
            model_file - name of the file where the model is saved.
            '''
            self.__dict__.update(pickle.load(open(model_file, 'rb')))
    
    RandomForest = sklearn_RF
