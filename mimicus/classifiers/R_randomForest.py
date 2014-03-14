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
R_randomForest.py

Implements the R_randomForest class. 

Created on May 16, 2013.
'''

import random
import string
import tempfile
import threading

import numpy
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
from sklearn.base import BaseEstimator

from mimicus.tools import datasets

# Initialize R
rinterface.initr()
# Import randomForest library into R
robjects.r('library(randomForest)')
# Set R terminal output width for better printing
robjects.r('options(width=120)')
# Specifies the R input variable types for CSV imports according to the 
# PDFrate types in tools.featureedit.FeatureDescriptor
_r_colClasses = 'c("factor",rep("integer",8),"logical",rep("integer",39),"numeric",rep("integer",10),"numeric",rep("integer",7),"numeric",rep("integer",2),"numeric",rep("integer",6),"numeric",rep("integer",14),"logical",rep("numeric",15),rep("integer",7),rep("numeric",4),rep("integer",16))'

# Threading lock to make work with the R randomForest package thread-safe
_R_lock = threading.Lock()

class R_randomForest(BaseEstimator):
    '''
    A class representing the Random Forest classifier as implemented by the 
    randomForest package in the R programming language. 
    '''
    
    def __init__(self):
        '''
        Constructor.
        '''
        super(R_randomForest, self).__init__()
        self.traindata_Rname = self._get_new_R_name()
        self.model_Rname = self._get_new_R_name()
        self.noveldata_Rname = self._get_new_R_name()
        self.predictions_Rname = self._get_new_R_name()
        self.model_trained = False
        self.persistent_model_Rname = 'savedrfmodel353'
    
    def get_model_name(self):
        return self.model_Rname
    
    @staticmethod
    def _get_new_R_name():
        '''
        Returns a randomly generated 7-character word consisting of lower 
        case characters that is unique in the current R global namespace, 
        i.e., it can be safely used as a variable name without overwriting 
        existing variables.
        '''
        while 1:
            name = ''.join(random.choice(string.lowercase) for _ in range(7))
            with _R_lock:
                if not robjects.r('exists("{var}")'.format(var=name))[0]:
                    robjects.r('{var} <- 0'.format(var=name))
                    return name
    
    def decision_function(self, X):
        '''
        Classifies novel data points using a trained model. Returns a 
        list of predictions, one per data point, giving the probability 
        of the given data point belonging to the positive class. 
        '''
        assert self.model_trained, 'Must train a model before classification'
        with _R_lock:
            with tempfile.NamedTemporaryFile() as tmpfile:
                datasets.numpy2csv(tmpfile, X, numpy.zeros((X.shape[0],)))
                tmpfile.seek(0)
                # Read in the CSV file with the samples to be classified, omitting the second column (filename)
                robjects.r('{novel} <- read.csv("{csv}", header=TRUE, colClasses={cc})'.format(novel=self.noveldata_Rname, csv=tmpfile.name, cc=_r_colClasses))
                # Classify the new data points
                robjects.r('{pred} <- predict({model}, {novel}, type="prob")'.format(pred=self.predictions_Rname, model=self.model_Rname, novel=self.noveldata_Rname))
                predictions = list(robjects.r['{pred}'.format(pred=self.predictions_Rname)])
        # The first half of predictions is for the negative class, so get rid of the second half
        predictions = predictions[len(predictions) / 2:]
        res = numpy.zeros((X.shape[0], 1))
        for r, i in zip(predictions, range(X.shape[0])):
            res[i] = r
        return res
    
    def fit(self, X, y):
        '''
        Trains a new random forest classifier. 
        '''
        with _R_lock:
            with tempfile.NamedTemporaryFile() as tmpfile:
                datasets.numpy2csv(tmpfile, X, y)
                tmpfile.seek(0)
                # Read in the CSV file with the training samples, omitting the second column (filename)
                robjects.r('{train} <- read.csv("{csv}", header=TRUE, colClasses={cc})'.format(train=self.traindata_Rname, csv=tmpfile.name, cc=_r_colClasses))
                # Train a random forest named myforest using 1000 decision trees with 33 variables sampled at each split
                robjects.r('{model} <- randomForest(x={train}[,-1], y={train}[,1], ntree=1000, mtry=43, importance=TRUE)'.format(model=self.model_Rname, train=self.traindata_Rname))
        self.model_trained = True
    
    def predict(self, X, cutoff=0.5):
        '''
        Classifies novel data points using a trained model. Returns a 
        numpy.array of predictions, one per data point, giving the 
        predicted class with the given cutoff value. 
        '''
        return numpy.array([1.0 if p >= cutoff else 0.0 
                            for p in self.decision_function(X)])
    
    def save_model(self, modelfile):
        '''
        Saves a trained R randomForest model into the specified file. 
        
        modelfile - name of the file where the model should be saved.
        '''
        assert self.model_trained, 'Must train a model before classification'
        with _R_lock:
            # Rename the model into a known name so we can later recover it
            robjects.r('{persistentname} <- {model}'.format(persistentname=self.persistent_model_Rname, model=self.model_Rname))
            # Save the model into a file
            robjects.r('save({persistentname}, file="{file}")'.format(persistentname=self.persistent_model_Rname, file=modelfile))
    
    def load_model(self, modelfile):
        '''
        Loads a trained R randomForest model from the specified file. 
        
        modelfile - name of the file where the model is saved.
        '''
        with _R_lock:
            # Load the model from the file
            robjects.r('load("{file}")'.format(file=modelfile))
            # Rename the model from the persistent name back to the original name
            robjects.r('{model} <- {persistentname}'.format(model=self.model_Rname, persistentname=self.persistent_model_Rname))
        self.model_trained = True