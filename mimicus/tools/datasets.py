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
datasets.py

Created on Jun 4, 2013
'''

import csv

import numpy
from sklearn.preprocessing import StandardScaler

from mimicus.tools.featureedit import FeatureDescriptor

def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with 
    training vectors (numpy.array) and labels (numpy.array), respectfully. 
    
    csv_in - name of a CSV file with training data points; 
                the first column in the file is supposed to be named 
                'class' and should contain the class label for the data 
                points; the second column of this file will be ignored 
                (put data point ID here). 
    '''
    # Parse CSV file
    csv_rows = list(csv.reader(open(csv_in, 'rb')))
    classes = {'FALSE':0, 'TRUE':1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = numpy.array(numpy.zeros((TOTAL_ROWS, FeatureDescriptor.get_feature_count())), dtype=numpy.float64, order='C')
    y = numpy.array(numpy.zeros(TOTAL_ROWS), dtype=numpy.float64, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        file_names.append(row[1])
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y, file_names

def numpy2csv(csv_out, X, y, file_names=None):
    '''
    Creates a CSV file from the given data points (X, scipy matrix) and labels 
    (y, numpy.array). The CSV file has a header. The first column is named 
    'class' and the others after PDFrate features. All features are written 
    in their respective type format (e.g., True/False for booleans). 
    
    If 'csv_out' is an open Python file, it will not be reopened. If 
    it is a string, a file will be created with that name. 
    '''
    we_opened_csvfile = type(csv_out) == str
    csvfile = open(csv_out, 'wb+') if we_opened_csvfile else csv_out
    # Write header
    csvfile.write('class')
    if file_names:
        csvfile.write(',filename')
    names = FeatureDescriptor.get_feature_names()
    for name in names:
        csvfile.write(',{}'.format(name))
    csvfile.write('\n')
    descs = FeatureDescriptor.get_feature_descriptions()
    # Write data
    for i in range(0, X.shape[0]):
        csvfile.write('{}'.format('TRUE' if bool(y[i]) else 'FALSE'))
        if file_names:
            csvfile.write(',{}'.format(file_names[i]))
        for j in range(0, X.shape[1]):
            feat_type = descs[names[j]]['type']
            feat_val = X[i, j]
            if feat_type == bool:
                feat_val = 'TRUE' if feat_val >= 0.5 else 'FALSE'
            elif feat_type == int:
                feat_val = int(round(feat_val))
            csvfile.write(',{}'.format(feat_val))
        csvfile.write('\n')
    
    if we_opened_csvfile:
        csvfile.close()

def standardize_csv(csv_in, csv_out, standardizer=None):
    '''
    Standardizes data (subtracts the mean and divides by the standard deviation 
    every feature independently for every data point) from a CSV file 'csv_in' 
    and writes it into 'csv_out'. If no 'standardizer' 
    (sklearn.preprocessing.StandardScaler) is provided, one will be created 
    and fit on the dataset from the input CSV file. 
    
    Returns the standardizer so you can use it for other datasets. 
    '''
    X, y, file_names = csv2numpy(csv_in)
#     X = X.todense()
    if standardizer is None:
        standardizer = StandardScaler(copy=False)
        standardizer.fit(X)
    standardizer.transform(X)
    numpy2csv(csv_out, X, y, file_names)
    del X
    return standardizer
