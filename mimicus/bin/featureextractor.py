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
feature_extractor.py

This program will extract features from all files whose filename ends 
in .pdf in the directory or file list given and save them in CSV 
format in the given file. 

Created on May 16, 2013.
'''

from argparse import ArgumentParser
import multiprocessing
import sys

import numpy

from mimicus.tools import datasets, featureedit
from mimicus.tools import utility

def get_features(pdf_name):
    try:
        pdf = featureedit.FeatureEdit(pdf_name)
    except:
        return None
    feats = pdf.retrieve_feature_dictionary()
    if feats['size'] == 0 or isinstance(feats['size'], Exception):
        return None
    return pdf_name, pdf.retrieve_feature_vector_numpy()

def extract_features(pdfs_ben, pdfs_mal, csv_name):
    feat_vecs = []
    labels = []
    file_names = []
    # Extract malicious and benign features
    pool = multiprocessing.Pool()
    for pdf, feats in pool.imap(get_features, pdfs_mal):
        if feats is not None:
            feat_vecs.append(feats)
            labels.append(1.0)
            file_names.append(pdf)
    
    for pdf, feats in pool.imap(get_features, pdfs_ben):
        if feats is not None:
            feat_vecs.append(feats)
            labels.append(0.0)
            file_names.append(pdf)
    
    # Convert the data points into numpy.array
    X = numpy.array(numpy.zeros((len(feat_vecs), 
                                 featureedit.FeatureDescriptor.get_feature_count())), 
                                 dtype=numpy.float64, order='C')
    for i, v in enumerate(feat_vecs):
        X[i, :] = v
    # Write the resulting CSV file
    datasets.numpy2csv(csv_name, X, labels, file_names)

def main():
    # Setup argument parser
    parser = ArgumentParser()
    parser.add_argument('--mal', help='Malicious PDFs (directory or file with list of paths)')
    parser.add_argument('--ben', help='Benign PDFs (directory or file with list of paths)')
    parser.add_argument('csv', help='Resulting CSV file')
    
    # Process arguments
    args = parser.parse_args()
    pdfs_mal, pdfs_ben = [], []
    if args.mal:
        pdfs_mal = sorted(utility.get_pdfs(args.mal))
    if args.ben:
        pdfs_ben = sorted(utility.get_pdfs(args.ben))
    
    extract_features(pdfs_ben, pdfs_mal, args.csv)
    return 0

if __name__ == "__main__":
    sys.exit(main())