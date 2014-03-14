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
utility.py

Created on July 2, 2013.
'''

import errno
import hashlib
import os
import sys

from sklearn.metrics import accuracy_score, confusion_matrix

def print_stats_cutoff(y_true, y_pred, cutoffs):
    '''
    Prints experiment results (accuracy and confusion matrix) for the given 
    list of cutoff points for classifiers which give a real-valued output. 
    '''
    for cutoff in cutoffs:
        sys.stdout.write('Cutoff: {}\n'.format(cutoff))
        y_cutoff = [0 if v <= cutoff else 1 for v in y_pred]
        score = accuracy_score(y_true, y_cutoff)
        sys.stdout.write('Accuracy: {}\n'.format(score))
        cm = confusion_matrix(y_true, y_cutoff, labels=[0, 1])
        sys.stdout.write('      TRUE  FALSE\n')
        sys.stdout.write('POS {tp:>6} {fp:>6}\n'.format(tp=cm[1][1], fp=cm[0][1]))
        sys.stdout.write('NEG {tn:>6} {fn:>6}\n'.format(tn=cm[0][0], fn=cm[1][0]))

def print_stats_binary(y_true, y_pred):
    '''
    Prints experiment results (accuracy and confusion matrix) for 
    classifiers which give a binary output. 
    '''
    score = accuracy_score(y_true, y_pred)
    sys.stdout.write('Accuracy: {}\n'.format(score))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sys.stdout.write('      TRUE  FALSE\n')
    sys.stdout.write('POS {tp:>6} {fp:>6}\n'.format(tp=cm[1][1], fp=cm[0][1]))
    sys.stdout.write('NEG {tn:>6} {fn:>6}\n'.format(tn=cm[0][0], fn=cm[1][0]))

def file_sha256_hash(filename):
    '''
    Returns the SHA256 hash of the given file.
    '''
    checksum_builder = hashlib.sha256()
    BLOCK_SIZE = checksum_builder.block_size * 1024
    with open(filename, 'rb') as infile:
        data = infile.read(BLOCK_SIZE)
        while data:
            checksum_builder.update(data)
            data = infile.read(BLOCK_SIZE)
    
    return checksum_builder.hexdigest()

def get_pdfs(source):
    '''
    Returns a list of all PDF files found in the given source. 
    
    source - a directory containing PDF files or a file with a list of 
        newline-separated paths to PDF files. 
    '''
    pdfs = []
    if os.path.isdir(source):
        pdfs = [os.path.join(source, f) for f in os.listdir(source) if 
                os.path.isfile(os.path.join(source, f)) and 
                os.path.splitext(f)[1] == '.pdf']
    else:
        pdfs = open(source, 'rb').read().splitlines()
    pdfs = map(os.path.abspath, pdfs)
    return pdfs

def mkdir_p(path):
    '''
    Creates a directory regardless whether it already exists or not.
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
