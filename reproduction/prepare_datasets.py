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
prepare_datasets.py

Prepares all experimental datasets required to reproduce results.

Created on March 12, 2014.
'''

import pickle
import shutil
import sys

from mimicus import config as mimicusconfig # Just to create the configuration
from mimicus.tools.datasets import standardize_csv

import config

def merge_CSVs(csv1, csv2, out):
    '''
    Merges two CSV files into out. Skips any header or comment 
    lines in the second file.
    '''
    with open(out, 'wb+') as f:
        # Copy csv1
        f.write(open(csv1).read())
        # Skip junk in csv2
        with open(csv2) as csv2in:
            l = 'a'
            while l:
                l = csv2in.readline()
                if l and l[:4].lower() in ('true', 'fals'):
                    f.write(l)

def main():
    print 'Creating the contagio dataset...'
    merge_CSVs(config.get('datasets', 'contagio_ben'), 
               config.get('datasets', 'contagio_mal'), 
               config.get('datasets', 'contagio'))
    
    print 'Creating the contagio-full dataset...'
    merge_CSVs(config.get('datasets', 'contagio'), 
               config.get('datasets', 'contagio_nopdfrate'), 
               config.get('datasets', 'contagio_full'))
    
    print 'Creating the contagio-scaled-full dataset...'
    scaler = standardize_csv(config.get('datasets', 'contagio_full'), 
                             config.get('datasets', 'contagio_scaled_full'))
    pickle.dump(scaler, open(config.get('datasets', 'contagio_scaler'), 
                             'wb+'))
    
    print 'Creating the contagio-scaled dataset...'
    standardize_csv(config.get('datasets', 'contagio'), 
                                config.get('datasets', 'contagio_scaled'),
                                scaler)
    
    print 'Creating the contagio-test dataset...'
    shutil.copy(config.get('datasets', 'contagio_nopdfrate'),
                config.get('datasets', 'contagio_test'))
    
    print 'Creating the contagio-scaled-test dataset...'
    standardize_csv(config.get('datasets', 'contagio_test'), 
                    config.get('datasets', 'contagio_scaled_test'),
                    scaler)
    
    print 'Creating the surrogate dataset...'
    merge_CSVs(config.get('datasets', 'google_ben'), 
               config.get('datasets', 'virustotal_mal'), 
               config.get('datasets', 'surrogate'))
    
    print 'Creating the surrogate-scaled dataset...'
    standardize_csv(config.get('datasets', 'surrogate'), 
                    config.get('datasets', 'surrogate_scaled'),
                    scaler)
    return 0

if __name__ == '__main__':
    sys.exit(main())
