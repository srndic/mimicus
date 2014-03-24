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
fig9.py

Reproduction of Figure 9 of the paper "Practical Evasion of a 
Learning-Based Classifier: A Case Study" by Nedim Srndic and 
Pavel Laskov. 

Created on March 21, 2014.
'''

from argparse import ArgumentParser
import multiprocessing
import os
import random
import sys

from matplotlib import pyplot
from mimicus.tools.featureedit import FeatureDescriptor, FeatureEdit, \
                                        FileDefined
from mimicus.classifiers.RandomForest import RandomForest
from mimicus.tools.datasets import csv2numpy
from sklearn.metrics import accuracy_score

import common
import config

def mimicry(wolf_fname, sheep_feats, m_id):
    '''
    Mimics file with the features sheep_feats using the attack file 
    with the name wolf_fname. Returns the resulting feature vector. 
    '''
    mimic = FeatureEdit(wolf_fname).modify_file(sheep_feats, '/run/shm')
    os.remove(mimic['path'])
    return mimic['feats'], m_id

def mimicry_wrap(args):
    '''
    Helper function for calling the mimicry function in parallel.
    '''
    return mimicry(*args)

def fig9(tr_vec, tr_labels, te_vec, te_labels, fnames):
    '''
    Reproduction of results published in Table 10 of "Malicious PDF Detection 
    Using Metadata and Structural Features" by Charles Smutz and 
    Angelos Stavrou, ACSAC 2012.
    '''
    print 'Loading random forest classifier...'
    rf = RandomForest()
    rf.load_model(config.get('experiments', 'FTC_model'))
    ben_means, ben_devs = common.get_benign_mean_stddev(tr_vec, tr_labels)
    res = []
    # te_vec will be randomly modified in feature space.
    # f_vec will be randomly modified in feature space but the 
    # randomly generated variables will be adjusted to be 
    # valid for the given feature
    f_vec = te_vec.copy()
    print 'Got {} samples. Modifying them for attack...'.format(len(te_vec))
    print '{:>25s} {:>15s} {:>15s}'.format('Feature name', 'Feature space', 
                                           'Problem space')
    pool = multiprocessing.Pool(processes=None)
    # Modify top features one by one
    for f_name in common.top_feats:
        f_i = FeatureDescriptor.get_feature_names().index(f_name)
        f_desc = FeatureDescriptor.get_feature_description(f_name)
        print '{:>25s}'.format(f_name),
        
        # For all files
        for i in range(len(te_vec)):
            if te_labels[i] != 1:
                # Modify only malicious files
                continue
            
            first_val = True
            while True:
                # Keep randomly generating a new value
                # Stop when it becomes valid for the current feature
                new_val = random.gauss(ben_means[f_i], ben_devs[f_i])
                if first_val:
                    # Make sure we generate random values for te_vec
                    te_vec[i][f_i] = new_val
                    first_val = False
                
                # If not valid, retry 
                if f_desc['type'] == bool:
                    new_val = False if new_val < 0.5 else True
                elif f_desc['type'] == int:
                    new_val = int(round(new_val))
                if f_desc['range'][0] == FileDefined and new_val < 0:
                    continue
                elif (f_desc['range'][0] != FileDefined and 
                        new_val < f_desc['range'][0]):
                    continue
                if f_desc['type'] != bool and f_desc['range'][1] < new_val:
                    continue
                # Valid, win!
                f_vec[i][f_i] = new_val
                break
        
        # mod_data has feature values read from the problem space, 
        # i.e., by converting feature vectors to files and back
        mod_data = f_vec.copy()
        pargs = [(fnames[i], f_vec[i], i) 
                 for i, l in enumerate(te_labels) if l == 1]
        for mimic, m_id in pool.imap(mimicry_wrap, pargs):
                mod_data[m_id] = mimic
        pred = rf.predict(te_vec)
        fspace = accuracy_score(te_labels, pred)
        print '{:>15.3f}'.format(fspace),
        pred = rf.predict(mod_data)
        pspace = accuracy_score(te_labels, pred)
        print '{:>15.3f}'.format(pspace)
        res.append((fspace, pspace))
    return res

def main():
    random.seed(0)
    parser = ArgumentParser()
    parser.add_argument('--plot', help='Where to save plot (file name)',
                        default=False)
    parser.add_argument('--show', help='Show plot in a window', default=False, 
                        action='store_true')
    args = parser.parse_args()
    
    print 'Loading training data from CSV...'
    tr_data, tr_labels, _ = csv2numpy(config.get('datasets', 'contagio'))
    
    print 'Loading test data from CSV...'
    te_data, te_labels, te_fnames = csv2numpy(config.get('datasets', 
                                                         'contagio_test'))
    
    print 'Evaluating...'
    scores = fig9(tr_data, tr_labels, te_data, te_labels, te_fnames)
    
    if not (args.plot or args.show):
        return 0
    
    # Plot
    feat_points, file_points = zip(*scores)
    fig = pyplot.figure()
    pyplot.plot(feat_points, label='Feature space', 
                marker='o', color='k', linewidth=2)
    pyplot.plot(file_points, label='Problem space', 
                marker='^', color='k', linewidth=2, linestyle='--')
    axes = fig.gca()
    
    # Set up axes and labels
    axes.yaxis.set_ticks([r / 10.0 for r in range(11)])
    axes.yaxis.grid()
    axes.set_ylim(0, 1)
    axes.set_ylabel('Accuracy')
    xticklabels = [common.top_feats[0]] + ['(+) ' + name 
                                           for name in common.top_feats[1:]]
    axes.set_xticklabels(xticklabels, rotation=60, ha='right')
    
    fig.subplots_adjust(bottom=0.34, top=0.95, left=0.11, right=0.98)
    pyplot.legend(loc='lower left')
    if args.show:
        pyplot.show()
    if args.plot:
        pyplot.savefig(args.plot, dpi=300, bbox_inches='tight')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
