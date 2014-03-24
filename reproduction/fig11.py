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
fig11.py

Reproduction of Figure 11 of the paper "Practical Evasion of a 
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
from mimicus.tools.featureedit import FeatureEdit
from mimicus.tools.datasets import csv2numpy

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

def fig11(tr_data, tr_labels, te_data, te_labels, tr_files):
    '''
    Tests the vaccination defense against the Benign Random Noise (BRN) 
    attack seeded by results of our mimicry attack against itself and 
    original, unmodified data. Performs 5 trials. 
    '''
    mal_tr_ind = [i for i, l in enumerate(tr_labels) if l == 1]
    ben_tr_ind = [i for i, l in enumerate(tr_labels) if l == 0]
    mim_data, mim_labels = common.get_FTC_mimicry()
    TRIALS = 5
    
    print '\n{:>6}{:>15}{:>15}'.format('%', 'ORIGINAL', 'OUR MIMICRY')
    pool = multiprocessing.Pool(processes=None)
    scores = []
    for subset in (0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1):
        acc = [0.0, 0.0]
        sys.stdout.write('{:>6.2f}'.format(subset * 100))
        for _ in range(TRIALS):
            tr_mod = tr_data.copy()
            # Subsample malicious training files for attack
            wolf_ind = random.sample(mal_tr_ind, 
                                     int(round(subset * len(mal_tr_ind))))
            
            # Mimic random benign files using the sampled files
            pargs = [(tr_data[random.choice(ben_tr_ind)], tr_files[w_id], w_id) 
                        for w_id in wolf_ind]
            for mimic, w_id in pool.imap(mimicry_wrap, pargs):
                tr_mod[w_id] = mimic
            
            # Evaluate the classifier on both clean test data and mimicry data
            res = common.evaluate_classifier(tr_mod, 
                                             tr_labels, 
                                             [te_data, mim_data], 
                                             [te_labels, mim_labels])
            acc = [old + new for old, new in zip(acc, res)]
        acc = [acc[0] / TRIALS, acc[1] / TRIALS]
        print '{:>15.3f}{:>15.3f}'.format(acc[0], acc[1])
        scores.append(tuple(acc))
    return scores

def main():
    random.seed(0)
    parser = ArgumentParser()
    parser.add_argument('--plot', help='Where to save plot (file name)',
                        default=False)
    parser.add_argument('--show', help='Show plot in a window', default=False)
    args = parser.parse_args()
    
    print 'Loading training data from CSV...'
    tr_data, tr_labels, tr_fnames = csv2numpy(config.get('datasets', 
                                                         'contagio'))
     
    print 'Loading test data from CSV...'
    te_data, te_labels, _ = csv2numpy(config.get('datasets', 'contagio_test'))
    
    print 'Evaluating...'
    scores = fig11(tr_data, tr_labels, te_data, te_labels, tr_fnames)
    
    if not (args.plot or args.show):
        return 0
    
    # Plot
    original, our_mimicry = zip(*scores)
    fig = pyplot.figure()
    pyplot.plot(original, 
                label='Clean data', 
                marker='o', color='k', linewidth=2)
    pyplot.plot(our_mimicry, 
                label='Our mimicry', 
                marker='+', color='k', linewidth=2, linestyle=':')
    axes = fig.gca()
    
    # Set up axes and labels
    axes.yaxis.set_ticks([r / 10.0 for r in range(11)])
    axes.yaxis.grid()
    axes.set_ylim(0, 1)
    axes.set_ylabel('Accuracy')
    xticklabels = ['0', '0.05', '0.1', '0.5', '1', '5', '10', '50', '100']
    axes.set_xticklabels(xticklabels, rotation=0)
    axes.set_xlabel('Training set perturbation (%)')
    
    fig.subplots_adjust(bottom=0.13, top=0.95, left=0.11, right=0.96)
    pyplot.legend(loc='lower right')
    if args.show:
        pyplot.show()
    if args.plot:
        pyplot.savefig(args.plot, dpi=300, bbox_inches='tight')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
