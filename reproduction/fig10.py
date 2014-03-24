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
fig10.py

Reproduction of Figure 10 of the paper "Practical Evasion of a 
Learning-Based Classifier: A Case Study" by Nedim Srndic and 
Pavel Laskov. 

Created on March 21, 2014.
'''

from argparse import ArgumentParser
import multiprocessing
import random
import sys

from matplotlib import pyplot
from mimicus.tools.featureedit import FeatureDescriptor
from mimicus.tools.datasets import csv2numpy
import numpy
from sklearn.cross_validation import KFold

import common
import config

def perturbate(data, labels, subset, ben_means, ben_devs):
    '''
    Modifies a subset of malicious feature vectors in data. The 
    modified vectors have a subset of their features randomly sampled 
    from a normal distribution with the mean ben_means and standard 
    deviation ben_devs.
    '''
    feat_indices = [FeatureDescriptor.get_feature_names().index(feat) 
                    for feat in common.top_feats]
    num_malicious = int(round(sum(labels)))
    total = int(round(subset * num_malicious))
    indices = set(random.sample(range(num_malicious), total))
    i = mal_i = 0
    while total > 0:
        if labels[i] == 1:
            if mal_i in indices:
                for feat_i in feat_indices:
                    data[i][feat_i] = random.gauss(ben_means[feat_i], 
                                                   ben_devs[feat_i])
                total -= 1
            mal_i += 1
        i += 1
    
    return data

def perturbate_CV(data, labels, mim_data, mim_labels, ben_means, ben_devs, 
                  subset, TRIALS, nCV):
    '''
    Runs TRIALS trials of nCV-fold cross-validation, training 
    RandomForest     on a perturbated subset of data and testing on (1) 
    original, clean data, (2) 100% perturbated data, and (3) mimicry 
    attack samples. Returns a list of classification accuracy values, 
    one per test set, summed across all trials. 
    '''
    accs = [0., 0., 0.]
    for _ in range(TRIALS):
        # Shuffle input data
        shuf_indices = numpy.arange(len(data))
        numpy.random.shuffle(shuf_indices)
        trial_data = data[shuf_indices,]
        trial_labels = labels[shuf_indices]
        
        # Run nCV-fold cross-validation
        kf = KFold(len(trial_data), n_folds=nCV, indices=True)
        for tr, te in kf:
            test_data = [trial_data[te], 
                         perturbate(trial_data[te], 
                                    trial_labels[te], 
                                    1.0, 
                                    ben_means, 
                                    ben_devs), 
                         mim_data]
            test_labels = [trial_labels[te], 
                           trial_labels[te], 
                           mim_labels]
            acc = common.evaluate_classifier(perturbate(trial_data[tr], 
                                                        trial_labels[tr], 
                                                        subset, 
                                                        ben_means, 
                                                        ben_devs), 
                                             trial_labels[tr], 
                                             test_data, 
                                             test_labels)
            accs = [old + new for old, new in zip(accs, acc)]
    return accs, subset

def perturbate_CV_parallel(args):
    '''
    Helper function for calling the perturbate_CV function in parallel.
    '''
    return perturbate_CV(*args)

def fig10(data, labels):
    '''
    Reproduction of results published in Table 12 of "Malicious PDF 
    Detection Using Metadata and Structural Features" by Charles Smutz 
    and Angelos Stavrou, ACSAC 2012.
    '''
    ben_means, ben_devs = common.get_benign_mean_stddev(data, labels)
    mim_data, mim_labels = common.get_FTC_mimicry()
    TRIALS = 5
    nCV = 10
    subsets = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    
    pool = multiprocessing.Pool(processes=None)
    pool_args = [(data, labels, mim_data, mim_labels, ben_means, ben_devs, 
                  subset, TRIALS, nCV) for subset in subsets]
    print '\n     % {:>15}{:>15}{:>15}'.format('ORIGINAL', 
                                              'MIMICRY', 
                                              'OUR MIMICRY'),
    norm = TRIALS * nCV
    res = []
    for accs, subset in pool.imap(perturbate_CV_parallel, pool_args):
        print '\n{:>6.2f}'.format(subset * 100),
        for acc in accs: sys.stdout.write('{:>15.3f}'.format(acc / norm))
        res.append(tuple([acc / norm for acc in accs]))
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
    
    print 'Evaluating...'
    scores = fig10(tr_data, tr_labels)
    print 'Done'
    
    if not (args.plot or args.show):
        return 0
    
    # Plot
    original, mimicry, our_mimicry = zip(*scores)
    fig = pyplot.figure()
    pyplot.plot(original, label='Unmodified data', 
                marker='o', color='k', linewidth=2)
    pyplot.plot(mimicry, label='BRN attack', 
                marker='^', color='k', linewidth=2, linestyle='--')
    pyplot.plot(our_mimicry, label='Our mimicry attack', 
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
    pyplot.legend(loc='center right')
    if args.show:
        pyplot.show()
    if args.plot:
        pyplot.savefig(args.plot, dpi=300, bbox_inches='tight')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
