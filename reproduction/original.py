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
original.py

Reproduction of the original attack and defense measures from 
"Malicious PDF Detection Using Metadata and Structural Features" 
by Charles Smutz and Angelos Stavrou, ACSAC 2012.

Created on September 18, 2013.
'''

from argparse import ArgumentParser
import multiprocessing
import os
import random
import sys

import numpy
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

from mimicus.tools.featureedit import FeatureDescriptor, FileDefined, \
                                        FeatureEdit
from mimicus.classifiers.RandomForest import RandomForest
from mimicus.tools import datasets, utility

import config

'''
Top features sorted by variable importance as reported by the R 
randomForest package for the trained model in the FTC scenario. 
'''
top_feats = ['count_font', 
             'count_js', 
             'count_javascript', 
             'pos_box_max', 
             'pos_eof_avg', 
             'pos_eof_max', 
             'len_stream_min', 
             'count_obj', 
             'count_endobj',
             'producer_len']

'''
Cached indices of the top features.
'''
feat_indices = [FeatureDescriptor.get_feature_names().index(feat) for feat in top_feats]

def get_benign_mean_stddev(data, labels):
    '''
    Returns feature medians and standard deviations for benign training data. 
    '''
    sys.stdout.write('Calculating feature medians and standard deviations for benign training data...\n')
    benign_vectors = data[[i for i, l in enumerate(labels) if l == 0],]
    return (numpy.mean(benign_vectors, axis = 0), numpy.std(benign_vectors, axis = 0))

def table10(realistic_feats, with_files, train_data, train_labels, test_data, test_labels, file_names):
    '''
    Reproduction of results published in Table 10 of "Malicious PDF Detection 
    Using Metadata and Structural Features" by Charles Smutz and 
    Angelos Stavrou, ACSAC 2012.
    '''
    sys.stdout.write('Loading random forest classifier...\n')
    rf = RandomForest()
    rf.load_model(config.get('experiments', 'contagio_rf_model'))
    benign_means, benign_stddevs = get_benign_mean_stddev(train_data, train_labels)
    sys.stdout.write('Got {} samples. Modifying them for attack...\n'.format(len(test_data)))
    all_feats = FeatureDescriptor.get_feature_names()
    for feat in top_feats:
        feat_i = all_feats.index(feat)
        feat_desc = FeatureDescriptor.get_feature_description(feat)
        sys.stdout.write('  modifying feature "{}" [mean {}, stddev {}] bounds: ({}, {})\n'.format(feat, benign_means[feat_i], benign_stddevs[feat_i], feat_desc['range'][0], feat_desc['range'][1]))
        for i in range(len(test_data)):
            if test_labels[i] == 1:
                while True:
                    new_val = random.gauss(benign_means[feat_i], benign_stddevs[feat_i])
                    if realistic_feats == False:
                        test_data[i][feat_i] = new_val
                        break
                    
                    if feat_desc['type'] == bool:
                        new_val = False if new_val < 0.5 else True
                    elif feat_desc['type'] == int:
                        new_val = int(round(new_val))
                    if feat_desc['range'][0] == FileDefined and new_val < 0:
                        continue
                    elif feat_desc['range'][0] != FileDefined and new_val < feat_desc['range'][0]:
                        continue
                    if feat_desc['type'] != bool and feat_desc['range'][1] < new_val:
                        continue
                    test_data[i][feat_i] = new_val
                    break
        
        sys.stdout.write('Predicting...\n')
        if with_files:
            mod_data = test_data.copy()
            for i in range(len(test_data)):
                if test_labels[i] == 1:
                    reply = FeatureEdit(file_names[i]).modify_file(test_data[i])
                    os.remove(reply['path'])
                    mod_data[i] = reply['feats']
            pred = rf.decision_function(mod_data)
            utility.print_stats_cutoff(test_labels, pred, [0.5])
        else:
            pred = rf.decision_function(test_data)
            utility.print_stats_cutoff(test_labels, pred, [0.5])

def get_mimicry_test_set():
    '''
    Returns a numpy.array of size (number of samples, number of features) 
    with feature values of all mimicry attack results.
    '''
    sys.stdout.write('Loading feature vectors from mimicry attack results...\n')
    results = numpy.array(numpy.zeros((100, FeatureDescriptor.get_feature_count())))
    pdfs = utility.get_pdfs(config.get('results', 'mimicry_contagio_rf'))
    for i in range(100):
        results[i,] = FeatureEdit(pdfs[i]).retrieve_feature_vector_numpy()
    
    return results, [1.0 for i in range(100)]

def evaluate_classifier(data, labels, test_data, test_labels):
    '''
    Returns the classification accuracies of the RandomForest classifier 
    trained on (data, labels) and tested on a list of (test_data, test_labels). 
    '''
    rf = RandomForest()
    rf.fit(data, labels)
    accs = []
    for ted, tel in zip(test_data, test_labels):
        pred = rf.predict(ted)
        accs.append(accuracy_score(tel, pred))
    return accs

def perturbate(data, labels, subset, ben_means, ben_stddevs):
    '''
    Modifies a subset of malicious feature vectors in data. The modified 
    vectors have a subset of their features randomly sampled from a normal 
    distribution with the mean ben_means and standard deviation ben_stddevs.
    '''
    num_malicious = int(round(sum(labels)))
    total = int(round(subset * num_malicious))
    indices = set(random.sample(range(num_malicious), total))
    i = mal_i = 0
    while total > 0:
        if labels[i] == 1:
            if mal_i in indices:
                for feat_i in feat_indices:
                    data[i][feat_i] = random.gauss(ben_means[feat_i], ben_stddevs[feat_i])
                total -= 1
            mal_i += 1
        i += 1
    
    return data

def perturbate_CV(data, labels, mimicry_labels, mimicry_data, ben_means, ben_stddevs, subset, TRIALS, nCV):
    '''
    Runs TRIALS trials of nCV-fold cross-validation, training RandomForest 
    on a perturbated subset of data and testing on (1) original, clean data, 
    (2) 100% perturbated data, and (3) mimicry attack samples. Returns a list 
    of classification accuracy values, one per test set, summed across all 
    trials. 
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
            test_data = [trial_data[te], perturbate(trial_data[te], trial_labels[te], 1.0, ben_means, ben_stddevs), mimicry_data]
            test_labels = [trial_labels[te], trial_labels[te], mimicry_labels]
            acc = evaluate_classifier(perturbate(trial_data[tr], trial_labels[tr], subset, ben_means, ben_stddevs), trial_labels[tr], test_data, test_labels)
            accs = [old + new for old, new in zip(accs, acc)]
    return accs

def perturbate_CV_parallel(args):
    '''
    Helper function for calling the perturbate_CV function in parallel.
    '''
    return perturbate_CV(*args)

def table12(data, labels):
    '''
    Reproduction of results published in Table 12 of "Malicious PDF Detection 
    Using Metadata and Structural Features" by Charles Smutz and 
    Angelos Stavrou, ACSAC 2012.
    '''
    ben_means, ben_stddevs = get_benign_mean_stddev(data, labels)
    feat_indices = [FeatureDescriptor.get_feature_names().index(feat) for feat in top_feats]
    mimicry_data, mimicry_labels = get_mimicry_test_set()
    TRIALS = 5
    nCV = 10
    subsets = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    
    pool = multiprocessing.Pool(processes=None)
    pool_args = [(data, labels, mimicry_labels, mimicry_data, ben_means, ben_stddevs, subset, TRIALS, nCV) for subset in subsets]
    sys.stdout.write('\n{:>6}{:>15}{:>15}{:>15}\n'.format('%', 'ORIGINAL', 'MIMICRY', 'OUR MIMICRY'))
    for accs, subset in zip(pool.imap(perturbate_CV_parallel, pool_args), subsets):
        sys.stdout.write('{:>6.2f}'.format(subset * 100))
        sys.stdout.write('{}\n'.format(''.join(['{:>15.3f}'.format(acc / (TRIALS * nCV)) for acc in accs])))

def mimicry(sheep_fname, wolf_fname):
    '''
    Mimics file with the name sheep_fname using the attack file 
    with the name wolf_fname. Returns the resulting feature vector. 
    '''
    sheep_feats = FeatureEdit(sheep_fname).retrieve_feature_vector()
    mimic = FeatureEdit(wolf_fname).modify_file(sheep_feats)
    os.remove(mimic['path'])
    return mimic['feats']

def mimicry_parallel(args):
    '''
    Helper function for calling the mimicry function in parallel.
    '''
    return mimicry(*args)

def our_mimicry(train_data, train_labels, test_data, test_labels, train_fnames):
    '''
    Tests the vaccination defense against the Benign Random Noise (BNR) 
    attack seeded by results of our mimicry attack against itself and 
    original, unmodified data. Performs 5 trials. 
    '''
    mal_train_is = [i for i in range(len(train_labels)) if train_labels[i] == 1]
    num_malicious = len(mal_train_is)
    ben_train_is = [i for i in range(len(train_labels)) if train_labels[i] == 0]
    mimicry_data, mimicry_labels = get_mimicry_test_set()
    TRIALS = 5
    
    sys.stdout.write('\n{:>6}{:>15}{:>15}\n'.format('%', 'ORIGINAL', 'OUR MIMICRY'))
    pool = multiprocessing.Pool(processes=None)
    for subset in [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        accs = [0.0, 0.0]
        total = int(round(subset * num_malicious))
        sys.stdout.write('{:>6.2f}'.format(subset * 100))
        for _ in range(TRIALS):
            new_data = train_data.copy()
            wolf_is = random.sample(mal_train_is, total)
            
            pool_args = [(train_fnames[random.choice(ben_train_is)], train_fnames[wolf_i]) for wolf_i in wolf_is]
            for mimic, wolf_i in zip(pool.imap(mimicry_parallel, pool_args), wolf_is):
                new_data[wolf_i] = mimic
            
            acc = evaluate_classifier(new_data, train_labels, [test_data, mimicry_data], [test_labels, mimicry_labels])
            accs = [old + new for old, new in zip(accs, acc)]
        sys.stdout.write('{}\n'.format(''.join(['{:>15.3f}'.format(acc / (TRIALS)) for acc in accs])))

def main():
    random.seed(0)
    parser = ArgumentParser()
    parser.add_argument('-f', '--with-files', default=False, help='Test on PDFs instead of feature vectors for Table 10', action='store_true')
    parser.add_argument('-r', '--realistic-feats', default=False, help='Only generate realistic feature values (within bounds) for Table 10', action='store_true')
    parser.add_argument('-e', '--experiment', type=str, choices=['table10', 'table12', 'our_mimicry'], nargs=1, help='Which experiment to run')
    args = parser.parse_args()
    
    args.experiment = args.experiment[0]
    print args
    
    sys.stdout.write('Loading training data from CSV...\n')
    train_data, train_labels, train_fnames = datasets.csv2numpy(config.get('datasets', 'contagio'))
    
    sys.stdout.write('Loading test data from CSV...\n')
    test_data, test_labels, test_fnames = datasets.csv2numpy(config.get('datasets', 'contagio_test'))
    
    if args.experiment == 'table10':
        table10(args.realistic_feats, args.with_files, train_data, train_labels, test_data, test_labels, test_fnames)
    elif args.experiment == 'table12':
        table12(train_data, train_labels)
    elif args.experiment == 'our_mimicry':
        our_mimicry(train_data, train_labels, test_data, test_labels, train_fnames)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())