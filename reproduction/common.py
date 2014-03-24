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
common.py

Code common to all scenarios.

Created on March 5, 2014.
'''

import multiprocessing
from os import path
import pickle
import random
import shutil
import sys

from matplotlib import pyplot
from mimicus import config as _ # Just to create the configuration file
from mimicus.attacks.mimicry import mimicry
from mimicus.attacks.gdkde import gdkde
from mimicus.classifiers.RandomForest import RandomForest
from mimicus.classifiers.sklearn_SVC import sklearn_SVC
from mimicus.tools import datasets, utility
from mimicus.tools.featureedit import FeatureDescriptor, FeatureEdit
import numpy
from sklearn.metrics import accuracy_score

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

def get_benign_mean_stddev(data, labels):
    '''
    Returns feature medians and standard deviations for benign 
    training data. 
    '''
    print 'Getting medians and std. dev. for features of benign training data'
    benign_vectors = data[[i for i, l in enumerate(labels) if l == 0],]
    return (numpy.mean(benign_vectors, axis = 0), 
            numpy.std(benign_vectors, axis = 0))

def get_FTC_mimicry():
    '''
    Returns a numpy.array of size (number of samples, number of 
    features) with feature values of all mimicry attack results in 
    the FTC scenario.
    '''
    pdfs = utility.get_pdfs(config.get('results', 'FTC_mimicry'))
    if not pdfs:
        # Generate the attack files
        attack_mimicry('FTC')
        pdfs = utility.get_pdfs(config.get('results', 'FTC_mimicry'))
    
    print 'Loading feature vectors from mimicry attack results...'
    results = numpy.zeros((len(pdfs), FeatureDescriptor.get_feature_count()))
    for i in range(len(pdfs)):
        results[i,] = FeatureEdit(pdfs[i]).retrieve_feature_vector_numpy()
    
    return results, [1.0 for i in range(len(pdfs))]

def evaluate_classifier(data, labels, test_data, test_labels):
    '''
    Returns the classification accuracies of the RandomForest 
    classifier trained on (data, labels) and tested on a list of 
    (test_data, test_labels). 
    '''
    rf = RandomForest()
    rf.fit(data, labels)
    accs = []
    for ted, tel in zip(test_data, test_labels):
        pred = rf.predict(ted)
        accs.append(accuracy_score(tel, pred))
    return accs

'''
A dictionary encoding adversarial knowledge for every scenario.
'''
_scenarios = \
{'F' : {'classifier' : 'svm', 
        'model' : config.get('experiments', 'F_scaled_model'), 
        'targets' : config.get('experiments', 'surrogate_attack_targets'),
        'training' : config.get('datasets', 'surrogate_scaled')}, 
 'FT' : {'classifier' : 'svm',
         'model' : config.get('experiments', 'FT_scaled_model'), 
         'targets' : config.get('experiments', 'contagio_attack_targets'),
         'training' : config.get('datasets', 'contagio_scaled')}, 
 'FC' : {'classifier' : 'rf',
         'model' : config.get('experiments', 'FC_model'),
         'targets' : config.get('experiments', 'surrogate_attack_targets'), 
         'training' : config.get('datasets', 'surrogate')}, 
 'FTC' : {'classifier' : 'rf',
         'model' : config.get('experiments','FTC_model'),
         'targets' : config.get('experiments', 'contagio_attack_targets'), 
         'training' : config.get('datasets', 'contagio')},}

def _learn_model(scenario_name):
    '''
    Learns a classifier model for the specified scenario if one does 
    not already exist. 
    '''
    scenario = _scenarios[scenario_name]
    if path.exists(scenario['model']):
        return
    
    print 'Training the model for scenario {}...'.format(scenario_name)
    # Decide on classifier
    classifier = 0
    if scenario['classifier'] == 'rf':
        classifier = RandomForest()
        sys.stdout.write('TRAINING RANDOM FOREST\n')
        cutoff = [c * 0.1 for c in range(1, 10)]
    elif scenario['classifier'] == 'svm':
        classifier = sklearn_SVC(kernel='rbf', C=10, gamma=0.01)
        sys.stdout.write('TRAINING SVM\n')
        cutoff = [0.0]
    
    # Load the required dataset and train the model
    X, y, _ = datasets.csv2numpy(scenario['training'])
    classifier.fit(X, y)
    
    # Evaluate the model on the training dataset
    y_pred = classifier.decision_function(X)
    sys.stdout.write('Performance on training data:\n')
    utility.print_stats_cutoff(y, y_pred, cutoff)
    
    # Save the model in the corresponding file
    classifier.save_model(scenario['model'])

def _attack_files_missing(attack_files):
    sys.stderr.write('Unable to locate list of attack files {}. '
                     .format(attack_files))
    sys.stderr.write(('Please list the paths to your attack files in '
                      'this file, one per line.\n'))
    sys.exit()

def _initialize():
    '''
    Assembles missing datasets and learns missing models. 
    '''
    
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
    
    if not path.exists(config.get('datasets', 'contagio')):
        print 'Creating the contagio dataset...'
        merge_CSVs(config.get('datasets', 'contagio_ben'), 
                   config.get('datasets', 'contagio_mal'), 
                   config.get('datasets', 'contagio'))
    
    if not path.exists(config.get('datasets', 'contagio_full')):
        print 'Creating the contagio-full dataset...'
        merge_CSVs(config.get('datasets', 'contagio'), 
                   config.get('datasets', 'contagio_nopdfrate'), 
                   config.get('datasets', 'contagio_full'))
    
    standardize_csv = datasets.standardize_csv
    
    if not path.exists(config.get('datasets', 'contagio_scaler')):
        print 'Creating the contagio-scaled-full dataset...'
        scaler = standardize_csv(config.get('datasets', 'contagio_full'), 
                                 config.get('datasets', 'contagio_scaled_full'))
        pickle.dump(scaler, open(config.get('datasets', 'contagio_scaler'), 
                                 'wb+'))
    
    if not path.exists(config.get('datasets', 'contagio_scaled')):
        print 'Creating the contagio-scaled dataset...'
        standardize_csv(config.get('datasets', 'contagio'), 
                                    config.get('datasets', 'contagio_scaled'),
                                    scaler)
    
    if not path.exists(config.get('datasets', 'contagio_test')):
        print 'Creating the contagio-test dataset...'
        shutil.copy(config.get('datasets', 'contagio_nopdfrate'),
                    config.get('datasets', 'contagio_test'))
    
    if not path.exists(config.get('datasets', 'contagio_scaled_test')):
        print 'Creating the contagio-scaled-test dataset...'
        standardize_csv(config.get('datasets', 'contagio_test'), 
                        config.get('datasets', 'contagio_scaled_test'),
                        scaler)
    
    if not path.exists(config.get('datasets', 'surrogate')):
        print 'Creating the surrogate dataset...'
        merge_CSVs(config.get('datasets', 'google_ben'), 
                   config.get('datasets', 'virustotal_mal'), 
                   config.get('datasets', 'surrogate'))
    
    if not path.exists(config.get('datasets', 'surrogate_scaled')):
        print 'Creating the surrogate-scaled dataset...'
        standardize_csv(config.get('datasets', 'surrogate'), 
                        config.get('datasets', 'surrogate_scaled'),
                        scaler)
    
    _learn_model('F')
    _learn_model('FC')
    _learn_model('FT')
    _learn_model('FTC')
    
    utility.mkdir_p(config.get('results', 'F_gdkde'))
    utility.mkdir_p(config.get('results', 'F_mimicry'))
    utility.mkdir_p(config.get('results', 'FC_mimicry'))
    utility.mkdir_p(config.get('results', 'FT_gdkde'))
    utility.mkdir_p(config.get('results', 'FT_mimicry'))
    utility.mkdir_p(config.get('results', 'FTC_mimicry'))

def _gdkde_wrapper(ntuple):
    '''
    A helper function to parallelize calls to gdkde().
    '''
    try:
        return gdkde(*ntuple)
    except Exception as e:
        return e

def attack_gdkde(scenario_name, plot=False):
    '''
    Invokes the GD-KDE attack for the given scenario and saves the 
    resulting attack files in the location specified by the 
    configuration file. If plot evaluates to True, saves the resulting 
    plot into the specified file, otherwise shows the plot in a window. 
    '''
    print 'Running the GD-KDE attack...'
    _initialize()
    scenario = _scenarios[scenario_name]
    output_dir = config.get('results', '{}_gdkde'.format(scenario_name))
    # Make results reproducible
    random.seed(0)
    # Load and print malicious files
    wolves = config.get('experiments', 'contagio_attack_pdfs')
    if not path.exists(wolves):
        _attack_files_missing(wolves)
    print 'Loading attack samples from "{}"'.format(wolves)
    malicious = utility.get_pdfs(wolves)
    if not malicious:
        _attack_files_missing(wolves)
    
    # Load an SVM trained with scaled data
    scaler = pickle.load(open(
                        config.get('datasets', 'contagio_scaler')))
    print 'Using scaler'
    svm = sklearn_SVC()
    print 'Loading model from "{}"'.format(scenario['model'])
    svm.load_model(scenario['model'])
    
    # Load the training data used for kernel density estimation
    print 'Loading dataset from file "{}"'.format(scenario['training'])
    X_train, y_train, _ = datasets.csv2numpy(scenario['training'])
    # Subsample for faster execution
    ind_sample = random.sample(range(len(y_train)), 500)
    X_train = X_train[ind_sample, :]
    y_train = y_train[ind_sample]
    
    # Set parameters
    kde_reg = 10
    kde_width = 50
    step = 1
    max_iter = 50
    
    # Set up multiprocessing
    pool = multiprocessing.Pool()
    pargs = [(svm, fname, scaler, X_train, y_train, kde_reg, 
                  kde_width, step, max_iter, False) for fname in malicious]
    
    if plot:
        pyplot.figure(1)
    print 'Running the attack...'
    for res, oldf in zip(pool.imap(_gdkde_wrapper, pargs), malicious):
        if isinstance(res, Exception):
            print res
            continue
        (_, fseq, _, _, attack_file) = res
        print 'Processing file "{}":'.format(oldf)
        print '  scores: {}'.format(', '.join([str(s) for s in fseq]))
        print 'Result: "{}"'.format(attack_file)
        if path.dirname(attack_file) != output_dir:
            shutil.move(attack_file, output_dir)
        if plot:
            pyplot.plot(fseq, label=oldf)
    
    print 'Saved resulting attack files to {}'.format(output_dir)
    
    if plot:
        pyplot.title('GD-KDE attack')
        axes = pyplot.axes()
        axes.set_xlabel('Iterations')
        axes.set_xlim(0, max_iter + 1)
        axes.set_ylabel('SVM score')
        axes.yaxis.grid()
        fig = pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
        if plot == 'show':
            pyplot.show()
        else:
            pyplot.savefig(plot, dpi=300)
            print 'Saved plot to file {}'.format(plot)

def _mimicry_wrap(ntuple):
    '''
    A helper function to parallelize calls to mimicry().
    '''
    try:
        return mimicry(*ntuple)
    except Exception as e:
        return e

def attack_mimicry(scenario_name, plot=False):
    '''
    Invokes the mimcry attack for the given scenario and saves the 
    resulting attack files in the location specified by the 
    configuration file. If plot evaluates to True, saves the resulting 
    plot into the specified file, otherwise shows the plot in a window. 
    '''
    print 'Running the mimicry attack...'
    _initialize()
    scenario = _scenarios[scenario_name]
    output_dir = config.get('results', '{}_mimicry'.format(scenario_name))
    # Make results reproducible
    random.seed(0)
    # Load benign files
    print 'Loading attack targets from file "{}"'.format(scenario['targets'])
    target_vectors, _, target_paths = datasets.csv2numpy(scenario['targets'])
    targets = zip(target_paths, target_vectors)
    # Load malicious files
    wolves = config.get('experiments', 'contagio_attack_pdfs')
    if not path.exists(wolves):
        _attack_files_missing(wolves)
    print 'Loading attack samples from file "{}"'.format(wolves)
    malicious = sorted(utility.get_pdfs(wolves))
    if not malicious:
        _attack_files_missing(wolves)
    
    # Set up classifier
    classifier = 0
    if scenario['classifier'] == 'rf':
        classifier = RandomForest()
        print 'ATTACKING RANDOM FOREST'
    elif scenario['classifier'] == 'svm':
        classifier = sklearn_SVC()
        print 'ATTACKING SVM'
    print 'Loading model from "{}"'.format(scenario['model'])
    classifier.load_model(scenario['model'])
    
    # Standardize data points if necessary
    scaler = None
    if 'scaled' in scenario['model']:
        scaler = pickle.load(open(config.get('datasets', 'contagio_scaler')))
        print 'Using scaler'
    
    # Set up multiprocessing
    pool = multiprocessing.Pool()
    pargs = [(mal, targets, classifier, scaler) for mal in malicious]
    
    if plot:
        pyplot.figure(1)
    print 'Running the attack...'
    for wolf_path, res in zip(malicious, pool.imap(_mimicry_wrap, pargs)):
        if isinstance(res, Exception):
            print res
            continue
        (target_path, mimic_path, mimic_score, wolf_score) = res
        print 'Modifying {p} [{s}]:'.format(p=wolf_path, s=wolf_score)
        print '  BEST: {p} [{s}]'.format(p=target_path, s=mimic_score)
        if path.dirname(mimic_path) != output_dir:
            print '  Moving best to {}\n'.format(path.join(output_dir, 
                                                 path.basename(mimic_path)))
            shutil.move(mimic_path, output_dir)
        if plot:
            pyplot.plot([wolf_score, mimic_score])
    
    print 'Saved resulting attack files to {}'.format(output_dir)
    
    if plot:
        pyplot.title('Mimicry attack')
        axes = pyplot.axes()
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Classifier score')
        axes.yaxis.grid()
        fig = pyplot.gcf()
        fig.set_size_inches(6, 4.5)
        fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
        if plot == 'show':
            pyplot.show()
        else:
            pyplot.savefig(plot, dpi=300)
            print 'Saved plot to file {}'.format(plot)
