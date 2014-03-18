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

import config
from mimicus.attacks.mimicry import mimicry
from mimicus.attacks.gdkde import gdkde
from mimicus.classifiers.RandomForest import RandomForest
from mimicus.classifiers.sklearn_SVC import sklearn_SVC
from mimicus.tools import datasets, utility

'''
A dictionary encoding adversarial knowledge for every scenario.
'''
scenarios = \
{'F' : {'classifier' : 'svm', 
        'model' : config.get('experiments', 'surrogate_scaled_svm_model'), 
        'targets' : config.get('experiments', 'surrogate_attack_targets'),
        'training' : config.get('datasets', 'surrogate_scaled')}, 
 'FT' : {'classifier' : 'svm',
         'model' : config.get('experiments', 'contagio_scaled_svm_model'), 
         'targets' : config.get('experiments', 'contagio_attack_targets'),
         'training' : config.get('datasets', 'contagio_scaled')}, 
 'FC' : {'classifier' : 'rf',
         'model' : config.get('experiments', 'surrogate_rf_model'),
         'targets' : config.get('experiments', 'surrogate_attack_targets'), 
         'training' : config.get('datasets', 'surrogate')}, 
 'FTC' : {'classifier' : 'rf',
         'model' : config.get('experiments','contagio_rf_model'),
         'targets' : config.get('experiments', 'contagio_attack_targets'), 
         'training' : config.get('datasets', 'contagio')},}

def learn_model(scenario_name):
    '''
    Learns a classifier model for the specified scenario if one does 
    not already exist. 
    '''
    scenario = scenarios[scenario_name]
    if path.exists(scenario['model']):
        return
    
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

def attack_files_missing(attack_files):
    sys.stderr.write('Unable to locate list of attack files {}. '
                     .format(attack_files))
    sys.stderr.write(('Please list the paths to your attack files in '
                      'this file, one per line.\n'))
    sys.exit()

def gdkde_wrapper(ntuple):
    '''
    A helper function to parallelize calls to gdkde().
    '''
    try:
        return gdkde(*ntuple)
    except Exception as e:
        return e

def attack_gdkde(scenario_name, output_dir, plot=False):
    '''
    Invokes the GD-KDE attack for the given scenario and saves the resulting 
    attack files in the location specified by 'output_dir'. If plot evaluates 
    to True, saves the resulting plot into the specified file, otherwise 
    shows the plot in a window. 
    '''
    scenario = scenarios[scenario_name]
    output_dir = path.abspath(output_dir)
    utility.mkdir_p(output_dir)
    # Make results reproducible
    random.seed(0)
    # Load and print malicious files
    wolves = config.get('experiments', 'contagio_attack_pdfs')
    if not path.exists(wolves):
        attack_files_missing(wolves)
    sys.stdout.write('Loading attack samples from "{}"\n'.format(wolves))
    malicious = utility.get_pdfs(wolves)
    if not malicious:
        attack_files_missing(wolves)
    
    # Load an SVM trained with scaled data
    scaler = pickle.load(open(
                        config.get('datasets', 'contagio_scaler')))
    sys.stdout.write('Using scaler\n')
    svm = sklearn_SVC()
    sys.stdout.write('Loading model from "{}"\n'.format(scenario['model']))
    svm.load_model(scenario['model'])
    
    # Load the training data used for kernel density estimation
    sys.stdout.write('Loading dataset from file "{}"\n'
                        .format(scenario['training']))
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
    pool_args = [(svm, fname, scaler, X_train, y_train, kde_reg, 
                  kde_width, step, max_iter, False) for fname in malicious]
    
    # Perform the attack
    pyplot.figure(1)
    for res, original_file in \
            zip(pool.imap(gdkde_wrapper, pool_args), malicious):
        if isinstance(res, Exception):
            print res
            continue
        (_, fseq, _, _, attack_file) = res
        sys.stdout.write('Processing file "{}":\n'.format(original_file))
        sys.stdout.write('  scores: {}\n'
                            .format(', '.join([str(s) for s in fseq])))
        sys.stdout.write('Result: "{}"\n'.format(attack_file))
        shutil.move(attack_file, output_dir)
        pyplot.plot(fseq, label=original_file)
    
    # Plot
    pyplot.title('GD-KDE attack')
    axes = pyplot.axes()
    axes.set_xlabel('Iterations')
    axes.set_xlim(0, max_iter + 1)
    axes.set_ylabel('SVM score')
    axes.yaxis.grid()
    fig = pyplot.gcf()
    fig.set_size_inches(6, 4.5)
    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
    if plot:
        pyplot.savefig(plot, dpi=300)
    else:
        pyplot.show()

def mimicry_parallel(ntuple):
    '''
    A helper function to parallelize calls to mimicry().
    '''
    try:
        return mimicry(*ntuple)
    except Exception as e:
        return e

def attack_mimicry(scenario_name, output_dir, plot=False):
    '''
    Invokes the mimcry attack for the given scenario and saves the resulting 
    attack files in the location specified by 'output_dir'. If plot evaluates 
    to True, saves the resulting plot into the specified file, otherwise 
    shows the plot in a window. 
    '''
    scenario = scenarios[scenario_name]
    output_dir = path.abspath(output_dir)
    utility.mkdir_p(output_dir)
    # Make results reproducible
    random.seed(0)
    # Load benign files
    sys.stdout.write('Loading attack targets from file "{}"\n'
                        .format(scenario['targets']))
    target_vectors, _, target_paths = datasets.csv2numpy(scenario['targets'])
    targets = zip(target_paths, target_vectors)
    # Load and print malicious files
    wolves = config.get('experiments', 'contagio_attack_pdfs')
    if not path.exists(wolves):
        attack_files_missing(wolves)
    sys.stdout.write('Loading attack samples from file "{}"\n'.format(wolves))
    malicious = sorted(utility.get_pdfs(wolves))
    if not malicious:
        attack_files_missing(wolves)
    
    # Set up classifier
    classifier = 0
    if scenario['classifier'] == 'rf':
        classifier = RandomForest()
        sys.stdout.write('ATTACKING RANDOM FOREST\n')
    elif scenario['classifier'] == 'svm':
        classifier = sklearn_SVC()
        sys.stdout.write('ATTACKING SVM\n')
    sys.stdout.write('Loading model from "{}"\n'.format(scenario['model']))
    classifier.load_model(scenario['model'])
    
    # Standardize data points if necessary
    scaler = None
    if 'scaled' in scenario['model']:
        scaler = pickle.load(open(
                        config.get('datasets', 'contagio_scaler')))
        sys.stdout.write('Using scaler\n')
    
    # Set up multiprocessing
    pool = multiprocessing.Pool()
    pool_args = [(mal, targets, classifier, scaler) for mal in malicious]
    
    # Perform the attack
    pyplot.figure(1)
    for wolf_path, res in \
            zip(malicious, pool.imap(mimicry_parallel, pool_args)):
        if isinstance(res, Exception):
            print res
            continue
        (target_path, mimic_path, mimic_score, wolf_score) = res
        sys.stdout.write('Modifying {path} [{score}]:\n'
                            .format(path=wolf_path, score=wolf_score))
        sys.stdout.write('  BEST: {path} [{score}]\n'
                            .format(path=target_path, score=mimic_score))
        sys.stdout.write('  Moving best to {}\n\n'
                            .format(path.join(output_dir, 
                                              path.basename(mimic_path))))
        shutil.move(mimic_path, output_dir)
        pyplot.plot([wolf_score, mimic_score])
    
    # Plot
    pyplot.title('Mimicry attack')
    axes = pyplot.axes()
    axes.set_xlabel('Iterations')
    axes.set_ylabel('Classifier score')
    axes.yaxis.grid()
    fig = pyplot.gcf()
    fig.set_size_inches(6, 4.5)
    fig.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.96)
    if plot:
        pyplot.savefig(plot, dpi=300)
    else:
        pyplot.show()

