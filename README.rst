================
Mimicus v1.0
================

--------------------------------------------------------
A Python library for adversarial classifier evasion 
--------------------------------------------------------

By Nedim Srndic and Pavel Laskov, University of Tuebingen.

Homepage: https://github.com/srndic/mimicus

Mimicus was used as the experimental platform for the paper:

Nedim Srndic and Pavel Laskov. **Practical Evasion of a 
Learning-Based Classifier: A Case Study**. *IEEE Symposium on 
Security and Privacy*, 2014 
(`PDF <http://www.ra.cs.uni-tuebingen.de/mitarb/srndic/srndic-laskov-sp2014.pdf>`_).

Mimicus consists of a reusable Python library (in the directory 
``mimicus/``) and code for the reproduction of experiments described in 
the paper (``reproduction/``). 



Installation and Setup
============================

Mimicus was developed in Python 2.7. Only the library files (the 
``mimicus/`` directory) are installed, code and data required for 
experiment reproduction (the ``reproduction/`` directory) are 
not installed. 

Before proceeding, please make sure you have a recent version of 
``setuptools`` (>= 3.1)::

    pip install --upgrade setuptools

You can install Mimicus directly from its Git repository::

    git clone https://github.com/srndic/mimicus.git
    cd mimicus
    python setup.py develop --user

This will install the Mimicus library for the current user and 
does not require administrative privileges. It will just create 
a link in the user's site-packages directory, usually 
``~/.local/lib/python2.7/site-packages``, to the Mimicus directory. 
That way, any modifications you make to Mimicus code will be 
immediately visible to any code importing Mimicus, so there is 
no need for reinstallation. Furthermore, because the code remains 
in the local git repository, you can easily contribute your great 
new features and bugfixes. Omit "``--user``" to install system-wide.

Alternatively, you can create a Python egg file::

    python setup.py bdist_egg

and install it for your user::

    easy_install --user dist/mimicus-*.egg

Omit "``--user``" to install system-wide.

To uninstall Mimicus, type::

    python setup.py develop --uninstall --user

Omit "``--user``" to uninstall a system-wide installation.


Required Dependencies
-------------------------

Mimicus requires the ``curl`` and ``perl`` executables to be installed::

    apt-get install curl perl
 
The following third-party Python libraries are required:

- ``matplotlib`` >= 1.1.1rc
- ``numpy`` >= 1.6.1
- ``scikit_learn`` >= 0.14.1
- ``scipy`` >= 0.9.0

They will be automatically installed by ``setuptools`` or you can 
install them manually using ``pip``::

    pip install matplotlib
    pip install numpy
    pip install scikit_learn

There might be problems with ``scipy`` installation if you do not 
already have ``BLAS`` installed. You can install ``scipy`` by following 
these `directions <http://www.scipy.org/install.html>`_.


Optional Dependencies
---------------------------

Mimicus provides two different implementations of the Random Forest 
classifier: 

1. ``R_randomForest``, using the ``randomForest`` package for the R 
   programming language, and 
2. the ``RandomForestClassifier`` class of ``scikit_learn``. 

If you wish to use the former, please install R, its ``randomForest`` 
package and the ``rpy2`` Python library. Otherwise, the 
``scikit_learn`` implementation will be used as a fallback. The R 
version is maintained because it is the one used by PDFrate. 
The ``mimicus.classifiers.RandomForest`` module decides during runtime 
which implementation to use, depending on whether you have the R 
implementation installed or not.


Setting up PDFrate Submissions
---------------------------------

Before submitting files to PDFrate, please read the `policies
<http://pdfrate.com/policies>`_.

In order to respect the PDFrate policies and minimize the number 
of submissions, submissions are scheduled to run periodically and 
individually, and PDFrate's replies are cached. 

New submissions are stored as JSON files in a query directory. The 
script ``mimicus/bin/pdfratequeryscheduler.py`` runs periodically and 
submits the query with the highest priority or, if there are 
multiple, the oldest one. The script will then 
query PDFrate to check any pending queries and save the reply, 
if it is ready, into the replies directory. The reply remains 
in the replies directory and is subsequently returned every time 
a script submits the same file to PDFrate, i.e., there is no 
danger of multiple submission. 

In order for this to work, please schedule the submission script to 
run in regular time intervals (e.g, using cron) and set up the 
query and reply directories in the Mimicus configuration file 
(see `Configuration Files`_).

Reproduction of Experiments
======================================

If you wish to reproduce the experiments described in the paper, 
you will find that everything is included in this project except 
the malicious attack candidate files.


Attack Files
---------------------------

Files from the Contagio dataset were used in the experiments 
described in the paper and we cannot distribute them. They are 
available `here 
<http://contagiodump.blogspot.de/2010/08/malicious-documents-archive-for.html>`_.

The attack files comprise the dataset called ``Attack``. A full list 
of files in the ``Attack`` dataset can be found in 
``data/attack.list``. They can be found under the same 
names in the Contagio repositories. 

If you wish to run the attacks using a different set of malicious 
attack candidate files, you can replace the attack.list file with 
your own list. 


Running Experiments
------------------------------

Experiments can be reproduced by running these scripts in the 
``reproduction/`` directory, one per attack scenario::

    python reproduction/F.py
    python reproduction/FC.py
    python reproduction/FT.py
    python reproduction/FTC.py


Submitting Files to PDFrate
--------------------------------

Before submitting files to PDFrate, please read the `policies
<http://pdfrate.com/policies>`_.

You can submit a directory of PDF files or PDF files listed in a 
text file using the ``reproduction/pdfrate_submit.py`` script, 
e.g.::

    python reproduction/pdfrate_submit.py results/F_mimicry

To print submission results when they are ready, use the 
``reproduction/pdfrate_report.py`` script, e.g.:: 

    python reproduction/pdfrate_report.py results/F_mimicry

See `Setting up PDFrate Submissions`_ if you haven't 
already configured PDFrate submissions.


Configuration Files
===============================

There are two configuration files in this project: one for the 
Mimicus library and the other for the reproduction code. Both 
files use the same `INI-file-like syntax 
<http://docs.python.org/2.7/library/configparser.html>`_.


Mimicus Library Configuration File
--------------------------------------

After the installation or the first time you run an attack, the 
directory ``$XDG_CONFIG_HOME/mimicus``, e.g., ``~/.config/mimicus``, 
will be created with the configuration file ``mimicus.conf`` inside. 
Use it to customize your library installation. 
Options are described in the ``mimicus/default.conf`` file. 


Reproduction Configuration File
-----------------------------------

The first time you run an attack, 
the configuration file ``reproduction/custom.conf`` will be created. 
Use it to customize the execution of experiments. Options are 
described in the ``reproduction/default.conf`` file. 


Project Layout
=======================

* ``mimicus/`` - Python package mimicus (library)

  - ``attacks/`` - attack method implementations
  - ``bin/`` - scripts
  - ``classifiers/`` - classifier implementations
  - ``data/`` - data files required for testing the library
  - ``test/`` - code for testing the library
  - ``tools/`` - code for feature extraction, etc.

* ``results`` - attack results will be saved in this directory

* ``reproduction/`` - Python code for experiment reproduction
* ``data/`` - data files required to reproduce the experiments

* ``COPYING`` - software license
* ``MANIFEST.in`` - Python setuptools configuration
* ``README`` - this file


Licensing
==============

Mimicus is free software: you can redistribute it and/or modify it 
under the terms of the `GNU General Public License 
<http://www.gnu.org/licenses/gpl.html>`_ as published by 
the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version. 

