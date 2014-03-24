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
config.py

A module that parses the configuration file '~/.config/mimicus/mimicus.conf' 
and exposes an object called 'config'. Include this module for automatic 
parsing of configuration options. 

Created on March 26, 2013.
'''

import ConfigParser
import errno
import os
import sys

def _mkdir_p(path):
    '''
    Creates a directory regardless whether it already exists or not.
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

'''
The configuration object of type SafeConfigParser. Use it to get() 
options from the configuration file. 
'''
config = ConfigParser.SafeConfigParser()

'''
Helper function to reduce typing effort.
'''
get = config.get

def parse_config():
    '''
    Parses the configuration file '$XDG_CONFIG_HOME/mimicus/mimicus.conf' if it 
    exists, otherwise creates it by copying 'default.conf'. 
    '''
    project_root = os.path.dirname(__file__)
    conf_parent = os.environ['XDG_CONFIG_HOME'] if 'XDG_CONFIG_HOME' in \
                os.environ else os.path.expanduser('~/.config')
    conf_root = os.path.join(conf_parent, 'mimicus')
    custom_conf = os.path.join(conf_root, 'mimicus.conf')
    
    # Generate the configuration file if it's missing
    custom_conf_created = False
    if not os.path.exists(custom_conf):
        # Create the conf_root directory
        try:
            _mkdir_p(conf_root)
        except:
            sys.stderr.write('Unable to create directory {}'.format(conf_root))
            sys.exit(1)
        # Copy default.conf to the custom configuration file
        try:
            default_conf = os.path.join(project_root, 'default.conf')
            open(custom_conf, 'wb').write(open(default_conf).read())
            custom_conf_created = True
        except:
            sys.stderr.write('Error: Unable to create file "{}".'
                             .format(custom_conf))
            sys.exit(1)
    
    # Parse the configuration
    global config
    config.readfp(open(custom_conf))
    
    # Try using the directory of this file as the project root on first run
    if custom_conf_created:
        data_root = os.path.join(project_root, 'data')
        config.set('DEFAULT', 'data_root', data_root)
        config.set('DEFAULT', 'conf_root', conf_root)
        _mkdir_p(config.get('pdfratequeryscheduler', 'query_dir'))
        _mkdir_p(config.get('pdfratequeryscheduler', 'reply_dir'))
        config.write(open(custom_conf, 'wb'))
    
    # A naive way to check if the configuration file was customized
    if (config.get('DEFAULT', 'data_root') == '/dev/null' or 
            not os.path.exists(config.get('DEFAULT', 'data_root'))):
        sys.stderr.write(('{d}\n\n'
            'Please customize your configuration file "{f}". See the README '
            'file for more details.'
            '\n\n{d}\n').format(f = custom_conf, d = '#' * 80))
        sys.exit(1)

parse_config()
