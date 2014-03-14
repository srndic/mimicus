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

A module that parses the configuration file 'custom.conf' 
and exposes an object called 'config'. Include this module for automatic 
parsing of configuration options. 

Created on March 26, 2013.
'''

import ConfigParser
import os
import sys

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
    Parses the configuration file 'custom.conf' if it exists, otherwise 
    creates it by copying 'default.conf'. 
    '''
    # default.conf and custom.conf must reside in the same directory as this file
    config_dir = os.path.dirname(__file__)
    custom_conf = os.path.join(config_dir, 'custom.conf')
    
    # If it's missing, generate the custom.conf file
    custom_conf_created = False
    if not os.path.exists(custom_conf):
        default_conf = os.path.join(config_dir, 'default.conf')
        try:
            open(custom_conf, 'wb').write(open(default_conf).read())
            custom_conf_created = True
        except:
            sys.stderr.write('Error: Unable to create file "custom.conf".')
            sys.exit(1)
    
    # Parse the configuration
    global config
    config.readfp(open(custom_conf))
    
    # Try using the directory of this file as the project root on first run
    if custom_conf_created:
        project_dir = os.path.dirname(config_dir)
        config.set('DEFAULT', 'project_root', project_dir)
        config.write(open(custom_conf, 'wb'))
    
    # A naive way to check if the configuration file was customized
    if (config.get('DEFAULT', 'project_root') == '/dev/null' or 
            not os.path.exists(config.get('DEFAULT', 'project_root'))):
        sys.stderr.write(('{d}\n\n'
            'Please customize your configuration file "{f}". See the README '
            'file for more details.'
            '\n\n{d}\n').format(f = custom_conf, d = '#' * 80))
        sys.exit(1)

parse_config()
