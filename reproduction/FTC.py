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
FTC.py

Reproduction of scenario FTC.

Created on March 5, 2014.
'''

from argparse import ArgumentParser
import sys

from common import attack_mimicry

def main():
    scenario_name = 'FTC'
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('output_dir', help='Where to save best mimics')
    parser.add_argument('--plot', default=False, help='Where to save plot (file name)')
    args = parser.parse_args()
    
    # Perform the attack
    attack_mimicry(scenario_name, args.output_dir, args.plot)
    return 0

if __name__ == '__main__':
    sys.exit(main())
