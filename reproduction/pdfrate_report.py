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
pdfrate_report.py

Given a list of PDF files, this script will display PDFrate reports 
which are already in. 

Created on June 12, 2013
'''

from argparse import ArgumentParser
import sys

from mimicus.tools import utility
from mimicus.tools.pdfratequeryhandler import PdfrateQueryHandler

def main():
    # Setup argument parser
    parser = ArgumentParser()
    parser.add_argument('pdfs', 
                        help='Input PDFs (directory or file with path list)')
    parser.add_argument('--with-unknown', action='store_true', default=False, 
                        help='Display queries for unknown files')
    parser.add_argument('--with-waiting', action='store_true', default=False, 
                        help='Display pending queries')
    
    # Process arguments
    args = parser.parse_args()
    
    pdfs = sorted(utility.get_pdfs(args.pdfs))
    handler = PdfrateQueryHandler()
    for pdf in pdfs:
        report = handler.poll(pdf)
        if not args.with_unknown and report['status'] == 'unknown':
            continue
        if not args.with_waiting and report['status'] == 'waiting':
            continue
        sys.stdout.write('{}: {}'.format(report['filename'], report['status']))
        if report['status'] in ['success', 'nometadata']:
            r = max(report['results'].keys())
            sys.stdout.write(' [{}%]'.format(report['results'][r]
                                             ['contagio_bm']))
        sys.stdout.write('\n')
    return 0

if __name__ == "__main__":
    sys.exit(main())