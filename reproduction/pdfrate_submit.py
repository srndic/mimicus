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
pdfrate_submit.py

Given a list of PDF files in a file or a directory, this script will 
submit those files to PDFrate. 

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
    parser.add_argument('--priority', type=int, default=0, 
                        help='Submission priority')
    
    # Process arguments
    args = parser.parse_args()
    
    pdfs = sorted(utility.get_pdfs(args.pdfs))
    sys.stdout.write("Submitting files to PDFrate:\n\n")
    handler = PdfrateQueryHandler()
    for i, pdf in enumerate(pdfs):
        handler.submit_query(pdf, get_metadata=True, priority=args.priority)
        sys.stderr.write('{d} File {i}/{n}: {f} {d}\n'
                         .format(d='-'*10, i=i+1, n=len(pdfs), f=pdf))
    return 0

if __name__ == "__main__":
    sys.exit(main())