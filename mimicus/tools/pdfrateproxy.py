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
pdfrateproxy.py

Created on Mar 11, 2013
'''

import json
import subprocess
import sys

from mimicus import config

class PdfrateProxyError(IOError):
    '''
    A class representing an exception when communicating with PDFrate.
    '''
    
    def __init__(self, message, *args, **kwargs):
        self.message = message
        IOError.__init__(self, *args, **kwargs)
    
    def __str__(self):
        return self.message

class PdfrateProxy(object):
    '''
    A class representing a local proxy object for PDFrate. Submit your 
    PDFrate queries to this class and it will run them against PDFrate. 
    It's not designed be used directly, but by a query scheduler. For 
    batch or manual submissions, use the PdfrateQueryHandler class. 
    '''

    def __init__(self):
        '''
        Constructor. 
        '''
        self.submit_url = config.get('pdfrateproxy', 'submit_url')
        self.report_url = config.get('pdfrateproxy', 'report_url')
        self.metadata_url = config.get('pdfrateproxy', 'metadata_url')
    
    def _invoke_curl(self, url, infile=None):
        '''
        Runs a child curl process that communicates to PDFrate.
        '''
        args = ['curl']
        if infile is not None:
            args.extend(['--form', 'filesubmission=@{file}'.format(file=infile.encode('utf-8'))])
        args.extend([url])
        sys.stderr.write("run$ {args}\n".format(args=' '.join(args)))
        curl = subprocess.Popen(args, stdout=subprocess.PIPE)
        (stdoutdata, stderrdata) = curl.communicate()
        if curl.returncode != 0 and stderrdata is not None and len(stderrdata) > 0:
            raise PdfrateProxyError(stderrdata)
        return stdoutdata
    
    def submit_file(self, infile):
        '''
        Submits a PDF file to PDFrate and returns the JSON result as a Python dictionary.
         The dictionary has a field 'status' indicating whether the submission was successful
         ('success') or not ('failsubmit').
        '''
        reply = json.loads(self._invoke_curl(self.submit_url, infile))
        reply['status'] = 'success' if reply['fileinfo']['size'] > 0 else 'failsubmit'
        return reply
    
    def get_report(self, search_hash):
        '''
        Searches on PDFrate for the classification report (in form of a Python dictionary 
        converted from a JSON object) of a previously submitted PDF file with 
        the given SHA256 hash. The dictionary has a field 'status' indicating if the 
        report for this hash exists ('success') or not ('noreport').
        '''
        reply = json.loads(self._invoke_curl(self.report_url.format(search_hash=search_hash)))
        reply['status'] = 'success' if reply['fileinfo']['size'] > 0 else 'noreport'
        return reply
    
    def get_metadata(self, search_hash):
        '''
        Searches on PDFrate for the metadata (a string) of a previously submitted PDF 
        file with the given SHA256 hash. Returns a dictionary with two fields:
          'metadata' - the metadata itself
          'status' a boolean indicating if the metadata for this hash exists.
        '''
        reply = {'metadata':self._invoke_curl(self.metadata_url.format(search_hash=search_hash)).strip()}
        reply['status'] = 'success' if len(reply['metadata']) > 0 else 'nometadata'
        return reply
