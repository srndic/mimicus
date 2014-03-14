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
pdfratequeryhandler.py

Created on Mar 21, 2013
'''

import json
import os
import time

from mimicus import config
from mimicus.tools import utility

class PdfrateQueryHandler(object):
    '''
    A class representing a simple interface for submitting files 
    to PDFrate and querying the results. 
    
    It enables reliable (optionally anonymized) communication with 
    PDFrate and randomly schedules queries in time. Use this class 
    for your submissions. 
    
    PDFrate queries for file submission and report retrieval return JSON 
    reports such as this one:
    
    {
        "report_url":"http:\/\/pdfrate.com\/view\/00a544f460df17c90481cef1a7399ae1db6e0efe963de2bbd1060915990a799c",
        "fileinfo":{
            "type":"pdf",
            "size":"16328",
            "md5":"f5bebcfdc4a48369c973b2560a6b58e6",
            "sha1":"7d8ba4b17cf2d63c7c9d84d61daa4912ac69e3cd",
            "sha256":"00a544f460df17c90481cef1a7399ae1db6e0efe963de2bbd1060915990a799c",
            "submitted":"2012-10-23 03:37:38"
        },
        "votes":{
            "ben":0,
            "mal":1,
            "opp":1,
            "tar":0
        },
        "results":{
            "2013-01-24 03:51:49":{
                "contagio_bm":"89.7",
                "contagio_ot":"24",
                "gmu_bm":"100",
                "gmu_ot":"0",
                "pdfrate_bm":"89.1",
                "pdfrate_ot":"3.5"
            },
            "2012-10-23 03:37:40":{
                "contagio_bm":"89.7",
                "contagio_ot":"24",
                "gmu_bm":"100",
                "gmu_ot":"0"
            }
        }
    }
    
    These JSON replies are then converted into Python dictionaries and 
    an additional top-level field 'success' is inserted with a boolean 
    value indicating if the query was performed successfully, i.e., file 
    submission was successful or the report was retrieved. 
    
    PDFrate queries for metadata retrieval return strings such as this one:
    
    00000000             Size: 1170
    00000000           Header: %PDF-1.4
    000002E7     CreationDate: D:20130320133300+01'00'
    00000236           Author: .N.e.d.i.m. .S.r.n.d.i.c
    00000274          Creator: .W.r.i.t.e.r
    0000029B         Producer: .L.i.b.r.e.O.f.f.i.c.e. .3...4
    000003FE           PdfID0: 0AB57027E1CDF7B143A15EF3EAF4FFC6
    000003FE           PdfID1: 0AB57027E1CDF7B143A15EF3EAF4FFC6
    00000116              Box: 612x792 (letter)
    0000019A              Box: 612x792 (letter)
    0000002A           Filter: /FlateDecode
    00000017        Structure: obj
    00000025        Structure: 3 0 R
    00000040        Structure: stream
    0000006C        Structure: endstream
    00000076        Structure: endobj
    00000082        Structure: obj
    00000089        Structure: endobj
    00000095        Structure: obj
    0000009F        Structure: endobj
    000000AB        Structure: obj
    000000B1        Structure: /Font
    000000B7        Structure: 5 0 R
    000000D4        Structure: endobj
    000000E0        Structure: obj
    000000EB        Structure: /Page
    000000F8        Structure: 4 0 R
    00000108        Structure: 6 0 R
    0000015A        Structure: 2 0 R
    00000162        Structure: endobj
    0000016E        Structure: obj
    0000018B        Structure: 6 0 R
    000001B1        Structure: 1 0 R
    000001C4        Structure: endobj
    000001D0        Structure: obj
    000001EA        Structure: 4 0 R
    000001F0        Structure: /OpenAction
    000001FC        Structure: 1 0 R
    00000224        Structure: endobj
    00000230        Structure: obj
    00000310        Structure: endobj
    00000318        Structure: xref
    000003D5        Structure: trailer
    000003EC        Structure: 7 0 R
    000003F8        Structure: 8 0 R
    0000047E        Structure: startxref
    0000048D        Structure: %EOF
    
    These are then saved in the 'metadata' field of a Python dictionary 
    and returned as result. An additional field 'success' is inserted with 
    a boolean value indicating if the query was performed successfully, 
    i.e., metadata was retrieved. 
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.query_dir = config.get('pdfratequeryscheduler', 'query_dir')
        self.reply_dir = config.get('pdfratequeryscheduler', 'reply_dir')
    
    def _get_query_for_file(self, filename):
        '''
        Given a file, generates the path to the query file for it. 
        '''
        sha256 = utility.file_sha256_hash(filename)
        return os.path.join(self.query_dir, '{0}.json'.format(sha256))
    
    def _get_reply_for_file(self, filename):
        '''
        Given a file, generates the path to the reply file for it. 
        '''
        sha256 = utility.file_sha256_hash(filename)
        return os.path.join(self.reply_dir, '{0}.json'.format(sha256))
    
    def submit_query(self, filename, get_metadata=False, force=False, priority=0):
        '''
        Submits the specified file to PDFrate and returns the query dictionary if 
        it was a new submission, otherwise returns an empty dictionary. 
        
        If the file was already submitted in the past, it does nothing. You 
        can force it to resubmit a previously submitted file by setting 
        'force' to True. By default, it will only return a classification 
        report from PDFrate. If you want it to also return file metadata, 
        set 'get_metadata' to True. You can set a scheduling priority for 
        this file using the 'priority' argument. 
        
        This method will submit the file to PDFrate asynchronously and return 
        control to the caller. It will not wait for the results from PDFrate. 
        In order to get the PDFrate results, use the poll() method of this 
        class. You should poll it from time to time, as it may take some time 
        for the results to get in, depending on the settings of 
        pdfratequeryscheduler. 
        
        A query is a JSON object representing a Python dictionary such as this one:
         
        {
            'datetime' : UNIX timestamp when the query was *first* submitted; used for scheduling on a first-come, first-served basis
            'get_metadata' : a boolean indicating whether to retrieve file metadata from PDFrate or not
            'filename' : the location of the file to submit on disk
            'priority' : a number used as a priority for scheduling in a priority queue, highest priority scheduled first, overrides the timestamp-based priority
        }
        '''
        queryfile = self._get_query_for_file(filename)
        if os.path.exists(queryfile):
            # Already scheduled for submission, update the query info
            query = json.load(open(queryfile, 'r'))
            updated = False
            if query['get_metadata'] != get_metadata:
                query['get_metadata'] = get_metadata
                updated = True
            if query['priority'] != priority:
                query['priority'] = priority
                updated = True
            if updated:
                json.dump(query, open(queryfile, 'w+'))
            return {}
        
        replyfile = self._get_reply_for_file(filename)
        if force == False and os.path.exists(replyfile):
            # Reply already exists
            if not get_metadata:
                # Metadata not requested, nothing to do
                return {}
            # Check if this is a new request for metadata
            reply = json.load(open(replyfile, 'r'))
            if 'metadata' in reply:
                # Nope, already retrieved
                return {}
        
        # Submit file
        query = {'datetime':int(time.mktime(time.localtime()))}
        query['get_metadata'] = get_metadata
        query['filename'] = filename
        query['priority'] = priority
        json.dump(query, open(queryfile, 'w+'))
        return query
    
    def poll(self, filename):
        '''
        Returns the PDFrate reply for the given file. 
        
        The reply is a dictionary such as the following:
        
        {
            'status' : the only field guaranteed to be present, one of the following values
                'success' - reply successfully obtained
                'waiting' - waiting for the scheduler to submit the query or PDFrate to return a reply
                'failsubmit' - problems with submitting this file to PDFrate, check the connection
                'noreport' - file not previously submitted to PDFrate, no report found
                'nometadata' - file not previously submitted to PDFrate, no metadata found
                'unknown' - there is no record of this file being submitted (maybe you never submitted it or a serious error occurred and it was lost)
            'report_url' : see class documentation for an example (present if file was submitted or report was requested but only on success)
            'fileinfo' : see class documentation for an example (present if file was submitted or report was requested but only on success)
            'votes' : see class documentation for an example (present if file was submitted or report was requested but only on success)
            'results' : see class documentation for an example (present if file was submitted or report was requested but only on success)
            'metadata' : see class documentation for an example (present if metadata was requested but only on success)
            'filename' : the name of the original file that was submitted
        }
        '''
        replyfile = self._get_reply_for_file(filename)
        if not os.path.exists(replyfile):
            queryfile = self._get_query_for_file(filename)
            if not os.path.exists(queryfile):
                return {'status':'unknown'}
            else:
                return {'status':'waiting'}
        else:
            reply = json.load(open(replyfile, 'r'))
            if 'results' in reply:
                for result in reply['results'].keys():
                    for classifier in reply['results'][result].keys():
                        reply['results'][result][classifier] = float(reply['results'][result][classifier]) / 100.0
            return reply
