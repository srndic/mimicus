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
pdfratequeryscheduler.py

A program representing a scheduler of queries to PDFrate. 

It takes query files saved in the 'query_dir' directory, schedules their 
submission and submits them in order of arrival. It saves query replies 
as files into the 'reply_dir' directory. 

'query_dir' and 'reply_dir' are configurable through the configuration 
file. You can find the technical 
description of query files and reply files in pdfratequeryhandler.py. 

Schedule this program to run regularly. 

Created on Mar 22, 2013
'''

import datetime
import json
from operator import itemgetter
import os
from random import randint
import sys
from time import sleep

from argparse import ArgumentParser

from mimicus import config
from mimicus.tools import pdfrateproxy, utility

def main():
    sys.stdout.write('PDFrate Query Scheduler running! [{0}]\n'.format(datetime.datetime.now()))
    parser = ArgumentParser(description = 'PDFrate Query Scheduler')
    parser.parse_args()
    
    QUERY_DIR = config.get('pdfratequeryscheduler', 'query_dir')
    REPLY_DIR = config.get('pdfratequeryscheduler', 'reply_dir')
    
    queries = []
    max_priority = 0
    sys.stdout.write('Queries found: ')
    # Process all files in the QUERY_DIR
    for f in os.listdir(QUERY_DIR):
        f = os.path.join(QUERY_DIR, f)
        if not os.path.isfile(f) or os.path.splitext(f)[1] != '.json':
            continue
        try:
            queries.append(json.load(open(f, 'r')))
        except Exception as ex:
            sys.stderr.write('Error reading query file \'{f}\': {ex}\n'.format(f=f, ex=ex))
            continue
        # Keep track of max priority
        queries[-1]['queryfile'] = f
        if queries[-1]['priority'] > max_priority:
            max_priority = queries[-1]['priority']
        #sys.stdout.write('{0}\n'.format(queries[-1]))
    
    # In case of no queries
    if not queries:
        sys.stdout.write("None\nExiting.\n")
        return
    else:
        sys.stdout.write('{}\n'.format(len(queries)))
    
    # Filter for max priority queries
    sys.stdout.write('Max priority: {0}\n'.format(max_priority))
    if max_priority != 0:
        queries = [q for q in queries if q['priority'] == max_priority]
    # The oldest one is next
    top_query = min(queries, key=itemgetter('datetime'))
    del queries
    sys.stdout.write('Next query: {0}\n'.format(top_query))
    
    # Submit query to PDFrate and save the reply
    proxy = pdfrateproxy.PdfrateProxy()
    sleep_time = randint(0, int(config.get('pdfratequeryscheduler', 'sleep_time')))
    sys.stdout.write('Sleeping for {0} seconds...\n'.format(sleep_time))
    sleep(sleep_time)
    sys.stdout.write('Getting report...\n')
    reply = proxy.get_report(utility.file_sha256_hash(top_query['filename']))
    if reply['status'] == 'noreport':
        sys.stdout.write('No report, submitting file...\n')
        reply = proxy.submit_file(top_query['filename'])
    
    if top_query['get_metadata'] == True and reply['status'] == 'success':
        # Also get metadata
        file_hash = os.path.splitext(os.path.basename(top_query['queryfile']))[0]
        sys.stdout.write('Getting metadata...\n')
        metadata_reply = proxy.get_metadata(file_hash)
        reply['metadata'] = metadata_reply['metadata']
        reply['status'] = metadata_reply['status']
    
    reply_filename = os.path.join(REPLY_DIR, os.path.basename(top_query['queryfile']))
    reply['filename'] = top_query['filename']
    sys.stdout.write('Writing reply to disk...\n')
    json.dump(reply, open(reply_filename, 'w+'))
    # Remove query file
    sys.stdout.write('Removing query file...\n')
    os.remove(top_query['queryfile'])
    sys.stdout.write('Exiting.\n')

if __name__ == '__main__':
    main()