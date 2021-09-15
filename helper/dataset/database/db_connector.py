#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymysql.cursors
import base64
from pathlib import Path
import os

# Store credentials in home .hca
CRED_DIR = os.path.join(str(Path.home()), ".flipped_classroom")

# Parse credentials
with open(CRED_DIR,'r') as f:
	[Username, Password] = f.read().strip().split('\n')[:2]
	Password = Password.encode('ascii')
HostAddress = 'cedegemac8.epfl.ch'


class MySQLConnector:

	# Initialiser
	def __init__(self):
		self.ReturnedRows = []
		self.Connector = pymysql.connect(user=Username, password=base64.b64decode(Password).decode('ascii'), host=HostAddress, charset='utf8mb4', use_unicode=True)

	# Destructor
	def close(self):
		self.Connector.close()

	# Executes an SQL query and returns its ouput (if executed through command line) #
	def execute(self, SQLQuery, ReturnCursor=False):
		# Initialise cursor
		self.Cursor = self.Connector.cursor()
		# Execute and commit SQL query
		self.Cursor.execute(SQLQuery)
		self.Connector.commit()
		# Return cursor if requested
		if ReturnCursor:
			return self.Cursor
		# Fill in list with returned rows
		fields = [x[0] for x in self.Cursor.description]

		self.ReturnedRows = []
		while True:
			Row = self.Cursor.fetchone()
			if Row is None:
				break
			else:
				self.ReturnedRows.append(Row)
		# Return list of returned rows
		return self.ReturnedRows, fields

	# Return list of returned rows
	def getRows(self):
		return self.ReturnedRows
