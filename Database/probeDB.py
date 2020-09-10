from ca_mysqldb import MySQLConnector

db = MySQLConnector()
out = db.execute('SELECT * FROM ca_courseware.Video_Info LIMIT 10')
print(out)
db.close()