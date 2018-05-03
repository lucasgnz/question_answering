import sqlite3
import os
import csv
from uuid import uuid1

with open('../../data/tsv/sec_raw.tsv', 'w') as output:
    writer = csv.writer(output, delimiter='\t')
    for element in os.listdir('../../data/sql/'):
        if element.endswith('.db') and element[0] != ".":
            db = sqlite3.connect('../../data/sql/'+element)
            db.text_factory = str
            cursor = db.cursor()
            cursor.execute('''SELECT * FROM documents''')
            rows = cursor.fetchall()

            for row in rows:
                doc = row[1].decode('utf8')
                printable = set(string.printable)
                doc = filter(lambda x: x in printable, doc)
                writer.writerow([uuid1(), doc])