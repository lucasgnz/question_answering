import sqlite3

docid = '33-10467'

db = sqlite3.connect('../../data/sql/'+docid+'.db')
db.text_factory = str
cursor = db.cursor()
cursor.execute('''SELECT * FROM documents''')
rows = cursor.fetchall()

full_text = ""
for row in rows:
    full_text += row[1]+"\n"
print(full_text)