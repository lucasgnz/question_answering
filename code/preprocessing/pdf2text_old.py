import textract
import sqlite3
import sys




name = raw_input("Entrer un nom pour ce document: ")

text=textract.process(sys.argv[1])
print(text)

db = sqlite3.connect("../../data/"+name+".db")
qc = db.cursor()


qc.execute("CREATE TABLE documents(id varchar, text longtext)")
qc.execute('''INSERT INTO documents VALUES(?,?)''',(name,sqlite3.Binary(text)))

qc.close()