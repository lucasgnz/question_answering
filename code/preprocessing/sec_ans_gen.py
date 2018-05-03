import spacy
import csv
from nltk.tokenize import word_tokenize
nlp = spacy.load('en')

#Load SEC TSV (id, context)
#Output TSV (id, context, answerspan)

with open('../../data/tsv/sec_ans.tsv', 'w') as output:
    writer = csv.writer(output, delimiter='\t')
    with open('../../data/tsv/sec_raw.tsv', 'w') as input:
        reader = csv.reader(input, delimiter='\t')
        for row in reader:
            doc = nlp(row[1])
            for ent in doc.ents:
                tokens = word_tokenize(doc)
                ans = word_tokenize(ent.text)
                i = 0
                while tokens[i] != ans and i < len(tokens):
                    i += 1
                if i < len(tokens):
                    print(ent.label_, ent.text)
                    writer.writerow([row[0], row[1], str(i)+":"+str(i+len(ans))])

print('Done!')