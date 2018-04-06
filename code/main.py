#PIPELINE
#
"""
load doc
NER + coref + ELMo embeddings + POS
LSTM
=> candidats

Question / candidats analysis
=> rerank + merge => answer


"""


import re

regex = '(?P<lettre>[a-zA-Z])\.\d+ '
html = re.sub(regex, '\g<lettre>. ', html)

print(html)