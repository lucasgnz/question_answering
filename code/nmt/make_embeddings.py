import unicodecsv as csv
import numpy as np
from preprocessing import *


c=0
with open("../../data/glove.txt", "r") as input, open("../../data/glove_ans.txt", "w") as output:
  reader = csv.reader(input, delimiter=' ')
  writer = csv.writer(output, delimiter=' ')
  print('Total: ',sum(1 for row in reader))
  input.seek(0)
  writer.writerow(["<unk_0>"] + [0 for _ in range(301)])
  for row in reader:
    writer.writerow([row[0] + "_0"] + row[1:] + [0])
    writer.writerow([row[0] + "_1"] + row[1:] + [1])
    print(c)
    c+=1

print('Done!')