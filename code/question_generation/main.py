from preprocessing import *
import numpy as np
from parameters import *

# Load and preprocess data

print('Loading and preprocessing...')

i2w = np.load('../../data/voc.npy')

(contexts, ans_locs, questions, w2i) = read_data(config['CORPUS_PATH'],
        i2w, config)



print(contexts[:100])