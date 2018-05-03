import matplotlib.pyplot as plt
import numpy as np

att=np.load("attention.npy")

print(att)
k=1
for (attention_, words, generated_word) in att:
	plt.subplot(len(att),1,k)
	x = np.arange(len(words))
	plt.bar(x,attention_, align='center', alpha=0.5)
	plt.xticks(x, words)
	plt.title(generated_word, loc="right")
	k+=1

plt.show()
