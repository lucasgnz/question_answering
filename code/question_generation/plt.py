import numpy as np
import matplotlib.pyplot as plt
import math

SESSION = raw_input("SESSION number:")

l_test = np.load("sessions/"+str(SESSION)+"/loss_test.npy")
l_train = np.load("sessions/"+str(SESSION)+"/loss_train.npy")

print(np.exp(l_train),np.exp(l_test))

plt.plot(np.exp(l_test)[1:])
plt.plot(np.exp(l_train))

plt.show()
