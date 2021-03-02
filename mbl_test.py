'''import sys
for path in sys.path:
     print(path)
 
sys.path.append('/abc')
'''


from datetime import datetime
import matplotlib.pyplot as plt
from numpy import load
from mbl import *
buildnDiag(method ='dense', L_high=10, disorder_realizations=10, disorders=5)


#rs = rStatFromFiles(L_high = 14,disorder_realizations = 1000, disorders = 20)

#print(rs)
#print(np.shape(rs))

#plotR(rs)

#print(load_eigs_npz('results/results-L-4-W-0.1-seed-1.npz'))
