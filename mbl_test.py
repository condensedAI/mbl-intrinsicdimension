'''import sys
for path in sys.path:
     print(path)
 
sys.path.append('/abc')
'''

from mbl import *
from datetime import datetime
import matplotlib.pyplot as plt
'''
number = 5
L = 4
bitstring = binaryConvert(number, L)
print(number, bitstring)


ones = nOnes( bitstring)
print('nOnes = ',  ones)


for i in range(2,6):
	print(binomial( i))
'''

#bs = basisStates(6)
#print(bs)

#ed = energyDiagonal()
#print(ed)

#H = constructHamiltonian(L=6, method='dense')
#H1 = constructHamiltonian(L=6, method='sparse')
#print(H_sparse)

#H_dense = constructHamiltonianDense(L=6)
#eig0 = diag(H)
#eig1 = diag(H1)

#eig1 = diagSparse(H_sparse)
#print(H)
#print(eig0)
#print(eig1)
#ls = levelSpacing(eig0) 
#print(ls)

benchmark(1000  ,8)

#obtain_s_r(save=False)
