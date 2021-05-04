import numpy as np
import matplotlib.pyplot as plt
from math import factorial


#r = np.loadtxt("time_to_run.csv", delimiter=',')
def binomial(n, k='half'):
	'''
	find binomial coefficient of n pick k,
	returns interger
	'''
	if k=='half':
		k = n//2
	return int(factorial(n)/factorial(k)**2)

data = np.array([
 [4, 0.130038],
 [6, 0.369047],
 [8, 2.051791],
 [10, 5.955474],
 [12, 33.929472],
 [14, 132.776045]])

def fit(L,alpha,beta):
	return beta*2**(alpha*L)

fit_L = np.linspace(4,14,6)
fit = fit(fit_L,2.7,0.0000006)

L = data[:,0]

pos = [binomial(n) for n in L]

print(pos)
print(data[:,1])
print(fit)

fit_L = np.linspace(4,3000,6)

plt.plot(pos,data[:,1], label='data')
plt.plot(fit_L,fit, label='fit: $6*10^{-9}* 2^{2.7 L}$')
plt.legend()
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('time to run (s)')
plt.savefig('../images/mbl_timetorun_alt.png',dpi=300)