# Single particle Anderson localization 1D

So we have a chain of atoms at fixed positions. The on-site energy at each site is drawn randomly from a uniform distribution.

The ends of the chain are connected, as to impose periodic boundary conditions.


## The program works as follows:

* I initiate an L x L-Matrix, i.e. the Hamiltonian
** on the diagonal I place the on-site energies
** on the over/under-diagonals i insert the hopping term.
** in the lower-left and upper-right corners of the Hamiltonian.
* I continue to find the eigen-values and -vectors.
* I determine the spread of the absolute squared eigenvectors
* I plot spread (1/localization) as a function of disorder 
