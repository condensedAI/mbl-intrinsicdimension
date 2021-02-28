import numpy as np
from numpy import linalg
from scipy.sparse import csr_matrix, save_npz, load_npz
from datetime import datetime
#startTime = datetime.now()

def generate_hamiltonian(size, on_site_disorder,hopping_term, set_seed=42):
    '''
    Generates Hamiltonian with random uniformly distributed,
    onsite energies, flat hopping and periodic boundary conditions,
    
    Also uses exact diagonalization to find eigenvalues and eigenvectors,
    '''
    np.random.seed(seed=set_seed)
    
    H = np.eye(size)
    onsite = on_site_disorder*(np.random.uniform(size=size)-0.5)
    H *= onsite
    hopping = np.eye(size, k=1)*hopping_term + np.eye(size, k=-1)*hopping_term
    H += hopping
    H[0,size-1] , H[size-1,0] = hopping_term, hopping_term
    (energy_levels,eigenstates)=linalg.eigh(H)
    return H, energy_levels,eigenstates

def gen_and_save(filename, size, on_site_disorder= 1.2,hopping_term= 1.5, seed=42, pickle=True):
    H, energy_levels, eigenstates = generate_hamiltonian(size,
                                                             on_site_disorder ,
                                                             hopping_term ,
                                                            set_seed = seed)
    data = np.vstack((H, energy_levels, eigenstates))
    sparse = csr_matrix(data)
    filename =filename+'_seed_'+str(seed)+'.npz'
    save_npz(filename, sparse)


sizes = [10,25,50,80,100,150,300,500,700,1000,1400,1800]
for size in sizes:
    for i in range(3):
        startTime = datetime.now()
        gen_and_save('test', size=size)
        print(datetime.now() - startTime,',')
print('Done')

