from scipy.sparse import eye, diags, spmatrix, linalg, save_npz
from numpy import savez_compressed, random
from datetime import datetime
startTime = datetime.now()


def gen_sparse_Hamiltonian(size=4, hopping=1.5, disorder_strength=1.2, seed=42):
    '''
    Build a sparse matrix (Hamiltonian) and populates it with onsite energies and hopping terms
    Returns: sparse Hamiltonion
    '''

    onsite = (random.uniform(size=size)-0.5) * disorder_strength

    sparse_onsite = diags(onsite)

    sparse_hop_over = eye(size, n=None, k=1, dtype=float, format=None)*hopping
    sparse_hop_under = eye(size, n=None, k=-1, dtype=float, format=None)*hopping

    

    sparse_Hamiltonian = sparse_onsite + sparse_hop_over + sparse_hop_under

    sparse_Hamiltonian[0,-1] = hopping
    sparse_Hamiltonian[-1,0] = hopping

    #print(spmatrix.todense(sparse_Hamiltonian))
    return sparse_Hamiltonian

def diag_sparse(sparse_hamiltonian, k=3):
    (eigvals, eigvecs) =  linalg.eigsh(sparse_hamiltonian, k=k)
    return eigvals,eigvecs.T

def build_solve_save(filename, k=3, size=4, hopping=1.5, disorder_strength=1.2, seed=42):
    H_sparse = gen_sparse_Hamiltonian(size, hopping, disorder_strength, seed)
    eigvals,eigvecs = diag_sparse(H_sparse, k)
    
    save_npz(filename+str(seed)+'H.npz', H_sparse, compressed=True)
    savez_compressed(filename+str(seed)+'eigvals.npz')
    savez_compressed(filename+str(seed)+'eigvecs.npz')

sizes = [10,25]#,100,300,700,1400,2500,8000,20000,40000]
for size in sizes:
    for i in range(5):
        startTime = datetime.now()
        build_solve_save('test',k=5, size=size)
        print(datetime.now() - startTime,',')
print('Done')