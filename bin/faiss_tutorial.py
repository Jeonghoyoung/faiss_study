import numpy as np
import faiss

# two matrices 'xb' and 'xq'

d = 64
nb = 100000    # database size
nq = 10000     # nb of queries

# torch.manual_seed(123)
np.random.seed(123)

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Building an index and adding the vectors to it
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(xb)
print(faiss_index.ntotal)
print(faiss_index.is_trained)

# Searching
k = 4
distance_b, index_b = faiss_index.search(xb[:5], k)
distance_q, index_q = faiss_index.search(xq[:5], k)
print(f'Database xb distance:\n{distance_b}')
print(f'Database xb index:\n{index_b}')

print(f'Queries xq distance:\n{distance_q}')
print(f'Queries xq index:\n{index_q}')

