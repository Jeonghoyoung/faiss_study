import numpy as np
import pandas as pd
import sent2vec
import faiss
import util.file_util as ft

# test data
data = ft.get_all_lines('../data/ai-hub_medical.en')
df = pd.DataFrame({'data':data})

# sent2vec_test 에서 학습한 모델 load
model = sent2vec.Sent2vecModel()
model.load_model('../ai-hub_medical_en_model.bin')

# data embedding
embed_data = model.embed_sentences(data)
d = embed_data.shape[1]
print(f'Dimension size : {d}')

# faiss 의 IndexFlatL2 알고리즘 적용
faiss_index = faiss.IndexFlatL2(d)
print(f'IndexFlatL2 is trained : {faiss_index.is_trained}')

faiss_index.add(embed_data)
print(faiss_index.ntotal)

k = 3
xq = model.embed_sentence('outpatient treatment')

D, I = faiss_index.search(xq, k)
# D : 거리 , I : search된 index 를 나타내며, 모두 type 은 numpy.ndarray 이다.

for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")


# Extract numerical vectors from faiss
vecs = np.zeros((k, d))
print(vecs.shape)
for j, val in enumerate(I[0].tolist()):
    vecs[j, :] = faiss_index.reconstruct(val)
print(vecs[0][:10])





