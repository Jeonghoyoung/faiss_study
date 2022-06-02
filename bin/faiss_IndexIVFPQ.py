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
print(embed_data.shape[1])
d = embed_data.shape[1]

# Use IndexIVFPQ
nlist = 50 # index에 포함할 파티션 수를 지정
m = 8 # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid

quant = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quant, d, nlist, m, bits)

index.train(embed_data)
print(index.is_trained)

index.add(embed_data)

k = 3
xq = model.embed_sentence('outpatient treatment')

# .nprobe를 통해 검색범위를 늘려 정확도 향상.
index.nprobe = 10
D, I = index.search(xq, k)
print(I)
for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")

