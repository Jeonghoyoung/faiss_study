import sent2vec

'''
Train Command : ./fasttext sent2vec -input ../fasttext_test.txt -output test_model -dropoutK 0 -dim 200 -epoch 9 -lr 0.2 -thread 10 -bucket 100000
'''
# Model Load
model = sent2vec.Sent2vecModel()
model.load_model('ai-hub_medical_en_model.bin')

# Information Embeddings and Vocabulary
uni_embs, vocab = model.get_unigram_embeddings()
print(uni_embs[0])
print(vocab[0])
