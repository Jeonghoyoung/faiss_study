# Sent2Vec
### 사전 지식
	* Subsampling 
		- 빈도 수가 높은 단어를 학습에서 제외해 학습 속도를 높여 불용어 제거와 유사한 효과를 갖게 한다.
		- 학습 배제는 랜덤 확률로 한다.
	* NegativeSampling
		- 하나의 중심 단어에 대해서 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어 놓고 마지막 단계를 이진 분류 문제로 변환, 주변 단어들을 긍정, 랜덤으로 샘플링된 단어들을 부정으로 레이블링한다면 이진 분류를 위한 데이터셋으로 하여 단어 집합의 크기만큼의 선택지를 두고 다중 클래스 분류로 풀던 방식보다 연산량에서 효율적인 방법.
		- 다중분류에서 이진분류로 변환하는 과정에서 Negative Sampling이라는 기법이 활용된다.
		- 기존의 중심 단어로부터 주변 단어를 예측하는 모델인 Skip-gram과 달리, 중심 단어와 주변 단어가 모두 입력되고, 두 단어가 실제로 윈도우 크기내에 존재하는 이웃 관계인지 그 확률을 예측한다.
		- 긍정적 예를 타깃으로 한 경우의 손실을 구하면서, 동시에 부정적 예를 샘플링하여 손실을 구한다. 부정적 예를 샘플링하는 좋은 방법으로 코퍼스에서 자주 등장하는 단어를 많이 추출하고 드물게 등장하는 단어를 적게 추출하는 방법이다.

	*** 다중분류 : 'you' 와 'goodbye' 라는 2개의 맥락 단어가 입력으로 주어졌을 때, 가운데에 나올 단어가 무엇인가?		
	*** 이진분류 : 'you' 와 'goodbye' 라는 2개의 맥락 단어가 입력으로 주어졌을 때, 가운데에 나올 단어는 'say'인가? 아닌가?
		

		

### Sent2Vec
	- 범용적인 문장 임베딩을 목표로 하는 비지도 학습 모델로, Word2Vec의 CBOW 모델을 문장 단위로 확장한 모델이다.
	- 문장에 존재하는 모든 요소의 벡터 합에 평균을 취한 값을 문장 임베딩 값을 갖는다.
	- 기본적인 학습방식은 문장의 네거티브 샘플링을 통해 컨텍스트(주변) 전체의 loss를 최소화 하는 형태로 학습한다.

	- CBOW(Cotinuous Bag-of-Words)
		- CBOW: 컨텍스트가 주어졌을때, 문맥 단어로부터 기준단어를 예측하는 모델로써, 기준 단어에 대해 앞 뒤로 N/2개씩, 총 N개의 문맥 단어를 입력으로 사용하여, 기준 단어를 맞추기 위한 네트워크를 만든다.
	- CBOW vs Sent2Vec
		1. 서브샘플링 비활성화 : 서브샘플링은 모든 n-gram 생성을 가로막고 문장에서 중요한 부분을 앗아갈수 있고, 서브샘플링된 단어만 남게되면 단어간 거리를 단축시켜 컨텍스트 윈도우의 크기를 암묵적으로 증가시키는 부작용을 낳는다.
		2. 다이나믹 컨텍스트 윈도우 비활성화 : 문장 전체의 의미를 살리기 위해 문장의 모든 n-gram을 조합하여 학습하기 때문에 다이나믹 컨텍스트 윈도우를 사용하지않고 문장의 전체 길이로 고정한다.
		3. CBOW가 토큰 단위로 업데이트 하는 것에 비해 Sent2Vec은 문장 길이에 비례한 만큼 업데이트 한다.


	- Module Install in Mac OS
	    1) brew install wget
	    2) wget https://github.com/epfml/sent2vec/archive/master.zip
	    3) unzip master.zip
	    4) make
	    5) sudo python3.7 -m pip install .
	    *) 만약 가상환경 사용시 로컬 default python 버전의 site-packages 디렉토리에 설치 되므로, cp -r path/sent2vec-0.0.0.dist-info / cp -r path/sent2vec-0.0.0.dist-info 실행
	    **) 설치후 sent2vec-master 디렉토리 삭제X (추후 학습에 필요)
	    ***) pip install sent2vec 또는 모듈 upgrade 금지. (다른 모듈 설치됨)

	- Train
	    * command:
	    	* ./fasttext sent2vec -input ../fasttext_test.txt -output test_model -dropoutK 0 -dim 200 -epoch 9 -lr 0.2 -thread 10 -bucket 100000

	    * Parameters
	       * -input: training file path
           * -output: output file path
	       * -lr: learning rate (default=0.2)
           * -lrUpdateRate: change the rate of updates for the learning rate (default=100)
           * -dim: dimension of word and sentence vectors (default=100)
           * -epoch: number of epochs (default=5)
           * -minCount: minimal number of word occurences (default=5)
           * -minCountLabel: minimal number of label occurences, sent2vec의 경우 비지도 학습으로 라벨이 필요치 않음으로 0 (default=0)
           * -neg: number of negatives sampled (default=10)
           * -wordNgrams: max length of word ngram (default=2, 한국어인 경우 0)
           * -loss: loss function {ns, hs, softmax} (default=ns) (ns : , hs : hierarchical softmax)
           * -bucket: number of hash buckets for vocabulary (default=2000000)
           * -thread: number of threads (default=2)
           * -t: sampling threshold (default=0.0001)
           * -dropoutK: number of ngrams dropped when training a sent2vec model (default=2)
           * -verbose: verbosity level (default=2)
           * -maxVocabSize: vocabulary exceeding this size will be truncated (default=None)
           * -numCheckPoints: number of intermediary checkpoints to save when training (default=1)

* loss funtion 
	1. ns(negative sampling) : softmax로 부터 발생하는 어마어마한 summation을 K개의 negative smapling 으로 해결하여 연산량을 줄여 속도를 향상
	2. hs (hierarchical softmax) : 출력층 값을 softmax함수로 얻는것 대신 binary tree를 이용하여 값을 얻는 방법으로 root에서 부터 leaf까지 가는길에 있는 확률을 모두 곱하여 출력층의 값을 계산한다.(skipgram을 사용할때 lossfunction 으로 사용)
  



# Faiss

Faiss : Facebook에서 만든 vector 유사도를 측정하는 라이브러리.
	
- vector 유사도를 측정하는데 주로 사용되는 numpy 나 scikit-learn에서 제공하는 cosine similarity 등 보다 빠르고 정확한 유사도를 측정할 수 있다.
- 내부적으로 C++로 구현되어있으며, GPU를 지원한다. 
- dence vector들의 클러스터링과 유사도를 구할 때 사용된다. 
- Similarity search의 방법들을 포함하는 라이브러리이다.

    * Similarity search란, 특정한 차원을 가진 벡터들의 집합이 있을때 데이터 구조들을 램 위에 올려두고 새로운 벡터가 들어 왔을때 거리가 가장 적은 벡터를 계산하는것.
        이때 거리는 유클라디안 거리를 사용한다.
    * faiss에서 data structure(데이터 구조)는 index라 한하며 객체들을 벡터들의 집합과 더하기 위한 더하기 method를 가진다.


    - 기능 및 특징
        1) nearest neighbor와 k-th nearest neighbor를 얻을 수 있다.
        2) 한번에 한 벡터가 아닌 여러 벡터를 검색할수 있다(batch processing). 여러 인덱스 타입들에서 여러 벡터를 차례로 검색하는것 보다 더 빠르다.
        3) 정확도와 속도 간에 트레이드 오프가 존재한다. 예를들어 10% 부정확한 결과를 얻는다고 할때, 10배 더 빠르거나, 10배 더 적은 메모리를 사용할 수 있다.
        4) 유클라디안 거리를 최소화 하는 검색이 아닌 maximum inner product가 최대로 되는 방식으로 계산한다.
        5) query point에서 주어진 radius에 포함되는 모든 element들을 반환한다.
        6) 인덱스는 디스크에 저장 된다.

    - Install in python
        * CPU : pip3 install pytorch faiss-cpu
        * GPU : pip3 install pytorch faiss-gpu

### Faiss example

#### faiss_tutorial.py

	* faiss에선 index 라는 개념을 사용, index는 데이터베이스 벡터들의 집합을 캡슐화 하고 효율적으로 검색하기 위해 선택적으로 전처리를 할 수도 있다.
    * 여러 index 타입들이 있으며, 가장 단순하게 사용할 수 있는 알고리즘은 brute-force L2 distance 검색을 하는 IndexFlatL2 이다.
    ** IndexFlat2는 별도의 학습을 필요로 하지않는다.
    
	* 필요 인자
    1. xb : 모든 인덱스 벡터들이 포함된 데이터 베이스 (shape: (database size, dimension))
    2. xq : nearest neighbor을 찾기 위한 쿼리 벡터 (shape: (database size of queries, dimension) 단, 단일 쿼리벡터인 경우 nq = 1)

    * Building an index and adding the vectors to it
    - 모든 인덱스들은 빌드 될때 어떤 벡터 차원 d에서 연산되는지 정보를 필요로 한다.
    - 대부분의 인덱스들은 학습 단계를 필요로 한다. 학습이 필요한 이유는 인덱스를 구성하는 벡터들의 분포를 분석할 필요 때문.( 단, IndexFlasL2의 경우 학습을 필요로 하지 않음.)
    - 인덱스를 빌드하고 학습할때 add와 search 두가지 연산이 인덱스에 대해 수행.
        * (1) add : 인덱스에 벡터를 더할때 사용
        * (2) is_trained : 학습이 되었는지 상태 확인
        * (3) ntotal : 인덱싱된 벡터들의 갯수

	* Searching
    - 기본적인 Searching 연산은 인덱스에서 k-nearest-neighbor 을 통해 할 수 있다.
    - 각 쿼리 벡터에 대해서 k개의 근접 이웃 벡터를 검색해준다. 이때 integer matrix 안에 저장되고 matrix의 shape은 (nq, k) 이다.
    - row i 는 쿼리벡터 i에 대한 neighbors의 id들을 담고 있다. neighbor 들은 거리가 증가하는 순서로 정렬 되어 잇으며, search 연산은 (nq, k)의 모양의 squared distance, floating-point 매트릭스를 반환한다.


#### faiss_IndexFlatL2
	- 앞서, sent2vec에서 학습한 모델을 통해 sentences 를 임베딩 후 faiss에 적용.
	- 학습을 필요로 하지 않는 IndexFlat2 알고리즘을 테스트.
	- IndexFlatL2 인데스만 사용할 경우 계산 비용과 시간이 많이 소모되고 잘 확장되지 않는다는 단점이 있다.
	- 시간과 비용소모를 줄이기 위해 검색범위를 줄여 정확한 답변보단 대략적인 답변을 생성한다.
	
~~~

import sent2vec
import faiss

# sent2vec model load
model = sent2vec.Sent2vecModel()
model.load_model('{model path}')

# data embedding
embed_data = model.embed_sentences(data)
d = embed_data.shape[1] # dimension

# IndexFlatL2 알고리즘 적용
index = faiss.IndexFlatL2(d)

faiss_index.add(embed_data)

# 군집수 설정, search sentence embedding
k = 3
xq = model.embed_sentence('outpatient treatment')
D, I = faiss_index.search(xq, k)

# result 
for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")


~~~

#### faiss.IndexIVFFLat
	- IndexFlatL2를 사용하여 인덱스를 양자화 단계로 사용하고 이 인덱스를 partitioning IndexIVFFLat 인덱스에 제공. 
	- partitioning: 인덱스를 관리하기 쉬운 단위로 분리하는 방법.
	- IndexFlatL2와 달리 학습이 필요하며 data를 index에 추가하기 전에 필수로 학습되어야 한다.
	- IndexFlatL2 보다 search 속도면에서 성능 향상되었다.
	- 검색 범위를 늘려 정확도를 높일수 있다.(index.nprobe)
	- 범위가 커지면 커질수혹 검색 속도도 늘어난다.
	- .nprobe : 검색할 주변 셀의 수
	- Vector Reconstruction
	- IVF 단계의 추가로 인해 원래의 vector와 index 위치 간에 직접적인 매핑이 없으므로 .make_direct_map()을 통해 매핑 이후 numerical vector을 확인할 수 있다.

~~~

# Use IndexIVFFLat
nlist = 50 # index에 포함할 파티션 수를 지정
quant = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quant, d, nlist)

# IndexIVFFLat train
index.train(embed_data)

index.add(embed_data)

k = 3
xq = model.embed_sentence('outpatient treatment')

D, I = index.search(xq, k)

for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")


# .nprobe를 통해 검색범위를 늘려 정확도 향상.
index.nprobe = 10
D, I = index.search(xq, k)
print(I)
for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")

~~~

#### faiss.IndexIVFPQ
	- Product Quantization(PQ)
	- IVF를 사용하면 검색범위를 줄여 근사치를 구하는 반면 PQ는 거리/유사성 계산을 근사화한다.
	- 세 단계로 구성된 벡터를 압축하여 유사성 연산을 수행.
				
		* (1) Vector를 여러개의 subvectors로 분리
		* (2) 각 subvector set에 대해 클러스터링 작업을 수행하여 여러 중심을 생성.
		* (3) 각 subvector에서 가장 가까운 set별 중심의 ID로 바꾼다.

	- IndexIVFFLat와 마찬가지로 학습이 필요한 알고리즘.

* IVF와 PQ는 정확한 결과를 얻지는 못하더라도 정답에 가까워지는 근사값을 보다 나은 속도로 얻을 수 있다.

~~~

# Use IndexIVFPQ
nlist = 50 # index에 포함할 파티션 수를 지정
m = 8 # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid

quant = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quant, d, nlist, m, bits)

index.train(embed_data)

index.add(embed_data)

k = 3
xq = model.embed_sentence('outpatient treatment')

# .nprobe를 통해 검색범위를 늘려 정확도 향상.
index.nprobe = 10
D, I = index.search(xq, k)

for i,z in zip(I[0], D[0]):
    print(f"Distance D : {z}")
    print(f"Index I {i} : {df['data'][i]}")

~~~
	* References:
		* https://github.com/epfml/sent2vec
		* https://arxiv.org/pdf/1703.02507.pdf
		* https://docs.likejazz.com/sent2vec/
		*  https://yjjo.tistory.com/14 (loss function)
		* https://www.pinecone.io/learn/faiss-tutorial/
		* https://github.com/facebookresearch/faiss




