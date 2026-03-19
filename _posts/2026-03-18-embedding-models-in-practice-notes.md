---
title: "임베딩 모델 실습 정리: One-hot, Word2Vec, GloVe를 코드로 확인하기"
date: 2026-03-18 23:22:13 +0900
categories: [Data Analysis, Text Mining]
tags: [nlp, text-mining, embedding, word2vec, glove]
description: "One-hot encoding부터 Word2Vec, GloVe까지 실습 코드 기준으로 정리하고, 개념이 코드에서 어떻게 연결되는지 기록한 글"
math: false
---

이 노트는 `words_embedding.ipynb` 실습을 바탕으로, 개념으로 배운 임베딩 모델들이 실제 코드에서 어떻게 사용되는지를 정리한 것이다.

실습의 흐름은 크게 다음 세 단계로 이루어진다.

1. One-hot encoding으로 단어를 단순 구분용 벡터로 표현
2. Word2Vec으로 단어를 의미 기반 벡터로 표현
3. GloVe로 단어를 의미 기반 벡터로 표현하고 유사도 비교

즉, 이 실습은 "단어를 숫자로 표현하는 방식이 어떻게 발전하는가"를 직접 확인하는 과정이라고 볼 수 있다.

원본 실습 노트북은 [words_embedding.ipynb](/assets/files/notebooks/words_embedding.ipynb)에서 확인할 수 있다.

이 글은 단순히 코드 실행 순서를 적는 데서 멈추지 않고, 다음 질문에 답하는 방향으로 정리했다.

- one-hot encoding은 실습에서 어떤 역할을 했는가
- Word2Vec과 GloVe는 코드에서 어떻게 쓰였는가
- 개념으로 배운 "분산 표현"이 실제 코드에서는 어떻게 보이는가
- 실습 기준으로 각 방식의 차이를 어떻게 이해하면 되는가

## 1. 실습의 전체 목적

이 실습의 핵심 목적은 단어를 숫자로 바꾸는 여러 방법을 비교하는 것이다.

처음에는 가장 단순한 방식인 one-hot encoding을 사용하고, 그다음 분산 표현 기반 임베딩인 Word2Vec과 GloVe를 적용해본다.  
이를 통해 다음 차이를 체감할 수 있다.

- one-hot encoding은 단어를 구분만 할 수 있음
- Word2Vec과 GloVe는 단어 의미 유사성을 반영할 수 있음

즉, 같은 "단어를 벡터로 바꾼다"는 작업이라도, 벡터가 담고 있는 정보 수준이 다르다는 점을 실습으로 확인하는 것이다.

## 2. One-hot Encoding은 실습에서 어떻게 사용되었는가

실습의 첫 단계에서는 `sklearn.preprocessing.OneHotEncoder`를 사용해 단어를 one-hot 벡터로 바꾸었다.

예를 들어 문장:

```text
사과는 맛있다 바나나는 맛있다
```

를 띄어쓰기 기준으로 나누면 다음과 같다.

```python
['사과는', '맛있다', '바나나는', '맛있다']
```

이후 이 단어들을 `numpy` 배열로 바꾸고 `reshape(-1, 1)` 형태로 만든 뒤, `OneHotEncoder`에 넣는다.

```python
words_array = np.array(words).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(words_array)
```

### 여기서 실제로 일어나는 일

- `fit` 단계
  - 문장 안에 어떤 단어들이 있는지 확인한다
  - 각 단어를 고유한 열 위치에 매핑한다
- `transform` 단계
  - 각 단어를 해당 위치만 1이고 나머지는 0인 벡터로 바꾼다

또한:

```python
encoder.get_feature_names_out()
```

을 통해 어떤 열이 어떤 단어를 의미하는지 확인할 수 있었다.

### 이 실습이 보여주는 것

이 단계는 one-hot encoding이 단어를 "구분"하는 데는 유용하지만, 단어 의미까지 반영하지는 못한다는 점을 보여준다.

예를 들어:

- `사과는`
- `바나나는`

가 과일이라는 공통 의미를 가진다는 정보는 벡터 안에 들어 있지 않다.  
각 단어는 단지 서로 다른 위치의 1로만 표현될 뿐이다.

즉, 실습에서는 one-hot encoding이 "희소하고 의미 정보가 없는 표현"이라는 점을 직접 확인하는 역할을 했다.

실제로 출력된 결과는 다음과 같았다.

```text
['사과는', '맛있다', '바나나는', '맛있다']

[['사과는']
 ['맛있다']
 ['바나나는']
 ['맛있다']]

[[0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]

['x0_맛있다', 'x0_바나나는', 'x0_사과는']
```

즉 `맛있다`, `바나나는`, `사과는`가 각각 고유한 열 위치로 매핑되고, 각 단어는 해당 위치만 1인 벡터로 표현되었음을 확인할 수 있었다.

## 3. Word2Vec은 실습에서 어떻게 사용되었는가

그다음 단계에서는 `gensim`을 통해 사전학습된 Word2Vec 모델을 불러왔다.

```python
from gensim.downloader import load
word2vec_model = load('word2vec-google-news-300')
```

이 모델은 영어 뉴스 말뭉치로 미리 학습된 단어 임베딩 모델이다.  
여기서 `300`은 각 단어가 300차원 벡터로 표현된다는 뜻이다.

### 단어 벡터 조회

실습에서는 먼저 특정 단어에 대해 벡터를 직접 조회했다.

```python
print(use_word2vec('apple', word2vec_model))
print(use_word2vec('instagram', word2vec_model))
```

이 단계는 다음 사실을 보여준다.

- Word2Vec에서는 단어 하나를 dense vector로 표현할 수 있다
- 모델이 알고 있는 단어만 벡터를 반환할 수 있다
- 학습 vocabulary에 없는 단어는 `KeyError`가 발생할 수 있다

즉, Word2Vec은 one-hot과 달리 단어 하나를 길고 희소한 0/1 벡터가 아니라, 의미를 담은 dense vector로 표현한다.

또한 노트북에서는 `apple`은 벡터를 반환했지만, `instagram`은 모델 어휘에 없어 임베딩이 되지 않는다는 점도 같이 확인했다.

### 단어 유사도 계산

실습에서는 `scipy.spatial.distance.cosine`을 이용해 두 단어 벡터 간 코사인 유사도를 계산했다.

```python
similarity = 1 - cosine(vector1, vector2)
```

예:

```python
word_similarity('football', 'basketball', word2vec_model)
word_similarity('football', 'airplane', word2vec_model)
```

이 단계는 Word2Vec이 비슷한 의미의 단어를 벡터 공간에서도 가깝게 표현한다는 점을 보여준다.

즉:

- `football`과 `basketball`은 높은 유사도
- `football`과 `airplane`은 낮은 유사도

같은 결과를 통해, 벡터가 단순한 식별 정보가 아니라 의미 관계를 담고 있음을 확인하는 것이다.

실제 유사도 출력은 다음과 같았다.

```text
football & basketball 유사도 : 0.6682468
football & airplane 유사도 : 0.15124393
```

즉 스포츠 종목끼리는 상대적으로 가깝고, 전혀 다른 범주의 단어와는 거리가 멀다는 점이 수치로 드러난다.

### 가장 유사한 단어 찾기

실습에서는 다음 메서드도 사용했다.

```python
model.most_similar(word, topn=5)
```

예:

```python
most_similar('football', word2vec_model, topn=5)
```

이것은 특정 단어와 가장 가까운 벡터를 가진 단어들을 찾는 기능이다.  
즉, Word2Vec이 학습한 의미 공간을 직접 탐색해보는 실습이라고 볼 수 있다.

예를 들어 `football`과 가장 유사한 단어 5개는 다음처럼 나왔다.

```text
[('soccer', 0.7313),
 ('fooball', 0.7140),
 ('Football', 0.7125),
 ('basketball', 0.6682),
 ('footbal', 0.6649)]
```

## 4. GloVe는 실습에서 어떻게 사용되었는가

다음 단계에서는 GloVe 모델을 불러와 Word2Vec과 비슷한 방식으로 실습했다.

```python
glove_model = load('glove-wiki-gigaword-300')
```

이 모델 역시 사전학습된 단어 임베딩 모델이며, 각 단어를 300차원 벡터로 표현한다.

### 단어 벡터 조회

실습에서는 GloVe에서도 먼저 특정 단어 벡터를 직접 조회했다.

```python
print(use_glove('apple', glove_model))
print(use_glove('airpods', glove_model))
```

이 과정은 GloVe도 Word2Vec처럼:

- 단어 -> dense vector

형태의 매핑을 제공한다는 점을 보여준다.

또한 학습 vocabulary에 없는 단어는 벡터를 제공할 수 없다는 점도 같이 확인된다.

### 유사도 계산과 유사 단어 탐색

실습에서는 Word2Vec에서 사용한 함수들을 그대로 GloVe에도 적용했다.

```python
word_similarity('football', 'basketball', glove_model)
word_similarity('football', 'airplane', glove_model)
most_similar('football', glove_model, topn=5)
```

이 단계의 의미는 다음과 같다.

- Word2Vec과 GloVe는 학습 방식은 다르지만
- 둘 다 단어 의미를 벡터 공간에 담는다는 점에서는 비슷하게 사용할 수 있다

즉, 실제 코드 레벨에서는:

- 벡터 조회
- 유사도 계산
- 비슷한 단어 검색

같은 작업을 거의 동일한 방식으로 수행할 수 있다.

GloVe에서도 실제 출력은 다음과 같았다.

```text
football & basketball 유사도 : 0.7341024
football & airplane 유사도 : 0.0022235513
```

그리고 `football`과 가장 유사한 단어는 아래처럼 확인되었다.

```text
[('soccer', 0.7683),
 ('basketball', 0.7341),
 ('league', 0.6600),
 ('baseball', 0.6480),
 ('rugby', 0.6430)]
```

Word2Vec과 마찬가지로 관련 스포츠 단어들이 상위에 나타났지만, 이 실습에서는 `football`과 `airplane`의 거리 차이가 더 극명하게 보였다.

## 5. 이 실습에서 개념과 연결되는 핵심 포인트

`words_embedding.ipynb` 실습은 단순히 코드를 따라 친 것이 아니라, 개념 수업에서 배운 내용을 실제 코드로 대응시키는 과정이다.

### 1. One-hot Encoding의 한계 확인

개념에서 배운 것처럼 one-hot encoding은:

- 희소 벡터를 만들고
- 단어 의미를 반영하지 못한다

실습에서는 `OneHotEncoder`를 통해 이 점을 직접 확인했다.

### 2. 분산 표현의 의미 확인

Word2Vec과 GloVe 실습에서는 단어마다 dense vector가 반환된다.  
즉, 분산 표현이 실제 코드에서는 "단어를 실수 벡터로 조회하는 것"으로 나타난다.

### 3. 의미 유사성의 수치화

개념에서는 "비슷한 단어는 비슷한 위치에 존재한다"라고 배웠고, 실습에서는 이를 코사인 유사도로 직접 계산했다.

즉:

- 벡터 공간의 가까움
- 단어 의미의 유사성

이 연결된다는 점을 수치로 확인한 것이다.

### 4. 사전학습 모델 활용 방식 이해

실습에서 `load('word2vec-google-news-300')`, `load('glove-wiki-gigaword-300')`를 사용한 것은, 우리가 직접 임베딩 모델을 학습한 것이 아니라 이미 큰 말뭉치로 학습된 결과를 가져다 쓴 것이라는 점을 보여준다.

즉:

- 지금 하는 일은 임베딩 학습이 아니라
- 사전학습된 임베딩 활용이다

## 6. 실습 관점에서 Word2Vec과 GloVe는 어떻게 다르게 느껴지는가

이 노트북에서는 두 모델을 거의 같은 함수로 다룬다.  
그래서 사용 방식만 보면 비슷하게 느껴질 수 있다.

하지만 개념적으로는 차이가 있다.

- Word2Vec
  - 주변 문맥 예측 기반
- GloVe
  - 전체 말뭉치의 공동 출현 통계 기반

즉, 실습에서는 같은 방식으로 벡터를 조회하지만, 그 벡터가 만들어진 원리는 다르다.

이 점이 바로 "사용법은 비슷하지만 학습 철학은 다르다"는 임베딩 모델 비교의 핵심이다.

## 7. 실습 기준 비교표

이 실습에서 다룬 방법들을 간단히 비교하면 다음과 같다.

| 방법 | 표현 대상 | 의미 반영 | 실습에서 확인한 것 | 한계 |
| --- | --- | --- | --- | --- |
| One-hot Encoding | 단어 | 아니오 | 단어를 구분용 벡터로 바꾸는 과정 | 희소 벡터이며 의미 유사성을 표현하지 못함 |
| Word2Vec | 단어 | 예 | 단어 벡터 조회, 코사인 유사도 계산, 유사 단어 탐색 | OOV와 문맥 반영에 한계가 있음 |
| GloVe | 단어 | 예 | Word2Vec과 유사한 방식으로 벡터 활용 가능함 | 정적 임베딩이라 문맥 차이를 반영하지 못함 |

## 8. 이 실습이 아직 하지 않은 것

이 실습은 단어 임베딩을 확인하고 비교하는 데 초점이 있다.  
아직 다음 단계까지는 가지 않았다.

- 문장 벡터 만들기
- 여러 단어 벡터를 평균내어 문장 표현 만들기
- 문장 임베딩으로 분류 모델 학습하기

즉, 현재 실습은 word embedding 활용의 입문 단계라고 볼 수 있다.

## 9. 핵심 요약

- one-hot encoding은 단어를 구분하는 데는 유용하지만 의미 유사성을 담지 못한다.
- Word2Vec과 GloVe는 단어를 dense vector로 표현해 의미 기반 유사도 계산이 가능하다.
- 실습 코드에서는 Word2Vec과 GloVe를 비슷하게 다루지만, 두 모델의 학습 원리는 다르다.
- 이번 실습은 "임베딩을 직접 학습"하기보다 "사전학습 임베딩을 활용"하는 흐름에 가깝다.
- 현재 단계는 단어 임베딩 입문이며, 다음 단계는 문장 벡터와 다운스트림 태스크 연결이다.

## 10. 정리

`words_embedding.ipynb` 실습은 다음 흐름으로 이해하면 된다.

1. One-hot encoding으로 단어를 단순 구분용 벡터로 표현
2. Word2Vec으로 단어를 의미 기반 dense vector로 표현
3. GloVe로 단어를 의미 기반 dense vector로 표현
4. 코사인 유사도와 유사 단어 검색으로 임베딩 공간을 직접 확인

즉, 이 실습은 "단어 벡터화 방식의 발전"을 코드로 확인하는 과정이다.

개념적으로는:

- 희소 표현에서 분산 표현으로 이동
- 단순 구분에서 의미 반영으로 이동
- 수동 벡터화에서 사전학습 임베딩 활용으로 이동

이라는 흐름을 보여준다.
