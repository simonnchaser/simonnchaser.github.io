---
title: "Yelp 감성 분석 실습에서 TF-IDF가 쓰이는 방식 정리"
date: 2026-03-19 18:23:35 +0900
categories: [Data Analysis, Text Mining]
tags: [nlp, text-mining, tfidf, yelp, sentiment-analysis]
description: "Yelp 감성 분석 노트북을 바탕으로 TF-IDF가 전처리, 문장 임베딩, 분류 모델까지 어떻게 연결되는지 코드 기준으로 정리한 글"
math: false
---

이 노트는 `yelp_sentiment_analysis_glove_tfidf.ipynb` 실습에서 `TF-IDF`가 실제로 어떤 흐름으로 사용되는지를 코드 기준으로 자세히 정리한 것이다.

원본 실습 노트북은 [yelp_sentiment_analysis_glove_tfidf.ipynb](/assets/files/notebooks/yelp_sentiment_analysis_glove_tfidf.ipynb)에서 함께 확인할 수 있다.

이번 정리의 목적은 다음과 같다.

- 실습에서 `TfidfVectorizer`가 정확히 무엇을 하는지 이해하기
- `fit`, `transform`, `fit_transform`이 여기서 어떤 의미인지 파악하기
- `tfidf_matrix`와 `tfidf_feature_names`가 각각 무엇인지 이해하기
- 최종적으로 TF-IDF가 GloVe 기반 문장 임베딩에 어떻게 연결되는지 보기

## 1. 실습에서 TF-IDF가 등장하는 이유

이 실습은 감성 분석을 위해 문장을 숫자 벡터로 표현하려는 과정이다.

처음에는 각 문장을 전처리해서 토큰 리스트로 만든다.

예:

```python
['wow', 'loved', 'place']
```

이후에는 두 가지 정보가 필요해진다.

- 각 단어의 의미 벡터
- 각 단어가 현재 문서에서 얼마나 중요한지에 대한 가중치

여기서:

- 단어 의미 벡터는 `GloVe`
- 단어 중요도 가중치는 `TF-IDF`

가 담당한다.

즉 TF-IDF는 단어 자체의 뜻을 주는 것이 아니라, "이 문장에서 이 단어를 얼마나 중요하게 반영할 것인가"를 정하는 역할을 한다.

## 2. TF-IDF 코드가 시작되는 부분

실습의 핵심 코드는 다음과 같다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in data['preprocessed']])
tfidf_feature_names = vectorizer.get_feature_names_out()
```

이 코드는 짧아 보이지만, 실제로는 여러 단계를 한 번에 수행하고 있다.

## 3. `TfidfVectorizer`는 무엇인가

`TfidfVectorizer`는 `scikit-learn`의 텍스트 벡터화 도구다.

역할은 다음과 같다.

- 여러 문서 문자열을 입력으로 받는다
- 전체 vocabulary를 만든다
- 각 단어의 IDF를 계산한다
- 각 문서를 TF-IDF 벡터로 변환한다

즉 `CountVectorizer`가 단어 빈도 행렬을 만든다면, `TfidfVectorizer`는 거기에 TF-IDF 가중치를 반영한 행렬을 만든다고 이해하면 된다.

## 4. 왜 `join`을 해주는가

실습에서 `data['preprocessed']` 컬럼에는 전처리된 단어 리스트가 들어 있다.

예를 들면 각 행은 대략 이런 모양이다.

```python
['wow', 'loved', 'place']
['service', 'terrible']
['food', 'great', 'fresh']
```

그런데 `TfidfVectorizer`는 기본적으로 이런 리스트를 직접 받기보다, 문자열 문서들의 집합을 기대한다.

예:

```python
"wow loved place"
"service terrible"
"food great fresh"
```

그래서 다음 코드가 필요하다.

```python
' '.join(doc)
```

이 코드는 토큰 리스트를 공백으로 이어서 하나의 문자열 문서로 다시 만들어준다.

즉:

```python
['wow', 'loved', 'place']
```

를

```python
"wow loved place"
```

로 바꾸는 것이다.

### 중요한 점

여기서 하는 일은 "모든 문장을 하나의 큰 문서로 합치는 것"이 아니다.

정확히는:

- 각 문서의 토큰 리스트를
- 다시 하나의 문자열 문서로 바꾼 뒤
- 여러 문서의 집합 형태로 `TfidfVectorizer`에 넣는 것

이다.

즉 이 코드는:

```python
[' '.join(doc) for doc in data['preprocessed']]
```

를 통해 각 리뷰 문장을 하나의 문서 문자열로 복원하는 과정이다.

## 5. 여기서 말하는 문서는 무엇인가

이 실습에서는 Yelp 데이터의 각 리뷰 문장 하나가 하나의 문서다.

예를 들어:

```python
"wow loved place"
```

이 하나의 문서이고,

```python
"crust good"
```

도 또 하나의 문서다.

즉:

- 문서 집합 = 전체 Yelp 리뷰 문장들
- 문서 하나 = 데이터 한 행에 해당하는 리뷰 문장

이라고 이해하면 된다.

## 6. `fit_transform()`은 여기서 무엇을 하는가

`fit_transform()`은 `fit`과 `transform`을 한 번에 수행하는 메서드다.

이 실습에서는 다음 두 일을 함께 한다.

### 1. `fit`

전체 문서 집합을 보고:

- 어떤 단어들이 존재하는지 vocabulary를 만들고
- 각 단어가 몇 개 문서에 등장했는지 세고
- 그 정보를 바탕으로 IDF를 계산한다

즉 `fit`은 전체 말뭉치 기준의 전역 정보를 학습하는 단계다.

### 2. `transform`

이미 학습한 vocabulary와 IDF를 바탕으로:

- 각 문서의 TF를 계산하고
- TF와 IDF를 결합해
- 각 문서를 TF-IDF 벡터로 바꾼다

즉 `transform`은 문서별 숫자 표현을 실제로 만드는 단계다.

### 한 줄 정리

- `fit` -> vocabulary + IDF 학습
- `transform` -> 각 문서를 TF-IDF 벡터로 변환

## 7. `tfidf_matrix`는 무엇인가

이 변수에는 최종적으로 TF-IDF 행렬이 들어간다.

```python
tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in data['preprocessed']])
```

이 행렬은 보통 다음 구조를 가진다.

- 행(row): 각 문서
- 열(column): 전체 vocabulary에 포함된 단어들
- 값(value): 해당 문서에서 해당 단어의 TF-IDF 값

즉:

```text
tfidf_matrix[d, t]
```

는 "문서 `d`에서 단어 `t`의 TF-IDF 값"을 의미한다.

### 예시로 생각하기

문서가 다음처럼 3개 있다고 하자.

```python
[
    "wow loved place",
    "service terrible",
    "food great fresh"
]
```

그럼 행렬은 개념적으로 이런 느낌이 된다.

| 문서 | wow | loved | place | service | terrible | food | great | fresh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 문서 0 | 값 | 값 | 값 | 0 | 0 | 0 | 0 | 0 |
| 문서 1 | 0 | 0 | 0 | 값 | 값 | 0 | 0 | 0 |
| 문서 2 | 0 | 0 | 0 | 0 | 0 | 값 | 값 | 값 |

실제 값은 TF-IDF 점수로 채워진다.

## 8. 왜 `tfidf_matrix`는 sparse matrix인가

실제로 `tfidf_matrix`를 출력하면 일반적인 2차원 배열처럼 보이지 않고, `sparse matrix` 형태로 보인다.

이유는 대부분의 문서가 vocabulary 전체의 모든 단어를 포함하지 않기 때문이다.

즉:

- 열은 전체 단어 집합 기준으로 매우 많고
- 각 문서에는 그중 일부 단어만 등장한다

그래서 0이 매우 많다.

이런 경우 메모리를 아끼기 위해 `scikit-learn`은 희소 행렬(sparse matrix)로 저장한다.

## 9. `tfidf_feature_names`는 무엇인가

다음 코드는 vocabulary에 들어간 단어 목록을 가져온다.

```python
tfidf_feature_names = vectorizer.get_feature_names_out()
```

이 값은 열(column) 이름 목록이라고 생각하면 된다.

예를 들어:

```python
['food', 'fresh', 'great', 'loved', 'place', 'service', 'terrible', 'wow']
```

처럼 나올 수 있다.

즉 이 배열은:

- `tfidf_matrix`의 각 열이 어떤 단어를 뜻하는지 알려주는 인덱스표

역할을 한다.

## 10. 특정 단어의 위치를 찾는 코드

실습에서는 다음 코드가 나온다.

```python
np.where(tfidf_feature_names == 'wow')[0][0]
```

이 코드는 vocabulary 배열 안에서 `'wow'`가 몇 번째 열에 있는지 찾는 것이다.

즉:

- `tfidf_feature_names`는 단어 목록
- `np.where(...)`는 특정 단어의 인덱스 찾기

다.

이 인덱스를 알아야 `tfidf_matrix`에서 해당 단어의 TF-IDF 값을 꺼낼 수 있다.

## 11. 특정 문서에서 특정 단어의 TF-IDF 값을 구하는 코드

실습에서는 다음 흐름으로 TF-IDF 값을 가져온다.

```python
for word in preprocessed_word:
    doc_idx = 0
    word_idx = np.where(tfidf_feature_names == word)[0][0]
    value = tfidf_matrix.toarray()[0][word_idx]
    print(f'{word}의 tf_idf 값 : {value:.4f}')
```

이 코드를 단계별로 보면 다음과 같다.

### 1. 현재 보고 있는 문서를 정한다

```python
doc_idx = 0
```

즉 0번째 리뷰 문장을 보고 있다는 뜻이다.

### 2. 현재 단어가 vocabulary의 몇 번째 열인지 찾는다

```python
word_idx = np.where(tfidf_feature_names == word)[0][0]
```

### 3. 그 문서 행에서 해당 단어 열의 값을 꺼낸다

```python
value = tfidf_matrix.toarray()[0][word_idx]
```

이 값이 바로:

- 0번째 문서에서
- 해당 단어의
- TF-IDF 중요도

이다.

### 왜 `toarray()`를 쓰는가

`tfidf_matrix`는 sparse matrix이기 때문에, 일반적인 배열처럼 바로 보기 어렵다.

그래서:

```python
tfidf_matrix.toarray()
```

를 사용해 일반 NumPy 배열 형태로 바꾼 뒤 값을 인덱싱한다.

실제 예시 문장 `['wow', 'loved', 'place']`에서 확인된 TF-IDF 값은 다음과 같았다.

```text
wow의 tf-idf 값 : 0.7135
loved의 tf-idf 값 : 0.6028
place의 tf-idf 값 : 0.3571
```

즉 같은 문장 안에서도 모든 단어가 같은 중요도를 갖는 것이 아니라, `wow`, `loved`, `place`가 서로 다른 가중치로 반영된다는 점을 확인할 수 있었다.

## 12. TF-IDF가 GloVe와 어떻게 연결되는가

실습의 다음 단계에서는 TF-IDF를 단독으로 쓰지 않고, GloVe 단어 임베딩과 결합한다.

핵심 함수는 다음과 같다.

```python
def sentence_embedding(tfidf_matrix, tfidf_feature_names, doc, doc_idx):
    embeddings = []
    for word in doc:
        if word in glove and word in tfidf_feature_names:
            word_idx = np.where(tfidf_feature_names == word)[0][0]
            tfidf_weight = tfidf_matrix.toarray()[doc_idx, word_idx]
            embeddings.append(glove[word] * tfidf_weight)
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(100)
```

이 함수는 다음 원리로 동작한다.

### 1. 문장의 각 단어를 순회한다

```python
for word in doc:
```

### 2. 그 단어의 GloVe 벡터를 가져온다

```python
glove[word]
```

### 3. 그 단어의 현재 문서 TF-IDF 값을 가져온다

```python
tfidf_weight = tfidf_matrix.toarray()[doc_idx, word_idx]
```

### 4. 단어 의미 벡터에 중요도 가중치를 곱한다

```python
glove[word] * tfidf_weight
```

### 5. 모든 단어의 가중 벡터를 평균내 문장 벡터를 만든다

```python
np.mean(embeddings, axis=0)
```

즉 이 실습에서 TF-IDF는 독립적인 최종 표현이 아니라, GloVe 단어 벡터를 문장 수준으로 묶을 때 중요도 가중치로 사용된다.

실제로 예시 문장에 대해 계산된 문장 임베딩은 100차원 벡터로 반환되었고, 이 값이 이후 분류 모델의 입력 특성으로 사용된다.

## 13. 이 실습을 자연어로 풀어쓰면

`yelp_sentiment_analysis` 실습에서 TF-IDF는 다음 흐름으로 사용된다.

1. 각 리뷰 문장을 전처리한다
2. 전처리된 토큰 리스트를 다시 문자열 문서로 만든다
3. 전체 문서 집합에 대해 TF-IDF를 학습한다
4. 각 문서에서 각 단어의 TF-IDF 값을 얻는다
5. GloVe 단어 벡터에 TF-IDF 가중치를 곱한다
6. 가중 단어 벡터를 평균내 문장 임베딩을 만든다

즉 TF-IDF는 "단어가 현재 문장에서 얼마나 중요한가"를 계산해주는 중간 단계다.

## 14. 실습 결과로 무엇을 확인했는가

노트북에서는 TF-IDF 가중 문장 임베딩을 만든 뒤 로지스틱 회귀 분류기를 학습했고, 다음과 같은 평가 지표를 확인했다.

```text
Accuracy: 0.73
Precision: 0.79
Recall: 0.67
F1-Score: 0.73
```

또한 예제 문장들에 대해서는 긍정/부정 예측이 전반적으로 기대한 방향과 비슷하게 동작하는 것을 확인할 수 있었다.

즉 이 실습은 단순히 TF-IDF 행렬을 만드는 데서 끝나는 것이 아니라:

- TF-IDF 계산
- GloVe 가중 문장 임베딩 생성
- 감성 분류 모델 학습
- 예제 문장 추론

까지 하나의 흐름으로 이어져 있다.

## 15. 핵심 요약

- `TfidfVectorizer`는 문서 집합 전체를 보고 vocabulary와 IDF를 학습한 뒤 각 문서를 TF-IDF 벡터로 바꾼다.
- `tfidf_matrix`는 문서-단어 중요도 행렬이고, `tfidf_feature_names`는 그 열 이름 목록이다.
- 이 실습에서 TF-IDF는 최종 표현이 아니라 GloVe 단어 벡터에 곱해지는 가중치 역할을 한다.
- 이렇게 만들어진 가중 문장 임베딩은 감성 분류 모델의 입력으로 사용된다.
- 즉 코드 전체 흐름은 "전처리 -> TF-IDF 계산 -> GloVe 가중 문장 임베딩 -> 분류"로 이해하면 된다.

## 16. 정리

이 실습에서 `TfidfVectorizer`가 하는 일은 단순히 단어 빈도를 세는 것을 넘어서, 전체 Yelp 리뷰 집합을 기준으로 각 단어의 상대적 중요도를 계산하는 것이다.

핵심 포인트는 다음과 같다.

- `data['preprocessed']`의 각 행은 전처리된 토큰 리스트다
- `' '.join(doc)`은 그 리스트를 다시 문자열 문서로 복원한다
- `fit_transform()`은 vocabulary와 IDF를 학습하고 문서를 TF-IDF 벡터로 바꾼다
- `tfidf_matrix`는 문서-단어 TF-IDF 행렬이다
- `tfidf_feature_names`는 각 열이 어떤 단어인지 알려주는 vocabulary 목록이다
- 이후 TF-IDF는 GloVe 벡터의 가중치로 사용되어 문장 임베딩 생성에 연결된다

이 흐름을 이해하면, 실습 코드가 단순한 문법 조합이 아니라 "문서 중요도 계산 -> 단어 임베딩 가중치 부여 -> 문장 벡터 생성"이라는 하나의 연결된 과정이라는 점이 분명해진다.
