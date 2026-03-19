---
title: "TF-IDF 가중 문장 임베딩 정리: GloVe 평균만으로 부족한 이유"
date: 2026-03-19 18:23:35 +0900
categories: [Data Analysis, Text Mining]
tags: [nlp, text-mining, tfidf, sentence-embedding, glove]
description: "GloVe 단어 벡터를 단순 평균내는 방식의 한계를 정리하고, TF-IDF 가중치를 곱한 문장 임베딩이 왜 필요한지 설명한 글"
math: false
---

이 노트는 `yelp_sentiment_analysis_glove_tfidf.ipynb` 실습을 바탕으로, 왜 GloVe 단어 벡터를 단순 평균내는 것만으로는 부족할 수 있는지, 그리고 TF-IDF 가중치를 곱한 문장 임베딩이 어떤 원리로 사용되는지를 정리한 것이다.

원본 실습 노트북은 [yelp_sentiment_analysis_glove_tfidf.ipynb](/assets/files/notebooks/yelp_sentiment_analysis_glove_tfidf.ipynb)에서 함께 확인할 수 있다.

핵심 질문은 다음과 같다.

- TF-IDF는 무엇인가
- 단어 임베딩을 그냥 평균내면 왜 문제가 생기는가
- TF-IDF를 곱하면 무엇이 달라지는가
- 실제 코드에서는 어떤 방식으로 구현되는가

## 1. 현재 실습에서 하고 있는 일

`yelp_sentiment_analysis_glove_tfidf.ipynb`에서는 다음 흐름으로 문장 임베딩을 만들고 있다.

1. Yelp 리뷰 문장을 전처리한다.
2. 전처리된 각 단어에 대해 GloVe 벡터를 조회한다.
3. 각 단어의 TF-IDF 값을 구한다.
4. `GloVe(word) * TF-IDF(word)`를 계산한다.
5. 이 가중 벡터들을 평균내 문장 임베딩을 만든다.

즉, 이 실습은 "단어 임베딩 + 단어 중요도 가중치"를 결합한 문장 표현을 만드는 과정이다.

## 2. TF-IDF는 무엇인가

TF-IDF는 문서 안에서 특정 단어가 얼마나 중요한지를 수치로 나타내는 방법이다.

이 값은 크게 두 부분의 곱으로 생각할 수 있다.

- `TF (Term Frequency)`
- `IDF (Inverse Document Frequency)`

즉:

```text
TF-IDF = TF x IDF
```

### TF

TF는 어떤 단어가 특정 문서 안에서 얼마나 자주 등장하는지를 뜻한다.

예를 들어 한 문장 안에서 `excellent`가 여러 번 나오면, 그 문장에서는 `excellent`가 중요한 단어일 가능성이 높다.

### IDF

IDF는 어떤 단어가 전체 문서 집합에서 얼마나 드물게 등장하는지를 나타낸다.

- 모든 문서에 자주 나오는 단어는 중요도가 낮다
- 일부 문서에서만 특징적으로 나오는 단어는 중요도가 높다

즉, IDF는 "어느 문서에서나 흔한 단어"의 영향을 줄이고, 특정 문서를 더 잘 설명하는 단어의 가치를 높여준다.

### TF-IDF가 의미하는 것

결국 TF-IDF는 다음 질문에 답하는 값이다.

> 이 단어는 이 문서 안에서 얼마나 중요한가?

즉, 단어 자체의 의미 벡터와는 다른 정보다.  
GloVe가 단어의 의미를 벡터로 준다면, TF-IDF는 그 단어가 현재 문장에서 얼마나 중요하게 다뤄져야 하는지를 알려주는 가중치라고 볼 수 있다.

## 3. 왜 GloVe 단어 벡터를 그냥 평균내면 부족할 수 있는가

문장 임베딩을 가장 단순하게 만드는 방법은 문장 안 단어들의 GloVe 벡터를 모두 평균내는 것이다.

예를 들어 문장이 다음과 같다고 하자.

```text
고양이가 강아지를 쫒는다
```

전처리 후 토큰이 다음과 같다면:

```text
[고양이가, 강아지를, 쫒는다]
```

가장 단순한 문장 임베딩은:

```text
(GloVe(고양이가) + GloVe(강아지를) + GloVe(쫒는다)) / 3
```

처럼 만들 수 있다.

이 방식은 구현이 쉽고 빠르지만, 몇 가지 한계가 있다.

### 1. 모든 단어를 똑같이 중요하다고 가정한다

단순 평균은 `고양이가`, `강아지를`, `쫒는다`를 모두 같은 비중으로 처리한다.

하지만 실제로는 문장마다 더 중요한 단어와 덜 중요한 단어가 있다.

예를 들어 감성 분석에서는:

- `good`
- `terrible`
- `excellent`
- `boring`

같은 단어가 감정 판단에 훨씬 더 중요한 경우가 많다.

그런데 단순 평균을 하면 이런 중요한 단어가 평범한 단어들과 같은 비중으로 섞여버린다.

### 2. 흔한 단어의 영향이 너무 커질 수 있다

전체 문서에서 자주 나오는 단어는 보통 문서를 구별하는 힘이 약하다.

예:

- `place`
- `food`
- `service`

같은 단어가 Yelp 리뷰 전반에 자주 등장할 수 있다.

이런 단어는 문장의 주제를 설명할 수는 있지만, 특정 문장이 긍정인지 부정인지 구분하는 데는 덜 중요할 수 있다.

단순 평균은 이런 자주 나오는 단어도 똑같이 평균에 넣어버린다.

### 3. 서로 다른 문장이 비슷한 벡터가 될 수 있다

단어 집합이 비슷하면 평균 벡터도 비슷해질 수 있다.

예를 들어:

- `고양이가 강아지를 쫒는다`
- `강아지가 고양이를 쫒는다`

는 의미가 다르다.  
하지만 사용된 핵심 단어가 비슷하면 평균 벡터도 매우 비슷하거나 거의 같아질 수 있다.

왜냐하면 평균은 단어 순서를 반영하지 않기 때문이다.

즉, 단순 평균은 "어떤 단어가 있었는가"는 반영하지만, "어떤 단어가 더 중요했는가"나 "단어들이 어떤 순서로 결합되었는가"는 잘 반영하지 못한다.

## 4. TF-IDF를 곱하면 무엇이 달라지는가

이 한계를 완화하기 위해, 각 단어 벡터에 TF-IDF 가중치를 곱한 뒤 평균을 낼 수 있다.

즉 다음처럼 바뀐다.

단순 평균:

```text
(GloVe(고양이가) + GloVe(강아지를) + GloVe(쫒는다)) / 3
```

TF-IDF 가중 평균:

```text
(GloVe(고양이가) * TF-IDF(고양이가)
 + GloVe(강아지를) * TF-IDF(강아지를)
 + GloVe(쫒는다) * TF-IDF(쫒는다)) / 3
```

이렇게 하면 각 단어가 문장 임베딩에 기여하는 정도가 달라진다.

### 기대 효과

- 중요한 단어는 더 크게 반영됨
- 흔한 단어는 상대적으로 덜 반영됨
- 문서 구분에 유용한 단어가 문장 임베딩에서 더 눈에 띄게 됨

즉, TF-IDF는 "단어 의미 벡터"에 "현재 문서에서의 중요도"를 곱해주는 역할을 한다.

## 5. 하지만 TF-IDF 가중 평균도 해결하지 못하는 것이 있다

이 부분은 매우 중요하다.

TF-IDF 가중 평균은 단순 평균보다 나아질 수 있지만, 모든 문제를 해결하는 것은 아니다.

특히 다음 한계는 여전히 남아 있다.

### 1. 단어 순서를 직접 반영하지 못한다

예를 들어:

- `고양이가 강아지를 쫒는다`
- `강아지가 고양이를 쫒는다`

는 의미가 다르지만, 둘 다 같은 단어 집합으로 구성되어 있다면 TF-IDF 값도 비슷하게 나올 수 있다.

즉:

- 단순 평균도 순서를 모름
- TF-IDF 가중 평균도 기본적으로 순서를 모름

이다.

따라서 TF-IDF는 단어 중요도 문제를 보완할 수는 있지만, 문법 구조나 단어 순서 자체를 이해하는 방법은 아니다.

### 2. 문맥 자체를 깊게 이해하지는 못한다

TF-IDF는 통계적 중요도이고, GloVe는 정적 단어 임베딩이다.

따라서:

- 부정 표현
- 문장 전체의 논리 구조
- 단어 순서에 따른 의미 변화

같은 것은 충분히 반영하지 못할 수 있다.

이런 문제를 더 잘 다루려면:

- RNN
- LSTM
- Transformer
- BERT

같이 순서와 문맥을 반영하는 모델이 필요하다.

## 6. Yelp 실습 코드에서 TF-IDF는 어떻게 구해지는가

노트북에서는 `TfidfVectorizer`를 사용한다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in data['preprocessed']])
tfidf_feature_names = vectorizer.get_feature_names_out()
```

여기서 중요한 점은:

- `data['preprocessed']`는 토큰 리스트 형태다
- `TfidfVectorizer`는 문자열 문서 입력을 기대하므로
- `[' '.join(doc) for doc in data['preprocessed']]`로 다시 문장 문자열로 바꿔 넣는다

이 결과로:

- 행(row): 각 문서(각 리뷰 문장)
- 열(column): 전체 corpus에서 등장한 단어들
- 값(value): 해당 문서에서 해당 단어의 TF-IDF 값

을 갖는 행렬이 만들어진다.

즉, `tfidf_matrix[d, t]`는 "문서 `d`에서 단어 `t`의 중요도"라고 이해하면 된다.

## 7. 특정 단어의 TF-IDF 값은 코드에서 어떻게 찾는가

노트북에서는 단어 인덱스를 찾아 해당 값을 직접 조회한다.

```python
word_idx = np.where(tfidf_feature_names == word)[0][0]
value = tfidf_matrix.toarray()[doc_idx][word_idx]
```

이 코드는 다음 의미를 가진다.

1. 전체 단어 목록에서 현재 단어의 열 위치를 찾는다
2. 현재 문서 행에서 그 열 값을 가져온다
3. 그 값이 그 단어의 TF-IDF 중요도다

즉:

- `glove[word]`는 단어의 의미 벡터
- `tfidf_matrix[doc_idx, word_idx]`는 그 단어의 현재 문서 중요도

를 뜻한다.

## 8. 문장 임베딩 함수는 어떤 원리로 동작하는가

노트북의 핵심 함수는 다음 흐름으로 이해할 수 있다.

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

이 코드를 단계별로 풀어쓰면 다음과 같다.

### 1. 문장 안 단어를 하나씩 확인한다

```python
for word in doc:
```

여기서 `doc`는 전처리된 한 문장의 단어 리스트다.

### 2. GloVe와 TF-IDF 양쪽에 존재하는 단어만 사용한다

```python
if word in glove and word in tfidf_feature_names:
```

즉:

- GloVe vocabulary에 없는 단어
- TF-IDF 단어장에 없는 단어

는 건너뛴다.

### 3. 현재 문서에서 그 단어의 중요도를 가져온다

```python
word_idx = np.where(tfidf_feature_names == word)[0][0]
tfidf_weight = tfidf_matrix.toarray()[doc_idx, word_idx]
```

이 값은 "이 문장에서 이 단어가 얼마나 중요한가"를 뜻한다.

### 4. 단어 의미 벡터에 중요도를 곱한다

```python
glove[word] * tfidf_weight
```

즉:

- GloVe는 단어의 의미를 주고
- TF-IDF는 그 의미를 현재 문장에서 얼마나 세게 반영할지 정한다

### 5. 가중 벡터들을 평균내 문장 벡터를 만든다

```python
np.mean(embeddings, axis=0)
```

이 결과가 해당 문장의 최종 임베딩이다.

## 9. 예시로 다시 보기

문장:

```text
고양이가 강아지를 쫒는다
```

단순 평균 방식:

```text
GloVe(고양이가)
GloVe(강아지를)
GloVe(쫒는다)
-> 평균
```

TF-IDF 가중 평균 방식:

```text
GloVe(고양이가) * TF-IDF(고양이가)
GloVe(강아지를) * TF-IDF(강아지를)
GloVe(쫒는다) * TF-IDF(쫒는다)
-> 평균
```

차이는 다음과 같다.

- 단순 평균: 세 단어를 모두 똑같이 취급
- TF-IDF 가중 평균: 더 중요한 단어가 더 크게 반영됨

즉, 두 방식 모두 단어 벡터를 평균내지만, TF-IDF 방식은 평균 전에 "가중치"를 주는 단계가 추가된다.

## 10. 감성 분석에서 왜 특히 유용한가

감성 분석에서는 모든 단어가 같은 중요도를 갖지 않는다.

예를 들어:

- `amazing`
- `terrible`
- `worst`
- `excellent`

같은 단어는 감정을 강하게 드러낸다.

반면:

- `place`
- `food`
- `staff`

같은 단어는 문맥에 따라 중요할 수도 있지만, 모든 리뷰에서 자주 등장할 경우 감정 구분력은 떨어질 수 있다.

그래서 TF-IDF를 곱하면 감정을 더 잘 설명하는 단어가 문장 임베딩에 더 크게 반영될 가능성이 높아진다.

## 11. 단순 평균 vs TF-IDF 가중 평균

| 방식 | 핵심 아이디어 | 장점 | 한계 |
| --- | --- | --- | --- |
| 단순 평균 | 모든 단어 벡터를 같은 비중으로 평균냄 | 구현이 쉽고 빠름 | 중요한 단어와 덜 중요한 단어를 구분하지 못함 |
| TF-IDF 가중 평균 | 단어 벡터에 중요도 가중치를 곱한 뒤 평균냄 | 문서 구분에 유용한 단어를 더 강하게 반영 가능 | 여전히 단어 순서와 문맥 구조는 직접 반영하지 못함 |

## 12. 핵심 요약

- GloVe는 단어 의미를 주지만, 현재 문장에서 어떤 단어가 더 중요한지는 알려주지 않는다.
- 단순 평균은 모든 단어를 같은 비중으로 취급하기 때문에 감정에 중요한 단어가 묻힐 수 있다.
- TF-IDF를 곱하면 문서 안에서 더 중요한 단어가 문장 임베딩에 더 크게 반영된다.
- 이 방식은 전통적인 문장 임베딩 방법으로 유용하지만, 문맥과 단어 순서를 직접 이해하는 것은 아니다.

## 13. 정리

이 실습에서 TF-IDF가 필요한 이유는 단순하다.

- GloVe는 단어 의미를 준다
- 하지만 어떤 단어가 현재 문장에서 더 중요한지는 알려주지 않는다
- 단순 평균은 모든 단어를 똑같이 본다
- TF-IDF는 단어 중요도를 수치로 주어, 문장 임베딩에서 중요한 단어를 더 강조해준다

즉, 이 실습의 문장 임베딩은 다음 두 정보를 합친 것이다.

- `GloVe`: 단어의 의미
- `TF-IDF`: 문서 안에서의 상대적 중요도

다만 꼭 기억해야 할 점은:

- TF-IDF 가중 평균은 단순 평균보다 더 똑똑한 요약 방식이지만
- 여전히 단어 순서와 문맥 구조를 완전히 이해하는 방법은 아니다

그래서 이 방식은 "전통적인 문장 임베딩 방법"으로 이해하면 좋고, 이후에는 BERT 같은 문맥 기반 모델로 자연스럽게 이어질 수 있다.
