---
title: "A/B 테스트 버킷팅 실습 해설"
date: 2026-03-28 16:35:00 +0900
categories: [Data Analysis, ABTest]
tags: [ab-test, lesson-02, bucketing]
description: "버킷팅 실습 노트북 흐름을 따라 해시 분배와 A/A 검증 과정을 설명한 해설 노트"
math: false
---
# A/B 테스트 버킷팅 실습 해설

## 1. 이 문서는 무엇을 설명하는가

이 문서는 [A_B_Test_02강_버킷팅_실습.ipynb](/assets/files/notebooks/ab_test/A_B_Test_02%EA%B0%95_%EB%B2%84%ED%82%B7%ED%8C%85_%EC%8B%A4%EC%8A%B5.ipynb) 흐름을 설명하는 해설 노트다.

노트북은 단순히 코드를 실행하는 실습이 아니라,  
아래 내용을 순서대로 익히도록 구성되어 있다.

- 실험용 원천 데이터를 읽는 방법
- `user_id`를 해시해서 A/B 버킷으로 나누는 기본 아이디어
- 그 기본 코드가 왜 실무에서는 부족한지
- SQL과 Python 양쪽에서 A/A 분포를 확인하는 방법
- 런타임 시스템에 더 가까운 버킷팅 함수가 어떤 모습인지

## 2. 노트북 전체 흐름 한눈에 보기

노트북 흐름은 크게 여섯 단계다.

1. Redshift에서 실습 데이터 읽기
2. Python으로 가장 단순한 해시 버킷팅 이해
3. 실무형 버킷팅의 한계와 개선 방향 이해
4. 같은 버킷팅 로직을 SQL로 표현
5. A/A Test를 SQL과 Python으로 각각 확인
6. Production에 더 가까운 버킷팅 함수 형태 보기

즉, 이 노트북은  
`해시 버킷팅 개념 이해 -> 실무 적용 -> 검증`  
순서로 짜여 있다고 보면 된다.

## 3. 1단계: Redshift 원천 데이터 읽기

맨 앞에서는 `raw_data.aa_example` 테이블을 읽는다.

테이블 컬럼은 다음과 같다.

- `user_id`
- `date`
- `job_position_id`
- `clicked`
- `checkedout`
- `applied`

이 데이터는 일별 사용자 행동 로그이며,  
이후 실습에서:

- 사용자 수
- 세션 수
- 전환 이벤트

같은 값을 계산하는 기반이 된다.

즉, 여기서는  
"A/B 실험 분석에 쓸 수 있는 원천 로그가 이런 식으로 생겼다"  
를 먼저 익히는 단계다.

## 4. 2단계: Python으로 기본 버킷팅 함수 이해

`Python` 섹션에서는 가장 단순한 형태의 버킷팅 함수가 나온다.

```python
def split_userid(id):
    h = hashlib.md5(str(id).encode())
    return int(h.hexdigest(), 16) % 2
```

이 함수의 의미는 단순하다.

- `user_id`를 문자열로 바꾼다
- MD5 해시를 만든다
- 큰 정수로 바꾼다
- `% 2`를 해서 `0` 또는 `1`을 만든다

즉, 사용자를 A/B 두 버킷 중 하나로 나누는 가장 기본적인 코드다.

이 파트의 핵심은  
`같은 사용자에게 항상 같은 버킷을 주는 결정적 분배`를 이해하는 것이다.

## 5. 3단계: 왜 이 기본 코드는 실무에서 부족한가

노트북 중간에 들어간 학습 셀은  
이 단순 함수가 개념 설명용으로는 좋지만  
실무용으로는 한계가 있다는 점을 정리한다.

핵심 한계는 세 가지다.

- 모든 사용자를 항상 실험에 태운다
- 어떤 실험을 하든 같은 사용자에게 같은 패턴이 반복될 수 있다
- 실험별 독립성이 약하다

예를 들어:

- 실무에서는 보통 1% -> 5% -> 10%처럼 점진적으로 rollout 한다
- 그런데 기본 함수는 항상 50:50으로 모든 사용자를 나눈다

즉, 이 함수는  
`버킷 번호 계산`은 해주지만  
`실험 운영`까지는 반영하지 못한다.

## 6. 실무형 버킷팅 예시가 추가된 이유

노트북에는 이 한계를 보완하기 위해 `split_userid_v2(...)`가 들어가 있다.

```python
def split_userid_v2(abtest_id, user_id, size_of_test, num_of_variants=2):
    mixed_id = f"{abtest_id}_{user_id}"
    h = hashlib.md5(mixed_id.encode())
    hashed_value = int(h.hexdigest(), 16)

    if (hashed_value % 100) >= size_of_test:
        return -1

    return hashed_value % num_of_variants
```

이 함수는 두 단계를 처리한다.

1. 이 사용자를 이번 실험에 포함할지 결정
2. 포함된다면 어떤 variant에 넣을지 결정

즉, 단순 A/B 분배가 아니라  
`실험 대상 선정 + variant 배정`까지 함께 처리한다.

## 7. `abtest_id`를 왜 같이 섞는가

`user_id`만 해시하면 같은 사용자는 여러 실험에서 비슷한 패턴으로 배정될 수 있다.

예:

- 어떤 사용자는 여러 실험에서 계속 A만 볼 수 있고
- 어떤 사용자는 여러 실험에서 계속 B만 볼 수 있다

그래서 `abtest_id`를 같이 섞어:

- 같은 사용자라도
- 실험 1과 실험 2에서는
- 서로 다른 해시 입력을 갖게 만든다

즉, `사용자 고정성`은 유지하되  
`실험 간 독립성`을 더 높이는 목적이다.

노트북에 들어간 비교 셀은 이 차이를 눈으로 확인하게 해준다.

## 8. 4단계: SQL로 같은 아이디어 표현하기

다음 `SQL` 섹션은  
Python에서 이해한 해시 버킷팅을 SQL에서 어떻게 구현하는지 보여준다.

예:

```sql
SELECT 100 user_id, MOD(STRTOL(LEFT(MD5(100),15), 16), 2) variant_id
```

핵심은 같다.

- `MD5`로 해시값 생성
- 해시를 숫자로 바꿔
- `MOD(..., 2)`로 A/B 버킷 생성

즉, Python 코드와 SQL 코드가  
본질적으로 같은 일을 한다는 점을 연결해주는 파트다.

## 9. 5단계: A/A Test 비교 - SQL

이제부터는  
"이 버킷팅이 실제로 대략 균등하게 나뉘는가?"  
를 검증하는 단계다.

먼저 SQL로 variant별 집계를 계산한다.

```sql
SELECT
    MOD(STRTOL(LEFT(MD5(user_id),15),16),2) variant_id,
    COUNT(DISTINCT user_id) user_sum,
    COUNT(1) session_sum
FROM (
    SELECT DISTINCT user_id, date
    FROM raw_data.aa_example
)
GROUP BY 1;
```

여기서 보는 것은 주로 두 가지다.

- `user_sum`
- `session_sum`

즉:

- 사용자 수가 A/B에서 크게 차이 나지 않는가
- 세션 수도 한쪽에 치우치지 않는가

를 먼저 점검하는 것이다.

이게 바로 A/A Test의 기본 검증이다.

## 10. 왜 A/A Test를 먼저 보나

A/A Test는 실제로는 같은 시스템을 두 그룹에 나누어 보는 검증이다.

이 경우 원칙적으로는 큰 차이가 나오면 안 된다.

그래서 A/A Test는:

- 버킷팅이 정상적인지
- 로그 수집이 잘 되는지
- 분석 쿼리가 이상 없는지

를 확인하는 QA 성격의 테스트라고 보면 된다.

즉, 이 노트북에서 SQL A/A Test 파트는  
`버킷팅이 최소한 크게 비틀어져 있지는 않은가`를 보는 단계다.

## 11. 6단계: A/A Test 비교 - Python

그다음은 같은 검증을 Python으로 다시 해본다.

먼저 사용자 목록만 따로 읽어온다.

```python
sql = """
SELECT DISTINCT user_id
FROM raw_data.aa_example
"""
user_df = sqlio.read_sql_query(sql, conn)
```

그 다음 Python 루프로 직접 센다.

```python
for index, row in user_df.iterrows():
    if split_userid(row["user_id"]) == 0:
        a_user_count += 1
    else:
        b_user_count += 1
```

즉, SQL에서 하던 일을  
이번에는 Python 함수로 직접 검증하는 것이다.

이 파트의 의미는:

- Python 함수가 실제로 대략 50:50에 가깝게 나누는지 보기
- SQL 로직과 Python 로직이 개념적으로 같은지 확인하기

에 있다.

## 12. 왜 `user_df`를 따로 쓰게 바꿨는가

이 노트북은 중간중간 `df`를 여러 번 다른 의미로 재사용한다.

예:

- 원천 로그 DataFrame
- SQL 집계 결과 DataFrame
- 사용자 목록 DataFrame

이렇게 `df`가 계속 바뀌면  
앞에서 기대한 컬럼이 뒤에서는 없어질 수 있다.

실제로 `KeyError: 'user_id'`가 났던 이유도  
`df`가 다른 구조의 DataFrame으로 덮인 상태에서  
`row["user_id"]`를 읽으려 했기 때문이다.

그래서 Python A/A Test 파트는  
`user_df`라는 별도 변수로 분리해 두는 편이 안전하다.

즉, 이 노트북을 읽을 때는  
`df가 항상 같은 의미가 아니다`  
라는 점을 기억해야 한다.

## 13. 세션 수를 따로 세는 이유

이후에는 `SELECT DISTINCT user_id, date`를 사용해  
세션 비슷한 단위의 개수를 센다.

즉:

- 사용자 수가 대략 비슷한지
- 세션 수까지도 대략 비슷한지

를 함께 확인한다.

사용자 수는 비슷해도  
한쪽 그룹에서 세션이 과도하게 많다면  
노출이나 집계 방식에 이상이 없는지 추가로 봐야 할 수 있다.

## 14. 맨 아래의 조금 더 완전한 버킷팅 함수는 무엇인가

맨 아래 `조금더 완전한 버킷팅 함수` 파트는  
앞쪽의 아주 단순한 `% 2` 함수보다  
Production runtime에 조금 더 가까운 형태를 보여주려는 목적이다.

```python
def split_userid(id, num_of_variants=2, b_percent=50):
     h = hashlib.md5(str(id).encode())
     val = int(h.hexdigest(), 16) % 100
     if val < b_percent:
       return 0
     else:
       return 1
```

이 함수의 핵심 차이는 `b_percent`다.

즉:

- 50이면 50:50
- 10이면 대략 10%만 한쪽 버킷
- 5면 대략 5%

처럼 비율을 조절할 수 있다.

이건 실제 rollout에서:

- 1%
- 5%
- 10%
- 50%

처럼 점진적으로 커버리지를 늘리는 생각과 연결된다.

다만 이 함수도 완전한 production 함수는 아니다.

이유:

- `abtest_id`가 없다
- 실험 미포함 그룹을 명시적으로 분리하지 않는다
- 실험 간 독립성까지는 충분히 반영하지 못한다

즉, 이 함수는  
`기본형과 실전형 사이의 중간 단계`로 이해하면 좋다.

## 15. 이 노트북에서 정말 배워야 하는 핵심

이 노트북의 핵심은 단순히 MD5 문법을 익히는 것이 아니다.

진짜 중요한 포인트는 아래다.

- 같은 사용자를 항상 같은 버킷에 넣는 결정적 분배가 필요하다
- 실무에서는 단순 50:50 분배만으로는 부족하다
- 실험별 독립성을 위해 `abtest_id` 같은 정보가 필요하다
- rollout을 위해 커버리지 개념이 필요하다
- SQL과 Python 양쪽에서 같은 버킷팅 로직을 해석할 수 있어야 한다
- A/A Test로 버킷팅과 로그가 정상인지 먼저 확인하는 습관이 중요하다

## 16. 이 노트북을 복습할 때 보면 좋은 질문

- 왜 같은 `user_id`는 항상 같은 버킷으로 가야 하는가
- 왜 `abtest_id`를 같이 섞어야 하는가
- 왜 모든 사용자를 바로 50:50으로 태우면 안 되는가
- 왜 SQL과 Python 두 방식으로 같은 로직을 확인하는가
- 왜 A/A Test에서 `User Size`와 `Session Size`를 먼저 보는가
- 왜 DataFrame 이름을 목적별로 분리하는 것이 중요한가

## 17. 한 줄 요약

이 노트북은  
`해시 기반 사용자 버킷팅의 기본 원리 -> 실무적 한계 -> 개선 방향 -> A/A 검증`  
을 SQL과 Python 양쪽에서 같이 익히도록 만든 실습 노트북이다.
