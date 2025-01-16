# 그래프 기반 향수 추천 시스템

### 김도훈

### 1. 목표
- 인풋: 사용자가 입력한 자연어 설명(예: "I want a floral perfume with jasmine and rose.").

- 아웃풋: 입력과 가장 유사한 향수 정보를 포함한 추천 결과.

- 추가 목표: 멀티모달 입력(예: 이미지, 리뷰)으로 확장 가능한 시스템 설계.
### 2. 데이터
#### 2.1 데이터 출처
- fragrantica.com에서 얻은 향수 데이터.
- kaggle에서 구함.
#### 2.1 데이터 구조


- {url;Perfume;Brand;Country;Gender;Rating Value;
Rating Count;Year;Top;Middle;Base;Perfumer1;Perfumer2;
mainaccord1;mainaccord2;mainaccord3;mainaccord4; mainaccord5}
의 정보가 포함됨.

- 약 24000개의 향수 데이터가 있음.
- 리뷰 요약, 사용자 피드백, 이미지 등의 확장 가능성.

### 3. 시스템 설계

#### 3.1 데이터 전처리

1. 텍스트 통합: 주요 필드(Brand, Top, Middle, Base, Main Accords)를 결합하여 자연어 임베딩의 입력으로 활용.

2. 결합 예시:

        def generate_description(data):
            data['description'] = data.apply(
                lambda row: (
                    f"{row['Perfume'].capitalize()} by {row['Brand'].capitalize()} is a {row['Gender']} fragrance featuring top notes of {row['Top']}, "
                    f"middle notes of {row['Middle']}, and base notes of {row['Base']}. "
                    f"The main accords are {', '.join(filter(pd.notna, [row['mainaccord1'], row['mainaccord2'], row['mainaccord3'], row['mainaccord4'], row['mainaccord5']]))}. "
                    f"Released in {row['Year']} from {row['Country']}, this fragrance has a rating of {row['Rating Value']} out of 5 from {row['Rating Count']} votes. "
                    f"{('Crafted by perfumer ' + row['Perfumer1'].capitalize() + '.') if pd.notna(row['Perfumer1']) else ''} {(' and ' + row['Perfumer2'].capitalize()) if pd.notna(row['Perfumer2']) else ''}"
                ),
                axis=1
            )
            return data
    description 예시

           'Peace love and juicy couture by juicy couture is a women fragrance featuring top notes of hyacinth, cassis, red apple, amalfi lemon, middle notes of red poppy, lime (linden) blossom, honeysuckle, jasmine, magnolia, and base notes of musk, patchouli, orris root. The main accords are floral, green, fruity, sweet, powdery. Released in 2010 from USA, this fragrance has a rating of 3.36 out of 5 from 1905 votes. Crafted by perfumer Rodrigo flores-roux. '
#### 3.2 그래프 구성

1. 노드 정의: 각 향수를 하나의 노드로 표현.

2. 엣지 정의: 향수 간의 유사도(예: 코사인 유사도) 기반 연결.

3. 그래프 구축 방법:

    - 노드: 향수 데이터에서 생성된 임베딩.

    - 엣지: 유사도 임계값(예: 0.6) 이상인 노드 간 연결.

#### 3.3 임베딩 생성

1. 텍스트 임베딩:

    - 모델: Sentence-BERT (예: all-MiniLM-L6-v2).
        - all-MiniLM-L6-v2
            - 성능과 속도의 균형: 
            <br>
            대형 모델(예: BERT, RoBERTa) 대비 약 2-3배 빠른 추론 속도를 제공하면서도, 텍스트 유사도 계산에서 우수한 성능을 유지
            - 메모리 효율성:
            <br>
                BERT 기반 모델보다 메모리 요구량이 적어, GPU(T4 등)에서 대규모 데이터 처리에 적합.
            <br>
                하나의 GPU에서 많은 문장을 배치로 처리할 수 있어 24000개 이상의 향수 데이터를 빠르게 임베딩 가능.
                
        - or gpt api 사용용

    - 각 향수의 설명을 입력으로 사용하여 고차원 벡터 생성.

2. 그래프 임베딩:

    - 기법: Node2Vec, GraphSAGE, 또는 DeepWalk.

    - 그래프의 구조적 정보를 반영한 노드 임베딩 생성.


### 4. 기술 스택

1. 프로그래밍 언어: Python

2. 필요 라이브러리:

    - 데이터 처리: pandas, numpy

    - 그래프: networkx, node2vec, py2neo

    - 딥러닝: sentence-transformers, sklearn

    - 데이터베이스: Neo4j (그래프 DB 저장)