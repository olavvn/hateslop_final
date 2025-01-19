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

### 3. 시스템 설계 계획

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
#### ~~3.2 그래프 구성~~

1. 노드 정의: 각 향수를 하나의 노드로 표현.

2. 엣지 정의: 향수 간의 유사도(예: 코사인 유사도) 기반 연결.

3. 그래프 구축 방법:

    - 노드: 향수 데이터에서 생성된 임베딩.

    - 엣지: 유사도 임계값(예: 0.6) 이상인 노드 간 연결.

#### 3.3 임베딩 생성

1. 텍스트 임베딩:

    - 모델: Sentence-BERT (예: all-MiniLM-L6-v2) or OpenAI Encoder(text-embedding-ada-002).
        - all-MiniLM-L6-v2
            - 성능과 속도의 균형: 
            <br>
            대형 모델(예: BERT, RoBERTa) 대비 약 2-3배 빠른 추론 속도를 제공하면서도, 텍스트 유사도 계산에서 우수한 성능을 유지
            - 메모리 효율성:
            <br>
                BERT 기반 모델보다 메모리 요구량이 적어, GPU(T4 등)에서 대규모 데이터 처리에 적합.
            <br>
                하나의 GPU에서 많은 문장을 배치로 처리할 수 있어 24000개 이상의 향수 데이터를 빠르게 임베딩 가능.
                
        - 

    - 각 향수의 설명을 입력으로 사용하여 고차원 벡터 생성.

    ~~2. 그래프 임베딩:~~

    - 기법: Node2Vec, GraphSAGE, 또는 DeepWalk.

    - 그래프의 구조적 정보를 반영한 노드 임베딩 생성.


### 4. 기술 스택

1. 프로그래밍 언어: Python

2. 필요 라이브러리:

    - 데이터 처리: pandas, numpy

    - 그래프: networkx, node2vec, py2neo

    - 딥러닝: sentence-transformers, sklearn

    - 데이터베이스: Neo4j (그래프 DB 저장)

### 5.1 진행 과정 - 설명 임베딩과 유사도 측정을 통한 추천 시스템

#### 5.1.1 데이터 전처리
1. 텍스트 통합: 주요 필드(Brand, Top, Middle, Base, Main Accords)를 결합하여 자연어 임베딩의 입력으로 활용.

2. 결합 예시는 3.1.2의 전처리 방식 차용. 다음은 생성된 description의 예시

- description:

           'Peace love and juicy couture by juicy couture is a women fragrance featuring top notes of hyacinth, cassis, red apple, amalfi lemon, middle notes of red poppy, lime (linden) blossom, honeysuckle, jasmine, magnolia, and base notes of musk, patchouli, orris root. The main accords are floral, green, fruity, sweet, powdery. Released in 2010 from USA, this fragrance has a rating of 3.36 out of 5 from 1905 votes. Crafted by perfumer Rodrigo flores-roux. '

#### 5.1.2 임베딩 생성
1. Sentence-BERT (all-MiniLM-L6-v2)

    Sentence Transformer를 이용해 각 향수의 description을 임베딩으로 변환(768차원).
    <br>
    embeddings_index.faiss로 저장하고 metadata.json에 다른 metadata와 함께 저장.
    <br>
    이후 동일한 모델로 입력받은 query의 임베딩 생성한 뒤 search method로 유사도 높은 3가지 description 출력.
    
    - 장점: 시간이 덜 걸림, 이후 파인튜닝 할 수 있음, 무료임
    - 단점: 차원이 비교적 작음. 유사도가 낮게 측정됨. 정보 필터링 능력 부족

            Enter your query: tom ford perfume made earlier than 2015

        Result 1:
        Description: Tom ford for men by Tom ford is a men fragrance featuring top notes of lemon leaf oil, ginger, mandarin orange, bergamot, basil, violet leaf, middle notes of tobacco leaf, pepper, tunisian orange blossom, grapefruit blossom, and base notes of amber, cedar, vetiver, virginian patchouli, oakmoss, leather, cypriol oil or nagarmotha. The main accords are citrus, warm spicy, woody, fresh spicy, aromatic. Released in 2007 from USA, this fragrance has a rating of 4.04 out of 5 from 2313 votes. Crafted by perfumer Yves cassar. 
        Similarity Score: 0.32
        More information: https://www.fragrantica.com/perfume/tom-ford/tom-ford-for-men-1172.html

        Result 2:
        Description: London by Tom ford is a unisex fragrance featuring top notes of cumin, saffron, cardamom, black pepper, coffee, coriander, middle notes of incense, labdanum, jasmine, geranium, and base notes of agarwood (oud), birch, cedar, musk, amyris. The main accords are warm spicy, fresh spicy, amber, smoky, woody. Released in 2013 from USA, this fragrance has a rating of 4.0 out of 5 from 783 votes.  
        Similarity Score: 0.15
        More information: https://www.fragrantica.com/perfume/tom-ford/london-18883.html

        Result 3:
        Description: Noir by Tom ford is a men fragrance featuring top notes of violet, pink pepper, caraway, bergamot, verbena, middle notes of tuscan iris, bulgarian rose, black pepper, nutmeg, geranium, clary sage, and base notes of indonesian patchouli leaf, amber, vanilla, civet, leather, opoponax, benzoin, vetiver, styrax. The main accords are amber, powdery, fresh spicy, woody, violet. Released in 2012 from USA, this fragrance has a rating of 4.04 out of 5 from 3786 votes. Crafted by perfumer Olivier gillotin. 
        Similarity Score: 0.13
        More information: https://www.fragrantica.com/perfume/tom-ford/noir-15727.html

2. Sentence-BERT (all-MiniLM-L6-v2)

    Sentence Transformer를 이용해 각 향수의 description을 임베딩으로 변환(1536차원).
    <br>
    사용량으로 인하여 batch size: 50으로 나눠서 임베딩 변환 작업 진행.
    <br>
    embeddings_index_openai.faiss로 저장하고 metadata_openai.json에 다른 metadata와 함께 저장.
    <br>
    이후 동일한 모델로 입력받은 query의 임베딩 생성한 뒤 search method로 유사도 높은 3가지 description 출력.
    
    - 장점: 차원이 큼. 정확도 및 유사도가 높음.
    - 단점: 시간이 오래 걸림. 비용이 듦. 향후 파인튜닝 작업 제한. 정보 필터링 능력 아직 부족

            Enter your query: tom ford perfume made earlier than 2015

        
        Result 1:
        Description: Tom ford for men by Tom ford is a men fragrance featuring top notes of lemon leaf oil, ginger, mandarin orange, bergamot, basil, violet leaf, middle notes of tobacco leaf, pepper, tunisian orange blossom, grapefruit blossom, and base notes of amber, cedar, vetiver, virginian patchouli, oakmoss, leather, cypriol oil or nagarmotha. The main accords are citrus, warm spicy, woody, fresh spicy, aromatic. Released in 2007 from USA, this fragrance has a rating of 4.04 out of 5 from 2313 votes. Crafted by perfumer Yves cassar. 
        Similarity Score: 0.69
        More information: https://www.fragrantica.com/perfume/tom-ford/tom-ford-for-men-1172.html

        Result 2:
        Description: Fucking fabulous by Tom ford is a unisex fragrance featuring top notes of lavender, clary sage, middle notes of leather, bitter almond, vanilla, orris, and base notes of leather, tonka bean, cashmeran, white woods, amber. The main accords are aromatic, leather, vanilla, almond, amber. Released in 2017 from USA, this fragrance has a rating of 3.8 out of 5 from 6592 votes.  
        Similarity Score: 0.66
        More information: https://www.fragrantica.com/perfume/tom-ford/fucking-fabulous-46513.html

        Result 3:
        Description: Tobacco oud by Tom ford is a unisex fragrance featuring top notes of whiskey, middle notes of spicy notes, cinnamon, coriander, and base notes of tobacco, agarwood (oud), incense, sandalwood, patchouli, benzoin, vanilla, cedar. The main accords are warm spicy, tobacco, woody, whiskey, oud. Released in 2013 from USA, this fragrance has a rating of 4.2 out of 5 from 3433 votes. Crafted by perfumer Olivier gillotin. 
        Similarity Score: 0.66
        More information: https://www.fragrantica.com/perfume/tom-ford/tobacco-oud-21402.html
    
#### 5.1.3 결과 분석 및 해결 과제
1. 결과 분석
    - description에 명시 되어 있는 정보에 대하여는 판단을 양호하게 진행
    - description에 포함되어 있지 않은 정보에 대한 inference 능력 없음
    - 범주형 정보에 관한 판단이 잘 안 됨(ex. rating, launched year 등 범위에 관한 정보 등)
2. 해결 방안
    - feature을 늘림(가격, 계절감, 지속력 등등 몇가지 정보 추가(가능하다면))
    - Query의 의미를 이해하고, 이를 데이터 필터링 조건으로 변환하는 의미 기반 매핑 모델
    <br>
    (Semantic Mapping Model)을 학습 -> query를 json형태로 정규화한 뒤에 임베딩과 비교.
3. 기타 적용 사항
    - webpage의 검색 쿼리에 적용
        - url: https://olavvn.github.io/pour_monsieur_web/

