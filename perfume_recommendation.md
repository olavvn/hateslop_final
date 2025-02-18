# **향수 추천 시스템: What’s in My Scent?**
### **STS 기반 향수 추천 시스템**
📌 **최종 프로젝트 보고서 - 김도훈**

---

## **📌 목차**
1. **소개 (Introduction)**
   - 연구 배경
   - 연구 목표
2. **학습 과정 (Training)**
   - 원본 데이터 및 전처리
   - FastText 학습
   - Sentence Transformer 학습
   - Cross Encoder 학습
3. **결과 (Results)**
   - Sentence Transformer 평가
   - Cross Encoder 평가
4. **응용 (Application)**
   - 향수 검색 시스템 구성
   - 검색 알고리즘
   - 데모 링크
5. **추가 연구 (Further Research)**
   - 오버피팅 완화
   - 웹 서비스 제공
   - 코드 최적화
6. **참고문헌 (References)**

---

## **📌 1. 소개 (Introduction)**

### **🔹 연구 배경**
- 후각(olfaction)은 다른 감각과 비교해 연구가 부족하며, 언어적 표현이 미비함 *(서종석, 2012)*.
- 향의 지각적 관계를 유지하면서 새로운 향의 특성을 예측할 수 있는 **Principal Odor Map(POM)** 개발 *(Brian K. Lee et al., 2022)*.
- 향수 선택은 개인적인 경험과 주관적 취향이 반영되므로, 기존의 추천 시스템보다 더 정밀한 모델이 필요함.

### **🔹 연구 목표**
- **입력**: 사용자의 **모호한 향수 표현**  
- **출력**: 향수의 **명확한 추천 제품 리스트**  
- **사용 기술**: **Sentence Transformer 기반 STS(Semantic Text Similarity) 모델**을 활용하여 향수 추천 정확도를 높임.

---

## **📌 2. 학습 과정 (Training)**

### **🔹 원본 데이터**
- **사용 데이터셋**: `olgagmiufana1/fragrantica-com-fragrance-dataset/fra_cleaned.csv`
- **향수 및 설명 데이터(PND, Perfume and Description) 활용**  
  - 기존 연구 *(Jooyoung Kim et al.)* 에서 수집한 데이터
  - 향수의 성분(노트)과 사용자의 서술적 설명 포함

### **🔹 데이터 전처리 (Data Preprocessing)**
**📌 주요 데이터셋 구성**
- `fra_cleaned`: 원본 데이터 정리  
- `fragrantica_database`: 향수 노트와 설명 통합  
- `pnd_examples`: 기존 향수 설명 데이터  
- `pnd_gpt`: GPT-4o-mini를 이용해 생성한 추가 설명 데이터  
- `prediction_train`: 향수 설명과 노트를 연결하는 학습 데이터  
- `training_pairs`: 긍정/부정 학습 쌍 생성  
- `corpus`: 문장 임베딩을 위한 텍스트 데이터  

**📌 주요 전처리 과정**
1. **랜덤 샘플링 (Random Sampling)**:  
   - `fragrantica_database.csv`에서 300개 이상의 샘플 선택  
2. **GPT-4o-mini를 이용한 설명 생성**:  
   - 기존 향수 설명과 일관된 스타일로 새로운 설명 추가  
3. **향수 노트 표준화**:  
   - 중복되는 향 노트 정리 및 표준화  

---

### **🔹 FastText 학습**
- **목적**: 향수 노트 간의 유사도를 학습하여, 추천 정확도를 높임.
- **방식**:
  1. `prediction_train`에서 향수 설명을 추출하여 `corpus` 생성
  2. Gensim의 **FastText 모델**을 사용하여 학습 (Word2Vec보다 향상된 방식)

---

### **🔹 Sentence Transformer 학습**
- **목적**: 향수 설명을 바탕으로 특정 향수 노트를 예측하는 신경망 구축.
- **모델 구조**:
  - `sentence-transformers/all-MiniLM-L6-v2` 기반으로 문장 내 의미적 관계를 학습.
- **학습 데이터 구성**
  - **긍정 페어 (Positive Pair)**: 향수 설명과 실제 포함된 노트  
  - **부정 페어 (Negative Pair)**: 향수 설명과 포함되지 않은 노트  

---

### **🔹 Cross Encoder 학습**
- **목적**: 문장 간 유사도를 보다 정확하게 계산하도록 최적화.
- **사용 모델**: `cross-encoder/stsb-roberta-large`
- **학습 데이터**:
  - STS(Semantic Text Similarity) 데이터셋 활용
  - Sentence Transformer 학습 데이터 활용

---

## **📌 3. 결과 (Results)**

### **🔹 Sentence Transformer 평가**
- **코사인 유사도(Cosine Similarity) 기준 정확도 비교**:
  ```
  Version 3 > Version 2 > Version 1 > Version 4
  ```
- **문제점**:
  - 데이터 다양성 부족 → 일부 모델에서 **과적합(Overfitting) 발생**

### **🔹 Cross Encoder 평가**
- **STS 데이터셋을 활용한 정확도 비교**:
  ```
  Version 3 > Version 2 > Version 1 > Version 4
  ```

---

## **📌 4. 응용 (Application)**

### **🔹 향수 검색 시스템 구성**
- **입력**: 브랜드, 국가, 성별, 향수 설명  
- **출력**: 적절한 향수 추천 리스트  
- **검색 과정**:
  1. 사용자의 입력 데이터를 Transformer v4 모델을 통해 벡터화
  2. **FAISS (Facebook AI Similarity Search)** 를 활용하여 유사 향수 검색
  3. Cross Encoder를 이용한 추가 필터링

---

### **🔹 데모 링크**
📌 [Google Colab](https://colab.research.google.com/drive/1oUCJ_aEKqFQh1j8k2T4ImHL136DeLQsK?usp=sharing)

---

## **📌 5. 추가 연구 (Further Research)**

1. **오버피팅 완화**:
   - 향수 설명 데이터의 다양성 증가
   - 학습 데이터 증강 (Data Augmentation) 기법 추가  
2. **웹 서비스 제공**:
   - 벡터 데이터베이스(VectorDB) 구축  
   - FastAPI를 활용한 API 서비스 개발  
   - [웹사이트 예시](https://olavvn.github.io/pour_monsieur_web/)  
3. **코드 최적화**:
   - 객체 지향 프로그래밍(OOP) 방식 개선  
   - **RAG (Retrieval-Augmented Generation) 모델 개선**  

---

## **📌 6. 참고문헌 (References)**
- **Kim, Jooyoung et al. (2024).** *NLP-Based Perfume Note Estimation Based on Descriptive Sentences*. [DOI](https://doi.org/10.3390/app14209293)
- **Brian K. Lee et al. (2022).** *A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception*. [DOI](https://doi.org/10.1101/2022.09.01.504602)
