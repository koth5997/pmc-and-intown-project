# MCT 공구 수명 예측 및 이상 탐지

> **Notice (보안 안내)**
> 본 프로젝트는 (주)제일피엠씨 및 (주)인타운과의 산학협력 과제로 수행되었습니다.
> 기업의 기술 보안 및 데이터 기밀 유지 협약(NDA)을 준수하기 위해, **원본 데이터셋은 공개하지 않으며** 제가 한 일부코드만 넣었습니다.
> 대신 **시스템 아키텍처, 핵심 알고리즘 설명, 그리고 최종 성과(시각화)**를 중심으로 기술합니다.

---

## 1.프로젝트 개요
제조 현장의 MCT 설비에서 수집되는 **고빈도 시계열 센서 데이터(전류, 진동)**를 분석하여, 공구의 마모 상태를 실시간으로 탐지하는 AI 모델을 개발했습니다. 라벨링 데이터가 부족한 현장 특성을 고려하여 **비지도 학습(Unsupervised Learning)** 기반의 이상 탐지 방법론을 적용했습니다.

## 2. 기술 스택
- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 3. 주요 기능 요약
1. CSV 데이터 로드 및 결측치 처리  
2. 필요 없는 컬럼 제거  
3. MinMaxScaler 기반 정규화  
4. 시계열 입력을 위한 Window Sequence 생성  
5. LSTM Autoencoder 모델 정의 및 학습  
6. Reconstruction Error 기반 이상(anomaly) 탐지  
7. 그래프 시각화

## 4. 데이터 처리 과정 (Preprocessing)
```python
df = df.drop(columns=available_drops)
df = df.ffill().bfill()
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)
```
## 5.LSTM Autoencoder 모델 구조
```
model = Sequential([
    Input(shape=(WINDOW_SIZE, n_features)),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.2),
    RepeatVector(WINDOW_SIZE),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(n_features))
])
```
Encoder: LSTM(64)
Bottleneck: RepeatVector
Decoder: LSTM(64)
Output: TimeDistributed(Dense)
손실 함수: MSE
EarlyStopping 사용

## 6.이상 탐지 방식
학습 데이터 Reconstruction Error 계산
평균 + 2×표준편차로 임계값(threshold) 설정
Test 데이터의 Reconstruction Error가 threshold를 초과하면 이상치로 판단
```
threshold = np.mean(train_mae_loss) + 2 * np.std(train_mae_loss)
anomalies = test_mae_loss > threshold
```
## 8. 코드 전체 흐름
```
데이터 로드 → 컬럼 정리 → 결측치 처리 
→ Train/Test 분리 → 정규화 → 시퀀스 생성
→ LSTM Autoencoder 학습 → Reconstruction Error 계산
→ Threshold 설정 → 이상 탐지 및 시각화
```

