# MCT 공구 수명 예측 및 이상 탐지

> **Notice (보안 안내)**
> 본 프로젝트는 (주)제일피엠씨 및 (주)인타운과의 산학협력 과제로 수행되었습니다.
> 기업의 기술 보안 및 데이터 기밀 유지 협약(NDA)을 준수하기 위해, **실제 소스 코드와 원본 데이터셋은 공개하지 않으며**
> 대신 **시스템 아키텍처, 핵심 알고리즘 설명, 그리고 최종 성과(시각화)**를 중심으로 기술합니다.

---

## 1.프로젝트 개요
제조 현장의 MCT 설비에서 수집되는 **고빈도 시계열 센서 데이터(전류, 진동)**를 분석하여, 공구의 마모 상태를 실시간으로 탐지하는 AI 모델을 개발했습니다. 라벨링 데이터가 부족한 현장 특성을 고려하여 **비지도 학습(Unsupervised Learning)** 기반의 이상 탐지 방법론을 적용했습니다.

* **Role:** AI Modeling Lead (모델 설계, 학습, 최적화 전담)
* **Tech Stack:** Python, TensorFlow(Keras), LSTM, Autoencoder

---

## 2.핵심 기술 및 로직

코드를 공개할 수 없는 대신, 제가 직접 설계하고 구현한 핵심 로직을 상세히 설명합니다.

### A. LSTM Autoencoder Modeling
시계열 데이터의 '정상 패턴'을 학습하여, 이 패턴에서 벗어나는 정도(Reconstruction Error)를 측정하는 방식을 사용했습니다.

* **Encoder:** 시계열 입력($t_1 \dots t_n$)을 받아 압축된 Latent Vector로 변환 (LSTM Layer 사용)
* **Decoder:** 압축된 정보를 다시 원래의 시계열로 복원
* **Thresholding:** 정상 데이터의 복원 오차 분포(Distribution)를 분석하여 이상 탐지 임계값 설정

### B. Signal Processing (전처리 파이프라인)
현장 데이터의 노이즈를 제거하기 위해 다음과 같은 전처리를 적용했습니다.
* **High Pass Filter:** 설비 자체 진동에 의한 저주파 노이즈 제거
* **STFT (Short-Time Fourier Transform):** 시간-주파수 도메인 특징 추출
* **Sliding Window:** 실시간 스트리밍 데이터를 모델 입력 시퀀스로 변환

---

## 3. Results & Visualization (결과 및 시각화)

### Model Training & Evaluation
학습이 진행됨에 따라 Loss가 안정적으로 수렴하였으며, 정상 데이터에 대한 복원 오차 분포를 통해 정밀한 임계치를 설정했습니다.

![Model Evaluation](./assets/model_loss_graph.png)

### Anomaly Detection Result
개발된 모델은 공구 마모가 진행되는 구간에서 **재구성 오차(Reconstruction Error)가 급격히 증가**하는 패턴을 정확하게 탐지했습니다.

* **Prediction Accuracy (R²):** `0.9463`
* **RMSE:** `131.89`

![Prediction Result](./assets/prediction_result.png)

### Real-time Dashboard (Grafana)
현장 관리자가 모델의 추론 결과를 직관적으로 확인할 수 있도록 시각화 대시보드를 구축했습니다.

![Dashboard](./assets/dashboard_view.png)

---

## 4. Retrospective (배운 점)
* **데이터 관점:** 실제 제조 데이터의 불균형(Imbalance) 문제를 해결하기 위해 정상 데이터만으로 학습하는 오토인코더 방식을 채택하여 유효성을 입증했습니다.
* **엔지니어링 관점:** 단순 모델링을 넘어, 노이즈 필터링부터 시각화 연동까지 전체 데이터 파이프라인의 흐름을 이해하는 계기가 되었습니다.
