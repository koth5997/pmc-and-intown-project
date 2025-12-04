# ==========================================
# 1. 환경 설정 및 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping


sns.set(style='whitegrid', palette='muted', font_scale=1.2)

csv_path = 'table_name_202505141345_1.csv' 

try:
    df = pd.read_csv(csv_path)
    print(f"데이터 로드 성공! Shape: {df.shape}")
except FileNotFoundError:
    print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()


drop_cols = [
    'MECHNO', 'MainPGM', 'RunPGM', 'SEQ', 'PRC_GB', 'AutoManual',
    'Emergency', 'Alarm',
    'ToolCounter4T', 'ToolCounter5T', 'ToolCounter6T', 'ToolCounter7T', 'ToolCounter8T',
    'ToolCounter5N', 'ToolCounter6N', 'ToolCounter7N', 'ToolCounter8N',
    'ToolCounter9N', 'ToolCounter10N', 'Feed', 'SpindleRPM', 'C_AsixPos', 'A_AsixPos',
    'PRC_DATE', 'Update_time'
]

available_drops = [col for col in df.columns if col in drop_cols]
df = df.drop(columns=available_drops)

df = df.ffill().bfill()



train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)


scaler = MinMaxScaler()
scaler.fit(train_df)

# 학습된 Scaler로 Train/Test 각각 변환
train_scaled = scaler.transform(train_df)
test_scaled = scaler.transform(test_df)

print(f"Train Dataset shape: {train_scaled.shape}")
print(f"Test Dataset shape: {test_scaled.shape}")

def create_sequences(X, window_size=20):

    Xs = []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
    return np.array(Xs)

WINDOW_SIZE = 20  # 과거 20개의 데이터를 보고 패턴 학습

# 학습용 시퀀스 생성
X_train = create_sequences(train_scaled, window_size=WINDOW_SIZE)
X_test = create_sequences(test_scaled, window_size=WINDOW_SIZE)

print(f"X_train sequence shape: {X_train.shape}")
print(f"X_test sequence shape: {X_test.shape}")



n_features = X_train.shape[2] # 특성 개수

model = Sequential([
   
    Input(shape=(WINDOW_SIZE, n_features)),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.2), # 과적합 방지


    RepeatVector(WINDOW_SIZE),

   
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),

  
    TimeDistributed(Dense(n_features))
])

model.compile(optimizer='adam', loss='mse')
model.summary()


#모델학습
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

print("모델 학습 시작...")
history = model.fit(
    X_train, X_train, 
    epochs=50,       
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    shuffle=False
)

# 학습 곡선 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=1) # 샘플별 평균 오차


threshold = np.mean(train_mae_loss) + 2 * np.std(train_mae_loss)
print(f"Anomaly Threshold (임계값): {threshold:.4f}")

plt.figure(figsize=(10, 6))
sns.histplot(train_mae_loss, bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution (Train Data)')
plt.legend()
plt.show()

# 4) Test 데이터에 대한 예측 및 이상 탐지
test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=1)


anomalies = test_mae_loss > threshold
print(f"검출된 이상 데이터 개수: {np.sum(anomalies)} / {len(anomalies)}")

# 5) 최종 결과 시각화 (Test 데이터의 오차 그래프)
plt.figure(figsize=(12, 6))
plt.plot(test_mae_loss, label='Test Reconstruction Error')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Anomaly Detection Result on Test Data')
plt.ylabel('Reconstruction Error (MAE)')
plt.xlabel('Time Step')
plt.legend()
plt.show()