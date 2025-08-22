1. 동행복권 공개 api를 1회차 부터 차례로 호출을 활용하여 최근 과거 8회차의 당첨번호를 입력 받음
2. 다음 회차의 6개 번호를 확률적으로 예측하는 LSTM 기반 파이프라인
3. 색상 분포, 최근 빈도 특징을 활용하여 Monte Carlo Dropout으로 불확실성을 여러 번 샘플링해 최빈값 집계로 최종 번호 6개를 뽑는 구조
4. Colab(T4) 환경에서 실행 할 수 있도록 만들어

1) 데이터 수집 (크롤링/API)
fetch_lotto_data()가 동행복권 공개 API를 회차 1부터 차례로 호출해 6개 당첨번호 리스트를 모읍니다.
한 회차는 [n1, n2, n3, n4, n5, n6] 형태로 저장되고, 더 이상 데이터가 없으면 반복을 멈춥니다.
결과: full_data = [[…,6개], […,6개], …] (전체 회차)

2) 특징 설계 (Feature Engineering)
예측력을 높이기 위해 원시 번호 외에 두 가지 보조 특징을 만듭니다.
색상 벡터:
1–10=yellow, 11–20=blue, 21–30=red, 31–40=gray, 41–45=green 으로 구간 색을 나눔
한 회차(6개 번호)에서 각 색이 몇 개 나왔는지 5차원 벡터로 표현 → get_color_vector(numbers)
최근 빈도 벡터(45차원):
입력 창(window)인 최근 8회차에서 각 번호(1~45)가 몇 번 등장했는지 카운트 → get_number_frequency_window(window)
한 타임스텝(한 회차)의 입력 벡터 구성:
6(원시번호) + 5(색상 분포) + 45(최근 빈도) = 총 56차원

3) 학습용 시퀀스 만들기 (슬라이딩 윈도우)
build_advanced_dataset(data, window_size=8)
입력 X: 최근 8회차(=타임스텝 8)의 각 회차를 56차원으로 만들어 시퀀스(8×56)로 쌓음 → LSTM에 들어갈 형태
정답 y: 바로 다음 회차의 6개 번호
preprocess(X, y)에서 MultiLabelBinarizer를 사용해 정답을 45차원 멀티-핫 벡터(6개의 위치가 1)로 변환
최종 형태: X.shape = (샘플수, 8, 56), y.shape = (샘플수, 45)

4) 모델 구조 (TensorFlow/Keras, Colab T4 호환)
build_lstm_model()
LSTM(128) → 시계열 패턴을 학습
Dropout(0.3) → 과적합 방지, 이후 추론 시 MC Dropout에도 활용
Dense(64, relu)
Dense(45, sigmoid) → 멀티라벨(6개 동시 예측)이라 softmax 대신 sigmoid + binary_crossentropy 사용
컴파일: loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']

5) 학습 전략
model.fit(..., epochs=1000, validation_split=0.2, callbacks=[EarlyStopping, ModelCheckpoint])
**EarlyStopping(patience=10)**으로 검증 손실이 좋아지지 않으면 조기 중단
ModelCheckpoint로 가장 좋은 가중치 저장
Colab의 T4 GPU에서 빠르게 학습되도록 설계 (TensorFlow/Keras 기본)

6) 추론(예측) + 불확실성 샘플링
입력: 최신 8회차로 (1, 8, 56) 시퀀스 구성
predict_with_dropout(..., repeat=1000)
추론 때도 training=True로 Dropout을 활성화 → Monte Carlo Dropout
1000번 반복 예측하여 매번 확률 상위 6개 번호를 뽑음(decode_prediction)
4개 이상 맞힌 시도 횟수를 집계해 모델 신뢰감을 가늠
summarize_predictions(predictions)
1000회 예측에서 나온 숫자들을 모두 합쳐 가장 자주 등장한 6개를 최종 번호로 선택(최빈값 집계)
