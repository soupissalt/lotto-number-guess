import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import csv, os
from google.colab import files

# --- 색상 및 빈도 처리 ---
def get_number_color(n):
    if 1 <= n <= 10: return "yellow"
    elif 11 <= n <= 20: return "blue"
    elif 21 <= n <= 30: return "red"
    elif 31 <= n <= 40: return "gray"
    else: return "green"

def get_color_vector(numbers):
    color_map = {"yellow": 0, "blue": 1, "red": 2, "gray": 3, "green": 4}
    vec = [0] * 5
    for num in numbers:
        vec[color_map[get_number_color(num)]] += 1
    return vec

def get_number_frequency_window(data_window):
    freq = [0] * 45
    for round in data_window:
        for num in round:
            freq[num - 1] += 1
    return freq

# --- 로또 데이터 수집 ---
def fetch_lotto_data(start=1):
    all_data = []
    draw_no = start
    while True:
        try:
            url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
            res = requests.get(url)
            if res.status_code != 200 or 'drwtNo1' not in res.text:
                break
            j = res.json()
            numbers = [j[f"drwtNo{i}"] for i in range(1, 7)]
            all_data.append(numbers)
            draw_no += 1
        except:
            break
    return all_data

# --- 학습 데이터셋 구성 ---
def build_advanced_dataset(data, window_size=8):
    X, y = [], []
    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        target = data[i + window_size]
        freq_vec = get_number_frequency_window(window)
        x_seq = [round + get_color_vector(round) + freq_vec for round in window]
        X.append(x_seq)
        y.append(target)
    return np.array(X), np.array(y)

def preprocess(X, y):
    mlb = MultiLabelBinarizer(classes=range(1, 46))
    y_bin = mlb.fit_transform(y)
    return X, y_bin, mlb

# --- 모델 정의 ---
def build_lstm_model():
    model = Sequential([
        LSTM(128, input_shape=(8, 56)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(45, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- 예측 및 저장 함수 ---
def decode_prediction(pred, mlb, top_k=6):
    pred = pred.flatten()
    top_indices = np.argsort(pred)[-top_k:][::-1] + 1
    return sorted(top_indices.tolist())

def evaluate_and_save(predicted_numbers, actual_numbers, file_path="valid_results.csv"):
    match_count = len(set(predicted_numbers) & set(actual_numbers))
    if match_count >= 4:
        file_exists = os.path.exists(file_path)
        with open(file_path, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["predicted", "actual", "match_count"])
            writer.writerow([predicted_numbers, actual_numbers, match_count])
        print(f"✅ {match_count}개 일치 - 저장됨: {predicted_numbers}")
    else:
        print(f"❌ {match_count}개 일치: {predicted_numbers}")

def save_prediction_only(predicted_numbers, expected_round, file_path="pending_predictions.csv"):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["predicted", "expected_round"])
        writer.writerow([predicted_numbers, str(expected_round)])

def save_high_accuracy_predictions(predictions, actual_numbers, file_path="high_accuracy.csv"):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["predicted", "actual", "match_count"])
        for pred in predictions:
            match_count = len(set(pred) & set(actual_numbers))
            if match_count >= 5:
                writer.writerow([pred, actual_numbers, match_count])

def predict_with_dropout(model, input_seq, mlb, actual_numbers, repeat=1000):
    predictions = []
    success_count = 0
    for _ in range(repeat):
        pred = model(input_seq, training=True).numpy()
        numbers = decode_prediction(pred, mlb)
        predictions.append(numbers)
        if len(set(numbers) & set(actual_numbers)) >= 4:
            success_count += 1
    print(f"\n🎯 4개 이상 적중한 예측: {success_count} / {repeat}회")
    return predictions

def summarize_predictions(predictions):
    flat = [n for pred in predictions for n in pred]
    counts = Counter(flat)
    most_common = [num for num, _ in counts.most_common(6)]
    return sorted(most_common)

# --- 메인 실행 ---
def main():
    print("📦 로또 데이터 수집 중...")
    full_data = fetch_lotto_data()
    print(f"✅ {len(full_data)}개 회차 수집 완료")

    if len(full_data) < 9:
        print("❌ 학습에 필요한 데이터가 부족합니다.")
        return

    latest_available = len(full_data)
    expected_target_round = latest_available + 1

    X, y = build_advanced_dataset(full_data)
    X, y, mlb = preprocess(X, y)

    model = build_lstm_model()
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    chkpt = ModelCheckpoint("best_model.h5", save_best_only=True)
    model.fit(X, y, epochs=1000, batch_size=32, validation_split=0.2,
              callbacks=[early, chkpt], verbose=1)
    print("✅ 학습 완료")

    recent_window = full_data[-8:]
    freq_vec = get_number_frequency_window(recent_window)
    input_seq = [round + get_color_vector(round) + freq_vec for round in recent_window]
    input_seq = np.array(input_seq).reshape((1, 8, 56))

    try:
        actual = full_data[expected_target_round - 1]
    except IndexError:
        actual = []

    predicted_sets = predict_with_dropout(model, input_seq, mlb, actual, repeat=1000)
    final_prediction = summarize_predictions(predicted_sets)
    print(f"\n🔮 최종 예측 번호({expected_target_round}회차): {final_prediction}")

    if actual:
        evaluate_and_save(final_prediction, actual)
        save_high_accuracy_predictions(predicted_sets, actual)
    else:
        save_prediction_only(final_prediction, expected_target_round)

    for file in ["valid_results.csv", "pending_predictions.csv", "high_accuracy.csv"]:
        if os.path.exists(file):
            files.download(file)

# 실행
main()