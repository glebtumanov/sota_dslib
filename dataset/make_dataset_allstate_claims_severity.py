#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore')

# Определяем пути к директориям
SRC_DIR = 'allstate-claims-severity'
DST_DIR = 'allstate-claims-severity'

# Проверяем существование директорий
if not os.path.exists(SRC_DIR):
    print(f"Директория источника {SRC_DIR} не существует.")
    exit(1)

if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)
    print(f"Создана директория назначения {DST_DIR}")

# Загружаем данные
print("Загрузка данных...")
train_path = os.path.join(SRC_DIR, 'train.csv')
test_path = os.path.join(SRC_DIR, 'test.csv')

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    print(f"Файлы {train_path} или {test_path} не найдены.")
    exit(1)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Определение категориальных и числовых признаков
print("Определение категориальных и числовых признаков...")
categorical_features = [col for col in train_df.columns if col.startswith('cat')]
continuous_features = [col for col in train_df.columns if col.startswith('cont')]
all_features = categorical_features + continuous_features

# Кодирование категориальных признаков
print("Кодирование категориальных признаков...")
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    # Объединяем значения из обучающего и тестового наборов для обучения энкодера
    le.fit(pd.concat([train_df[feature], test_df[feature]]).astype(str))

    # Применяем энкодер к данным
    train_df[feature] = le.transform(train_df[feature].astype(str))
    test_df[feature] = le.transform(test_df[feature].astype(str))

    # Сохраняем энкодер для возможного использования в будущем
    label_encoders[feature] = le

# Сохраняем результаты в формате parquet
print("Сохранение данных в формат parquet...")
train_parquet_path = os.path.join(DST_DIR, 'train.parquet')
test_parquet_path = os.path.join(DST_DIR, 'test.parquet')

train_df.to_parquet(train_parquet_path, index=False)
test_df.to_parquet(test_parquet_path, index=False)

# Записываем списки признаков в текстовые файлы
print("Запись списков признаков в текстовые файлы...")
features_path = os.path.join(DST_DIR, 'features.txt')
categorical_features_path = os.path.join(DST_DIR, 'categorical_features.txt')

with open(features_path, 'w') as f:
    f.write('\n'.join(all_features))

with open(categorical_features_path, 'w') as f:
    f.write('\n'.join(categorical_features))

# Создаем информационный файл
print("Создание информационного файла...")
info_path = os.path.join(DST_DIR, 'info.txt')

with open(info_path, 'w') as f:
    f.write(f"Ссылка на скачивание датасета: https://www.kaggle.com/c/allstate-claims-severity/data\n")
    f.write(f"Train shape: {train_df.shape[0]}, {train_df.shape[1]}, Test shape: {test_df.shape[0]}, {test_df.shape[1]}\n")
    f.write(f"Total features: {len(all_features)}, Categorical features: {len(categorical_features)}\n")
    f.write(f"Target column: loss, type: regression\n")

print("Обработка завершена!")
print(f"Данные сохранены в {DST_DIR}:")
print(f" - {train_parquet_path}")
print(f" - {test_parquet_path}")
print(f" - {features_path}")
print(f" - {categorical_features_path}")
print(f" - {info_path}")