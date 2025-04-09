#!/usr/bin/env python3
# Dataset: Forest Cover Type Prediction
# https://www.kaggle.com/competitions/forest-cover-type-prediction/data
# Определение путей к директориям и файлам
import pandas as pd
import os
import sys

# Определение путей к директориям и файлам
SRC_DIR = './forest-cover-type'
DST_DIR = './forest-cover-type'

# Определение путей к файлам
csv_path = os.path.join(SRC_DIR, 'covtype.csv')
parquet_path = os.path.join(DST_DIR, 'train.parquet')
features_path = os.path.join(DST_DIR, 'features.txt')
cat_features_path = os.path.join(DST_DIR, 'categorical_features.txt')

# Определение списков колонок
numerical_columns = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

categorical_columns = [
    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
    'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
    'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
    'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
    'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
    'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
    'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'
]

# Преобразование списков колонок к нижнему регистру
numerical_columns_lower = [col.lower() for col in numerical_columns]
categorical_columns_lower = [col.lower() for col in categorical_columns]
target_column = 'Cover_Type'.lower()

# Все признаки (предикторы) в нижнем регистре
predictive_columns_lower = numerical_columns_lower + categorical_columns_lower

def main():
    # Проверка существования исходного файла
    if not os.path.exists(csv_path):
        print(f"Ошибка: Исходный файл {csv_path} не найден.")
        sys.exit(1)

    try:
        # Чтение датасета
        print("Чтение CSV файла...")
        df = pd.read_csv(csv_path)

        # Проверка наличия всех колонок (с учетом оригинального регистра)
        missing_columns = [col for col in numerical_columns + categorical_columns + ['Cover_Type'] if col not in df.columns]
        if missing_columns:
            print(f"Ошибка: В датасете отсутствуют следующие колонки: {', '.join(missing_columns)}")
            sys.exit(1)

        # Преобразование имен колонок к нижнему регистру
        print("Преобразование имен колонок к нижнему регистру...")
        df.columns = [col.lower() for col in df.columns]

        # Создание директории назначения, если она не существует
        os.makedirs(DST_DIR, exist_ok=True)

        # Сохранение в формате parquet
        print("Преобразование в формат parquet...")
        df.to_parquet(parquet_path, index=False)

        # Запись списков колонок в текстовые файлы
        print("Запись списков колонок в текстовые файлы...")
        with open(features_path, 'w') as f:
            f.write('\n'.join(predictive_columns_lower))

        with open(cat_features_path, 'w') as f:
            f.write('\n'.join(categorical_columns_lower))

        print(f"Готово! Файлы сохранены:")
        print(f"  - {parquet_path} (с колонками в нижнем регистре)")
        print(f"  - {features_path}")
        print(f"  - {cat_features_path}")

    except Exception as e:
        print(f"Ошибка при обработке данных: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()