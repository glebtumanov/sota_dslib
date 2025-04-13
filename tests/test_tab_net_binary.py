#!/usr/bin/env python3
import pandas as pd
from torch.utils.data import DataLoader
from tab_net import TabNet
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset

# Константы для обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024  # Размер батча, {256, 512, 1024, 2048, 4096, 8192}
EPOCHS = 50  # Количество эпох, {20, 30, 40, 50, 100}
LEARNING_RATE = 0.0025  # Скорость обучения, {0.001, 0.0025, 0.005, 0.01}
EARLY_STOPPING_PATIENCE = 5  # Количество эпох без улучшения до остановки

# Гиперпараметры модели TabNet
CAT_EMB_DIM = 6  # Размерность эмбеддингов для категориальных признаков [5, 10]
N_STEPS = 4  # Количество шагов в TabNet, в диапазоне [3, 10]
HIDDEN_DIM = 16  # Размерность скрытого слоя, выбираются из {8, 16, 24, 32, 64, 128}
DECISION_DIM = 8  # Размерность решающего слоя, обычно меньше hidden_dim
N_GLU_LAYERS = 2  # Количество GLU слоев, обычно 2 слоя
DROPOUT = 0.6  # Вероятность дропаута, выбирается из {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
GAMMA = 1.5  # Коэффициент затухания для масок внимания, выбирается из {1.0, 1.2, 1.5, 2.0}
LAMBDA_SPARSE = 0.0001  # Коэффициент регуляризации разреженности, из {0, 0.000001, 0.0001, 0.001, 0.01, 0.1}
VIRTUAL_BATCH_SIZE = 128  # Размер виртуального батча для Ghost BatchNorm, {256, 512, 1024, 2048, 4096}
MOMENTUM = 0.9  # Параметр momentum для BatchNorm, {0.6, 0.7, 0.8, 0.9, 0.95, 0.98}
OUTPUT_DIM = 1  # Размерность выходного слоя

target_col = 'target'
index_col = 'sk_id_curr'


class TabNetDataset(Dataset):
    """Подготовка данных для TabNet из pandas DataFrame."""
    def __init__(self, df, features, cat_features, target_col=None, index_col=None):
        """
        :param df: DataFrame с исходными данными
        :param features: Список признаков (категориальные + числовые)
        :param cat_features: Список категориальных признаков
        :param target_col: Целевая переменная (если есть)
        :param index_col: Индексная колонка (если есть)
        """
        self.features = features
        self.cat_features = cat_features
        self.target_col = target_col

        # Создаем копию данных для безопасности
        df_copy = df.copy()

        if index_col is not None and index_col in df_copy.columns:
            df_copy = df_copy.drop(columns=[index_col])

        if target_col and target_col in df_copy.columns:
            self.targets = torch.tensor(df_copy[target_col].values, dtype=torch.float32).unsqueeze(1)
            df_copy = df_copy.drop(columns=[target_col])
        else:
            self.targets = None

        # Сохраняем позиции категориальных признаков
        self.cat_idxs = [features.index(f) for f in cat_features if f in features]

        # Нормализуем числовые признаки
        num_features = [f for f in features if f not in cat_features]
        if num_features:
            df_copy[num_features] = (df_copy[num_features] - df_copy[num_features].mean()) / df_copy[num_features].std()
            # Заменяем nan и inf значения на 0
            df_copy[num_features] = df_copy[num_features].fillna(0).replace([float('inf'), -float('inf')], 0)

        # Конвертируем данные в тензоры
        self.data = torch.tensor(df_copy[features].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.data[idx], self.targets[idx]
        else:
            return self.data[idx]


# Функция тренировки модели
def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # Получаем выходы и маски для регуляризации
        output, masks = model(x, return_masks=True)

        # Вычисляем основную функцию потерь
        main_loss = criterion(output, y)

        # Вычисляем регуляризацию разреженности
        sparse_loss = model.calculate_sparse_loss(masks)

        # Суммарная функция потерь
        loss = main_loss + sparse_loss

        loss.backward()

        # Добавляем градиентное клиппирование для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds.append(output.detach().cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    auc = roc_auc_score(targets, preds)
    return total_loss / len(loader), auc


# Функция валидации модели
def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Получаем только выходы без регуляризации разреженности при валидации
            output, _ = model(x, return_masks=True)
            loss = criterion(output, y)

            total_loss += loss.item()
            preds.append(output.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    auc = roc_auc_score(targets, preds)
    return total_loss / len(loader), auc

if __name__ == "__main__":
    with open('/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/features.txt', 'r') as f:
        features = f.read().splitlines()

    with open('/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/categorical_features.txt', 'r') as f:
        cat_features = f.read().splitlines()

    print("Загружаю данные...")
    train = pd.read_parquet('/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/train.parquet')
    # Проводим стратифицированное разделение, выделяя 10% данных для валидации
    train_data, val_data = train_test_split(train, test_size=0.2, random_state=42, stratify=train[target_col])
    print(f"Количество признаков: {len(features)}, из них категориальных: {len(cat_features)}")
    print(f"Размер обучающих данных: {train_data.shape}, валидационных данных: {val_data.shape}")

    print("Подготавливаю данные...")
    train_dataset = TabNetDataset(train_data, features, cat_features, target_col='target', index_col='sk_id_curr')
    val_dataset = TabNetDataset(val_data, features, cat_features, target_col='target', index_col='sk_id_curr')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Инициализация модели, оптимизатора и функции потерь
    print("Инициализирую модель...")
    model = TabNet(
        input_dim=len(features),
        cat_idxs=train_dataset.cat_idxs,
        cat_dims=[int(train_data[col].nunique()) for col in cat_features],
        cat_emb_dim=CAT_EMB_DIM,
        n_steps=N_STEPS,
        hidden_dim=HIDDEN_DIM,
        decision_dim=DECISION_DIM,
        n_glu_layers=N_GLU_LAYERS,
        dropout=DROPOUT,
        gamma=GAMMA,
        lambda_sparse=LAMBDA_SPARSE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        momentum=MOMENTUM,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)

    # Используем оптимизатор Adam с весовым decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Добавляем scheduler для уменьшения learning rate со временем
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Раннее остановка
    best_val_auc = 0
    no_improvement_epochs = 0
    best_model_state = None

    # Цикл обучения
    print(f"Начинаю обучение на {DEVICE}...")
    for epoch in range(EPOCHS):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = validate_epoch(model, val_loader, criterion)

        # Обновляем scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Validation loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

        # Проверяем улучшение метрик
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improvement_epochs = 0
            # Сохраняем лучшую модель
            best_model_state = model.state_dict().copy()
            print(f"Сохраняю лучшую модель с AUC: {best_val_auc:.4f}")
        else:
            no_improvement_epochs += 1
            print(f"Нет улучшения в течение {no_improvement_epochs} эпох")

            if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
                print("Останавливаю обучение из-за отсутствия улучшений")
                break

    # Загружаем лучшую модель для финальной оценки
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Загружена лучшая модель")

    # Финальная оценка на валидационном наборе
    val_loss, val_auc = validate_epoch(model, val_loader, criterion)
    print(f"\nФинальная оценка на validation наборе:")
    print(f"Validation loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

    # Загрузка тестовых данных
    print("\nЗагружаем тестовые данные...")
    test_data = pd.read_parquet('/www/dslib/spark_sota_modeling/dataset/home-credit-default-risk/test.parquet')
    test_dataset = TabNetDataset(test_data, features, cat_features, target_col=None, index_col='sk_id_curr')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Получение предсказаний модели на тестовом наборе
    def get_predictions(model, loader):
        model.eval()
        all_preds = []

        with torch.no_grad():
            for x in tqdm(loader, desc="Predicting"):
                x = x.to(DEVICE)
                outputs, _ = model(x, return_masks=True)
                all_preds.append(outputs.cpu().numpy())

        return np.concatenate(all_preds)

    # Получаем предсказания для внешнего тестового набора
    print("\nДелаем предсказания на внешнем тестовом наборе...")
    test_preds = get_predictions(model, test_loader)

    # Преобразуем логиты в вероятности с помощью сигмоиды
    test_probs = 1 / (1 + np.exp(-test_preds))

    # Создаем таблицу с предсказаниями
    submission = pd.DataFrame({
        'sk_id_curr': test_data[index_col].values,
        'prediction': test_probs.flatten()
    })

    print(f"\nРезультаты на внешнем тестовом наборе:")
    print(f"Количество предсказаний: {len(submission)}")
    submission.to_csv('submission.csv', index=False)
    print("submission.csv saved")
