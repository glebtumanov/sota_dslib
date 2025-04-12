#!/usr/bin/env python3
import pandas as pd
from torch.utils.data import DataLoader
from tab_net import TabNet, TabNetDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset

# Константы
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024  # Уменьшаем размер батча
EPOCHS = 50  # Увеличиваем количество эпох
LEARNING_RATE = 1e-3  # Уменьшаем скорость обучения
EARLY_STOPPING_PATIENCE = 5  # Количество эпох без улучшения до остановки

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
        output = model(x)
        loss = criterion(output, y)
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

            output = model(x)
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
        cat_emb_dim=8,  # Используем эмбеддинги размерности 4
        n_steps=5,  # Увеличиваем количество шагов
        hidden_dim=128,  # Увеличиваем размерность скрытого слоя
        decision_dim=64,  # Увеличиваем размерность решающего слоя
        n_glu_layers=3,  # Увеличиваем количество GLU слоев
        # norm_type='batch',
        dropout=0.6,  # Увеличиваем вероятность дропаута
        output_dim=1
    ).to(DEVICE)

    # Используем оптимизатор Adam с весовым decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Добавляем scheduler для уменьшения learning rate со временем
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=True
    )

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
    external_test_dataset = TabNetDataset(test_data, features, cat_features, target_col=None, index_col='sk_id_curr')
    external_test_loader = DataLoader(external_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Получение предсказаний модели на тестовом наборе
    def get_predictions(model, loader):
        model.eval()
        all_preds = []

        with torch.no_grad():
            for x in tqdm(loader, desc="Predicting"):
                x = x.to(DEVICE)
                outputs = model(x)
                all_preds.append(outputs.cpu().numpy())

        return np.concatenate(all_preds)

    # Получаем предсказания для внешнего тестового набора
    print("\nДелаем предсказания на внешнем тестовом наборе...")
    test_preds = get_predictions(model, external_test_loader)

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
