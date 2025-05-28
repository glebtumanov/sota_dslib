#! /usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score # Добавляем accuracy_score
import optuna
import torch
import warnings
import time
import datetime
import sys
import os
import math # <--- Добавляем импорт math

# --- Утилиты Форматирования ---

def format_value_for_code(key, value, apply_rounding=True):
    """Форматирует значение параметра для вставки в Python код с новыми правилами."""
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not apply_rounding:
            return f"{value}" # Возвращаем стандартное представление float

        if value == 0.0:
            return "0.0"

        # Специальное правило для learning_rate / lr
        if key in ['learning_rate', 'lr']:
            if abs(value) >= 1.0:
                p = 1 # Округляем до 1 знака если >= 1
            else:
                p = max(1, -math.floor(math.log10(abs(value))))
            return f"{value:.{p}f}"

        # Правило для очень маленьких чисел -> экспоненциальная форма
        if abs(value) < 1e-4:
            return f"{value:.1e}"

        # Общее правило форматирования
        if abs(value) >= 1.0:
            precision = 2 # 2 десятичных знака для чисел >= 1
        else: # 1e-4 <= abs(value) < 1.0
            # 2 значащие цифры после нулей
            precision = max(2, -math.floor(math.log10(abs(value))) + 1)
        return f"{value:.{precision}f}"

    # Для других типов (если вдруг появятся)
    return str(value)

def format_value_for_yaml(key, value, apply_rounding=True):
    """Форматирует значение параметра для вставки в YAML с новыми правилами."""
    if isinstance(value, (str, bool, int)):
        return str(value) # YAML обрабатывает их без кавычек (кроме спец. случаев)

    if isinstance(value, float):
        if not apply_rounding:
            return f"{value}" # Возвращаем стандартное представление float

        if value == 0.0:
            return "0.0"

        if key in ['learning_rate', 'lr']:
            if abs(value) >= 1.0:
                 p = 1
            else:
                 p = max(1, -math.floor(math.log10(abs(value))))
            return f"{value:.{p}f}"

        if abs(value) < 1e-4:
            return f"{value:.1e}"

        # Общее правило форматирования
        if abs(value) >= 1.0:
            precision = 2 # 2 десятичных знака для чисел >= 1
        else: # 1e-4 <= abs(value) < 1.0
            # 2 значащие цифры после нулей
            precision = max(2, -math.floor(math.log10(abs(value))) + 1)
        return f"{value:.{precision}f}"

    return str(value)

# --- Глобальные переменные для колбэка ---
tuning_start_time = 0
total_trials_global = 0

# --- Функции ---

def load_data(train_path, cat_features_path, features_path, target_col, index_col):
    """Загружает тренировочные данные и опционально список категориальных признаков и выбранных фичей."""
    print(f"Загрузка данных из {train_path}...")
    try:
        train_df = pd.read_parquet(train_path)
        print(f"Данные загружены. Форма: {train_df.shape}")

        # Загрузка списка выбранных фичей
        selected_features = None
        if features_path and os.path.exists(features_path):
            print(f"Загрузка списка фичей из {features_path}...")
            with open(features_path, 'r') as f:
                all_selected_features = [line.strip() for line in f.readlines()]
            
            # Фильтруем фичи, которые действительно есть в данных
            selected_features = [f for f in all_selected_features if f in train_df.columns]
            missing_features = [f for f in all_selected_features if f not in train_df.columns]
            if missing_features:
                print(f"Warning: Следующие фичи из файла не найдены в DataFrame: {missing_features}")
            if not selected_features:
                print("Warning: Не найдено валидных фичей из файла в данных. Используются все колонки.")
                selected_features = None
            else:
                print(f"Найдено {len(selected_features)} фичей из файла.")
        else:
            print("Файл фичей не указан или не найден. Используются все колонки.")

        # Проверяем наличие обязательных колонок
        if target_col not in train_df.columns:
            print(f"Error: Целевая колонка '{target_col}' не найдена в данных.")
            sys.exit(1)
        if index_col and index_col not in train_df.columns:
            print(f"Warning: Колонка индекса '{index_col}' не найдена. Игнорируется.")
            index_col = None

        # Фильтруем данные по выбранным фичам (если указаны)
        if selected_features is not None:
            # Добавляем обязательные колонки к списку фичей
            cols_to_keep = selected_features.copy()
            if target_col not in cols_to_keep:
                cols_to_keep.append(target_col)
            if index_col and index_col not in cols_to_keep:
                cols_to_keep.append(index_col)
            
            # Фильтруем датафрейм
            original_shape = train_df.shape
            train_df = train_df[cols_to_keep]
            print(f"Данные отфильтрованы по выбранным фичам. Форма: {original_shape} -> {train_df.shape}")

        # Загрузка категориальных признаков
        categorical_features = []
        if cat_features_path and os.path.exists(cat_features_path):
            print(f"Загрузка категориальных признаков из {cat_features_path}...")
            with open(cat_features_path, 'r') as f:
                all_categorical_features = [line.strip() for line in f.readlines()]

            # Фильтруем категориальные фичи по доступным колонкам (после фильтрации по selected_features)
            categorical_features = [f for f in all_categorical_features if f in train_df.columns]
            missing_cat_features = [f for f in all_categorical_features if f not in train_df.columns]
            if missing_cat_features:
                print(f"Warning: Следующие категориальные признаки из файла не найдены в DataFrame: {missing_cat_features}")
            if not categorical_features:
                 print("Warning: Не найдено валидных категориальных признаков из файла в данных.")
            else:
                 print(f"Найдено {len(categorical_features)} категориальных признаков из файла.")
        else:
             print("Файл категориальных признаков не указан или не найден.")

        return train_df, categorical_features, target_col, index_col

    except FileNotFoundError as e:
        print(f"Error: Не найден файл данных '{train_path}' или файл признаков '{cat_features_path}'.")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Произошла ошибка при загрузке данных: {e}")
        sys.exit(1)


def split_data(df, target_col, index_col, test_size, random_state, stratify_col=None):
    """Разделяет данные на обучающую и тестовую выборки с опциональной стратификацией."""
    print("Разделение данных на обучающую и тестовую выборки...")
    cols_to_drop = [target_col]
    if index_col:
        cols_to_drop.append(index_col)

    features = df.drop(columns=cols_to_drop)
    target = df[target_col]

    stratify_data = None
    if stratify_col and stratify_col in df.columns:
        stratify_data = df[stratify_col]
        print(f"Выполняется стратифицированное разделение по колонке: {stratify_col}")
    elif stratify_col:
        print(f"Warning: Колонка для стратификации '{stratify_col}' не найдена. Выполняется обычное разделение.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=stratify_data
    )
    print(f"Размеры выборок: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def objective(trial, estimator_class, param_space_func, static_params_objective,
              X_train, y_train, X_test, y_test, eval_metric, metric_mode, cat_features, device):
    """Универсальная objective функция для Optuna."""

    # 1. Получаем гиперпараметры из пространства поиска
    hyperparams = param_space_func(trial)

    # 2. Объединяем с статичными параметрами
    model_params = {**static_params_objective, **hyperparams}

    # 3. Извлекаем параметры, которые должны идти в fit, а не в конструктор
    fit_params = {}
    if 'early_stopping_rounds' in model_params:
        fit_params['early_stopping_rounds'] = model_params.pop('early_stopping_rounds')

    # 4. Убедимся что device передан если он есть
    if 'device' in static_params_objective:
         model_params['device'] = device # Перезаписываем на актуальный device

    # 5. Создаем экземпляр модели
    try:
        model = estimator_class(**model_params)
    except TypeError as e:
         print(f"\nError: Ошибка при создании эстиматора {estimator_class.__name__} с параметрами: {model_params}")
         print(f"Original Error: {e}")
         # Можно выбросить исключение Optuna, чтобы триал не засчитался
         raise optuna.TrialPruned(f"Ошибка инициализации модели: {e}")


    # 6. Обучаем модель
    # Эстиматор должен поддерживать fit с eval_set и cat_features (если они есть)
    try:
        # Используем метрику из static_params_objective, если она задана, иначе используем eval_metric
        model_eval_metric = static_params_objective.get('eval_metric', eval_metric)
        
        # Базовые параметры fit
        fit_kwargs = {
            'eval_set': (X_test, y_test),
            'eval_metric': model_eval_metric, # Используем метрику из параметров модели
            'mode': metric_mode,
            'pbar': False # Отключаем прогресс бар внутри objective
        }
        
        # Добавляем параметры для fit, которые были извлечены из model_params
        fit_kwargs.update(fit_params)
        
        # Добавляем cat_features если они есть и модель их принимает
        import inspect
        sig = inspect.signature(model.fit)
        if 'cat_features' in sig.parameters and cat_features:
            fit_kwargs['cat_features'] = cat_features
        
        # Фильтруем параметры, которые не поддерживаются методом fit
        supported_params = set(sig.parameters.keys())
        unsupported_params = [k for k in fit_kwargs if k not in supported_params]
        if unsupported_params:
            print(f"\nWarning: Следующие параметры не поддерживаются методом fit класса {estimator_class.__name__}: {unsupported_params}")
            for param in unsupported_params:
                fit_kwargs.pop(param)

        model.fit(X_train, y_train, **fit_kwargs)

        # 6.5 Проверяем наличие predict_proba для метрик вероятности
        predict_method = model.predict
        if eval_metric in ['roc_auc', 'AUC']:
            if hasattr(model, 'predict_proba'):
                predict_method = model.predict_proba
            else:
                print(f"\nWarning: Метрика '{eval_metric}' требует метод predict_proba, но он не найден у {estimator_class.__name__}. Используется predict.")

    except Exception as e:
        print(f"\nError: Ошибка при обучении модели в триале {trial.number}: {e}")
        # Пропускаем триал
        raise optuna.TrialPruned(f"Ошибка обучения: {e}")

    # 7. Делаем предсказания
    try:
        predict_kwargs = {}
        sig_predict = inspect.signature(predict_method)
        if 'cat_features' in sig_predict.parameters and cat_features:
            predict_kwargs['cat_features'] = cat_features
        if 'pbar' in sig_predict.parameters:
             predict_kwargs['pbar'] = False

        # Используем predict_proba если нужно
        y_pred_output = predict_method(X_test, **predict_kwargs)
    except Exception as e:
        print(f"\nError: Ошибка при предсказании моделью в триале {trial.number}: {e}")
        raise optuna.TrialPruned(f"Ошибка предсказания: {e}")

    # 8. Считаем метрику
    try:
        # Преобразование имен метрик между библиотеками
        metric_mapping = {
            'AUC': 'roc_auc',  # CatBoost -> sklearn
            'auc': 'roc_auc',  # LightGBM -> sklearn
            'roc_auc': 'roc_auc'  # прямое отображение
        }
        
        # Получаем соответствующее имя метрики для sklearn
        sklearn_metric = metric_mapping.get(eval_metric, eval_metric)
        
        if sklearn_metric == 'mae':
            metric_value = mean_absolute_error(y_test, y_pred_output)
        elif sklearn_metric == 'accuracy':
            metric_value = accuracy_score(y_test, y_pred_output)
        elif sklearn_metric == 'roc_auc':
            # predict_proba обычно возвращает [prob_0, prob_1]
            if y_pred_output.ndim == 2 and y_pred_output.shape[1] == 2:
                metric_value = roc_auc_score(y_test, y_pred_output[:, 1])
            else:
                # Если вернулся одномерный массив (например, только prob_1)
                metric_value = roc_auc_score(y_test, y_pred_output)
        else:
             print(f"\nWarning: Расчет метрики '{eval_metric}' не реализован в objective. Попытка расчета MAE.")
             try:
                 metric_value = mean_absolute_error(y_test, y_pred_output)
             except Exception as mae_e:
                  print(f"\nError: Не удалось рассчитать даже MAE: {mae_e}")
                  raise optuna.TrialPruned(f"Не удалось рассчитать метрику '{eval_metric}'")

    except Exception as e:
         print(f"\nError: Ошибка при расчете метрики '{eval_metric}' в триале {trial.number}: {e}")
         raise optuna.TrialPruned(f"Ошибка расчета метрики: {e}")

    # Optuna будет минимизировать или максимизировать это значение
    return metric_value


def progress_callback(study, trial):
    """Callback для вывода прогресса Optuna."""
    global tuning_start_time, total_trials_global

    # Получаем имя метрики из study
    metric_name = "value" # Имя по умолчанию
    if study.metric_names:
        metric_name = study.metric_names[0]

    now = time.time()
    duration_trial = trial.duration.total_seconds() if trial.duration else 0
    duration_total = now - tuning_start_time
    best_value = study.best_value if study.best_value is not None else (float('inf') if study.direction == optuna.study.StudyDirection.MINIMIZE else float('-inf'))
    current_value = trial.value if trial.value is not None else float('nan')

    trial_time_str = str(datetime.timedelta(seconds=int(duration_trial)))
    total_time_str = str(datetime.timedelta(seconds=int(duration_total)))

    progress_str = (
        f"Trial {trial.number:3d}/{total_trials_global}: "
        f"Current {metric_name}={current_value:.4f}, "
        f"Best {metric_name}={best_value:.4f}, "
        f"Trial Time={trial_time_str}, "
        f"Total Time={total_time_str}"
    )
    print(progress_str)


def save_results_to_excel(study, filename):
    """Сохраняет результаты исследования Optuna в Excel файл."""
    print(f"Сохранение результатов в {filename}...")
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = study.trials_dataframe()
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"Результаты успешно сохранены.")
    except ImportError:
        print("Error: Для сохранения в Excel установите 'openpyxl': pip install openpyxl")
    except Exception as e:
        print(f"Error: Не удалось сохранить результаты в Excel: {e}")


def print_study_summary(study, static_params_objective, rounded_output=True):
    """Выводит итоговую информацию и генерирует код для Python и YAML."""
    print("\n--- Результаты подбора гиперпараметров ---")
    # Определяем имя метрики из исследования
    metric_name = "value"
    if study.metric_names:
        metric_name = study.metric_names[0]


    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Количество успешно завершенных попыток: {len(completed_trials)}")

    if study.best_trial:
        print(f"\nЛучшее значение ({metric_name}): {study.best_value:.4f}")

        # Объединяем найденные и статичные параметры
        # Убедимся, что статичные параметры не перезапишут найденные
        all_best_params = {**static_params_objective, **study.best_params}

        # --- Генерация Python кода ---
        estimator_name = study.user_attrs.get("estimator_class_name", "Estimator") # Получаем имя класса из атрибутов
        python_code_lines = [f"model = {estimator_name}("]
        for key, value in sorted(all_best_params.items()):
            formatted_value = format_value_for_code(key, value, apply_rounding=rounded_output)
            python_code_lines.append(f"    {key} = {formatted_value},")
        python_code_lines.append(")")
        python_code = "\n".join(python_code_lines)

        # --- Генерация YAML кода ---
        yaml_code_lines = ["hyperparameters:"]
        for key, value in sorted(study.best_params.items()):
            formatted_value = format_value_for_yaml(key, value, apply_rounding=rounded_output)
            yaml_code_lines.append(f"    {key}: {formatted_value}")
        yaml_code = "\n".join(yaml_code_lines)

        # --- Вывод кода ---
        print("\n--- Код для Python ---")
        print("```python")
        print(python_code)
        print("```")

        print("\n--- Код для YAML (только оптимизированные параметры) ---")
        print("```yaml")
        print(yaml_code)
        print("```")

    else:
        print("\nЛучшая попытка не найдена.")


# --- Основная функция запуска ---

def run_tuning(train_data_path, cat_features_path, features_path, target_col, index_col, stratify_col,
               test_size, split_random_state, study_name, storage_dir, n_trials, timeout_seconds,
               metric_to_optimize, direction, estimator_class, param_space_func, static_params_objective,
               optuna_log_level=optuna.logging.WARNING, rounded_output=True):
    """Основная функция для запуска подбора гиперпараметров."""
    global tuning_start_time, total_trials_global

    warnings.filterwarnings('ignore')
    # Устанавливаем verbosity для Optuna
    optuna.logging.set_verbosity(optuna_log_level)

    # --- 1. Подготовка ---
    storage_path = os.path.join(storage_dir, f"{study_name}.db")
    results_excel_path = os.path.join(storage_dir, f"{study_name}_results.xlsx")
    os.makedirs(storage_dir, exist_ok=True) # Создаем директорию хранения

    # --- 2. Загрузка данных ---
    train_df, categorical_features, target_col, index_col = load_data(
        train_data_path, cat_features_path, features_path, target_col, index_col
    )

    # --- 3. Разделение данных ---
    X_train, X_test, y_train, y_test = split_data(
        train_df, target_col, index_col, test_size, split_random_state, stratify_col
    )
    del train_df # Освобождаем память

    # --- 4. Определение устройства ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # --- 5. Настройка исследования Optuna ---
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True
    )
    # Сохраняем имя метрики и класса для использования в print_study_summary
    study.set_user_attr("estimator_class_name", estimator_class.__name__)

    # --- Проверка количества существующих триалов ---
    completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Найдено завершенных триалов в существующем исследовании: {completed_trials_count}")

    skip_optimization = completed_trials_count >= n_trials
    if skip_optimization:
        print(f"Количество завершенных триалов ({completed_trials_count}) >= запрошенному ({n_trials}). Оптимизация будет пропущена.")

    # --- 6. Запуск оптимизации ---
    tuning_start_time = time.time()
    total_trials_global = n_trials # Для отображения в колбэке
    study_completed = False # Флаг успешного завершения optimize

    # Оборачиваем objective для передачи доп. аргументов
    obj_func = lambda trial: objective(
        trial,
        estimator_class=estimator_class,
        param_space_func=param_space_func,
        static_params_objective=static_params_objective,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        eval_metric=metric_to_optimize,
        metric_mode=direction.replace("imize",""), # 'min' или 'max'
        cat_features=categorical_features,
        device=device
    )

    optimization_executed = False
    try:
        if not skip_optimization:
            remaining_trials = n_trials - completed_trials_count
            print(f"\nЗапуск Optuna исследования '{study_name}'. Требуется {remaining_trials} новых попыток (всего {n_trials}, timeout: {timeout_seconds}s)...\n")
            study.optimize(
                obj_func,
                n_trials=remaining_trials, # Запускаем только недостающие триалы
                timeout=timeout_seconds,
                callbacks=[progress_callback],
                show_progress_bar=False
            )
            optimization_executed = True
            study_completed = True # Считаем успешным, если optimize не упал
        else:
            # Если оптимизация пропущена, считаем что все 'успешно' для блока finally
            study_completed = True

    except KeyboardInterrupt:
        print("\nОптимизация прервана пользователем (Ctrl+C).")
    except Exception as e:
        print(f"\nError: Произошла ошибка во время оптимизации: {e}")
        import traceback
        traceback.print_exc() # Печатаем стек ошибки
    finally:
        print()
        total_duration = time.time() - tuning_start_time
        if optimization_executed:
            print(f"Оптимизация завершена. Затраченное время: {str(datetime.timedelta(seconds=int(total_duration)))}")
        else:
            print("Оптимизация была пропущена.")

        # --- 7. Сохранение результатов и вывод статистики (всегда) ---
        save_results_to_excel(study, results_excel_path)
        print_study_summary(study, static_params_objective, rounded_output=rounded_output)

        # --- 8. Обучение финальной модели ---
        # Финальное обучение удалено

    print("\nСкрипт подбора гиперпараметров завершил работу.")