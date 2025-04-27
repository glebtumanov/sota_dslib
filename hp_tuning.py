#! /usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score # Добавляем roc_auc_score
import optuna
import torch
import warnings
import time
import datetime
import sys
import os
import math # <--- Добавляем импорт math
from typing import Callable, Dict, Any, List, Tuple, Optional

# --- Утилиты Форматирования ---

def format_value_for_code(key: str, value: Any) -> str:
    """Форматирует значение параметра для вставки в Python код с новыми правилами."""
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
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

def format_value_for_yaml(key: str, value: Any) -> str:
    """Форматирует значение параметра для вставки в YAML с новыми правилами."""
    if isinstance(value, (str, bool, int)):
        return str(value) # YAML обрабатывает их без кавычек (кроме спец. случаев)

    if isinstance(value, float):
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

def load_data(train_path: str, cat_features_path: Optional[str], target_col: str, index_col: Optional[str]) -> Tuple[pd.DataFrame, List[str], str, Optional[str]]:
    """Загружает тренировочные данные и опционально список категориальных признаков."""
    print(f"Загрузка данных из {train_path}...")
    try:
        train_df = pd.read_parquet(train_path)
        print(f"Данные загружены. Форма: {train_df.shape}")

        categorical_features = []
        if cat_features_path and os.path.exists(cat_features_path):
            print(f"Загрузка категориальных признаков из {cat_features_path}...")
            with open(cat_features_path, 'r') as f:
                all_categorical_features = [line.strip() for line in f.readlines()]

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


        if target_col not in train_df.columns:
            print(f"Error: Целевая колонка '{target_col}' не найдена в данных.")
            sys.exit(1)
        if index_col and index_col not in train_df.columns:
            print(f"Warning: Колонка индекса '{index_col}' не найдена. Игнорируется.")
            index_col = None

        return train_df, categorical_features, target_col, index_col

    except FileNotFoundError as e:
        print(f"Error: Не найден файл данных '{train_path}' или файл признаков '{cat_features_path}'.")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Произошла ошибка при загрузке данных: {e}")
        sys.exit(1)


def split_data(df: pd.DataFrame, target_col: str, index_col: Optional[str], test_size: float, random_state: int, stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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


def objective(trial: optuna.Trial,
              estimator_class: Callable,
              param_space_func: Callable[[optuna.Trial], Dict[str, Any]],
              static_params_objective: Dict[str, Any],
              X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              eval_metric: str, # Метрика для оптимизации
              metric_mode: str, # 'min' или 'max'
              cat_features: List[str],
              device: torch.device) -> float:
    """Универсальная objective функция для Optuna."""

    # 1. Получаем гиперпараметры из пространства поиска
    hyperparams = param_space_func(trial)

    # 2. Объединяем с статичными параметрами
    model_params = {**static_params_objective, **hyperparams}

    # Убедимся что device передан если он есть в static_params_objective
    if 'device' in static_params_objective:
         model_params['device'] = device # Перезаписываем на актуальный device

    # 3. Создаем экземпляр модели
    try:
        model = estimator_class(**model_params)
    except TypeError as e:
         print(f"\nError: Ошибка при создании эстиматора {estimator_class.__name__} с параметрами: {model_params}")
         print(f"Original Error: {e}")
         # Можно выбросить исключение Optuna, чтобы триал не засчитался
         raise optuna.TrialPruned(f"Ошибка инициализации модели: {e}")


    # 4. Обучаем модель
    # Эстиматор должен поддерживать fit с eval_set и cat_features (если они есть)
    try:
        fit_kwargs = {
            'eval_set': (X_test, y_test),
            'eval_metric': eval_metric,
            'mode': metric_mode,
            'pbar': False # Отключаем прогресс бар внутри objective
        }
        # Добавляем cat_features если они есть и модель их принимает
        import inspect
        sig = inspect.signature(model.fit)
        if 'cat_features' in sig.parameters and cat_features:
            fit_kwargs['cat_features'] = cat_features

        model.fit(X_train, y_train, **fit_kwargs)

        # 4.5 Проверяем наличие predict_proba для метрик вероятности
        predict_method = model.predict
        if eval_metric in ['roc_auc']:
            if hasattr(model, 'predict_proba'):
                predict_method = model.predict_proba
            else:
                print(f"\nWarning: Метрика '{eval_metric}' требует метод predict_proba, но он не найден у {estimator_class.__name__}. Используется predict.")

    except Exception as e:
        print(f"\nError: Ошибка при обучении модели в триале {trial.number}: {e}")
        # Пропускаем триал
        raise optuna.TrialPruned(f"Ошибка обучения: {e}")

    # 5. Делаем предсказания
    try:
        predict_kwargs = {}
        sig_predict = inspect.signature(model.predict)
        if 'cat_features' in sig_predict.parameters and cat_features:
            predict_kwargs['cat_features'] = cat_features
        if 'pbar' in sig_predict.parameters:
             predict_kwargs['pbar'] = False

        # Используем predict_proba если нужно
        y_pred_output = predict_method(X_test, **predict_kwargs)
    except Exception as e:
        print(f"\nError: Ошибка при предсказании моделью в триале {trial.number}: {e}")
        raise optuna.TrialPruned(f"Ошибка предсказания: {e}")


    # 6. Считаем метрику
    # !! Важно: используем метрику, которую Optuna должна оптимизировать !!
    # Здесь пока простой пример для MAE, нужно обобщить или передавать функцию расчета
    try:
        # TODO: Сделать расчет метрики более гибким (передавать функцию?)
        if eval_metric == 'mae':
            metric_value = mean_absolute_error(y_test, y_pred_output)
        elif eval_metric == 'roc_auc':
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


def progress_callback(study: optuna.Study, trial: optuna.Trial):
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


def save_results_to_excel(study: optuna.Study, filename: str):
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


def print_study_summary(study: optuna.Study, static_params_objective: Dict[str, Any]):
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
            formatted_value = format_value_for_code(key, value)
            python_code_lines.append(f"    {key} = {formatted_value},")
        python_code_lines.append(")")
        python_code = "\n".join(python_code_lines)

        # --- Генерация YAML кода ---
        yaml_code_lines = ["hyperparameters:"]
        for key, value in sorted(study.best_params.items()):
            formatted_value = format_value_for_yaml(key, value)
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


def train_final_model(estimator_class: Callable,
                        final_model_params: Dict[str, Any], # Принимаем готовый словарь параметров
                        X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        cat_features: List[str],
                        eval_metric: str,
                        metric_mode: str,
                        device: torch.device):
    """Обучает и оценивает финальную модель с лучшими параметрами."""
    print("\n--- Обучение лучшей модели с лучшими параметрами ---")

    # Объединяем лучшие найденные параметры с статичными параметрами для финального обучения
    # Параметры уже подготовлены и переданы в final_model_params
    # Обновляем device, если он был в статичных параметрах
    if 'device' in final_model_params: # Проверяем наличие ключа
         final_model_params['device'] = device

    final_model = estimator_class(**final_model_params)

    print("Обучение модели...")
    fit_kwargs = {
        'eval_set': (X_test, y_test),
        'eval_metric': eval_metric,
        'mode': metric_mode,
        'pbar': True # Включаем прогресс бар для финального обучения
    }
    import inspect
    sig_fit = inspect.signature(final_model.fit)
    if 'cat_features' in sig_fit.parameters and cat_features:
        fit_kwargs['cat_features'] = cat_features

    final_model.fit(X_train, y_train, **fit_kwargs)

    print("Оценка финальной модели...")
    predict_kwargs = {}
    sig_predict = inspect.signature(final_model.predict)
    if 'cat_features' in sig_predict.parameters and cat_features:
            predict_kwargs['cat_features'] = cat_features
    if 'pbar' in sig_predict.parameters:
             predict_kwargs['pbar'] = True # Включаем прогресс бар

    # --- Выбираем метод predict или predict_proba ---
    final_predict_method = final_model.predict
    if eval_metric in ['roc_auc']:
        if hasattr(final_model, 'predict_proba'):
             final_predict_method = final_model.predict_proba
        else:
             print(f"\nWarning: Финальная оценка метрики '{eval_metric}' требует predict_proba, но метод не найден. Используется predict.")

    y_pred_final_output = final_predict_method(X_test, **predict_kwargs)

    # TODO: Сделать расчет метрики более гибким
    if eval_metric == 'mae':
         final_metric = mean_absolute_error(y_test, y_pred_final_output)
    elif eval_metric == 'roc_auc':
        if y_pred_final_output.ndim == 2 and y_pred_final_output.shape[1] == 2:
            final_metric = roc_auc_score(y_test, y_pred_final_output[:, 1])
        else:
            final_metric = roc_auc_score(y_test, y_pred_final_output)
    else:
         print(f"Warning: Расчет финальной метрики '{eval_metric}' не реализован. Используем MAE.")
         try:
             final_metric = mean_absolute_error(y_test, y_pred_final_output)
         except Exception as mae_e:
             print(f"\nError: Не удалось рассчитать финальную MAE: {mae_e}")
             final_metric = float('nan') # Возвращаем NaN если ничего не получилось

    print(f"\nФинальное значение {eval_metric} лучшей модели: {final_metric:.4f}")
    return final_model, final_metric


# --- Основная функция запуска ---

def run_tuning(
    # Конфигурация данных
    train_data_path: str,
    cat_features_path: Optional[str],
    target_col: str,
    index_col: Optional[str],
    stratify_col: Optional[str], # <--- Добавляем параметр для стратификации
    # Конфигурация разделения
    test_size: float,
    split_random_state: int,
    # Конфигурация Optuna
    study_name: str,
    storage_dir: str, # Директория для SQLite и Excel
    n_trials: int,
    timeout_seconds: Optional[int],
    metric_to_optimize: str, # Имя метрики для оптимизации
    direction: str, # 'minimize' или 'maximize'
    # Конфигурация модели и поиска
    estimator_class: Callable,
    param_space_func: Callable[[optuna.Trial], Dict[str, Any]],
    static_params_objective: Dict[str, Any], # Параметры для objective
    # Конфигурация финального обучения
    train_final_flag: bool,
    # Параметры для переопределения при финальном обучении
    final_epochs: Optional[int] = None,
    final_early_stopping: Optional[int] = None,
    final_verbose: bool = True,
    # Опциональные параметры с дефолтами идут последними
    optuna_log_level: int = optuna.logging.WARNING # Уровень логирования Optuna
):
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
        train_data_path, cat_features_path, target_col, index_col
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


    # --- 6. Запуск оптимизации ---
    print(f"\nЗапуск Optuna исследования '{study_name}' на {n_trials} попыток (timeout: {timeout_seconds}s)...\n")
    tuning_start_time = time.time()
    total_trials_global = n_trials # Для отображения в колбэке
    study_completed = False

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

    try:
        study.optimize(
            obj_func,
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=[progress_callback],
            show_progress_bar=False
        )
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
        print(f"Оптимизация завершена. Общее время: {str(datetime.timedelta(seconds=int(total_duration)))}")

        # --- 7. Сохранение результатов и вывод статистики (всегда) ---
        save_results_to_excel(study, results_excel_path)
        print_study_summary(study, static_params_objective)

        # --- 8. Обучение финальной модели ---
        if train_final_flag and study_completed and study.best_trial:
            # Готовим параметры для финальной модели
            final_model_params = {**static_params_objective, **study.best_params}
            if final_epochs is not None:
                final_model_params['epochs'] = final_epochs
            if final_early_stopping is not None:
                final_model_params['early_stopping_patience'] = final_early_stopping
            # verbose всегда берем из final_verbose для финального обучения
            final_model_params['verbose'] = final_verbose

            train_final_model(
                estimator_class=estimator_class,
                final_model_params=final_model_params,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                cat_features=categorical_features,
                eval_metric=metric_to_optimize,
                metric_mode=direction.replace("imize",""),
                device=device
            )
        elif not train_final_flag:
            print("\nФинальное обучение пропущено (флаг train_final_flag установлен в False).")
        elif study.best_trial: # Оптимизация прервана, но есть лучший триал
             print("Warning: Оптимизация была прервана/завершена с ошибками, но лучшие параметры найдены. Пропускаем финальное обучение.")
        else: # Нет лучшего триала
            print("Warning: Не найдено лучших параметров. Пропускаем финальное обучение.")

    print("\nСкрипт подбора гиперпараметров завершил работу.")