from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Sequence, List, Optional, Union


# --------------------------------------------------------------------------
# 1.  GLU-блок
# --------------------------------------------------------------------------
class GLUBlock(nn.Module):
    """Linear → (опц.) Norm → GLU → Dropout."""
    def __init__(self,
                 in_features: int,
                 out_features: int | None = None,
                 norm: str | None = None,          # <= по умолчанию нет нормализации
                 p_dropout: float = 0.0,
                 dim: int = -1):
        super().__init__()
        out_features = out_features or in_features
        if norm == "batch":
            # Используем BatchNorm1d с настройками по умолчанию
            self.norm = nn.BatchNorm1d(in_features)
        elif norm == "layer":
            self.norm = nn.LayerNorm(in_features)
        else:                                       # None
            self.norm = nn.Identity()

        self.fc   = nn.Linear(in_features, out_features * 2)
        self.dim  = dim
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Обработка BatchNorm1d для 3D тензоров (B, SeqLen, Features)
        if isinstance(self.norm, nn.BatchNorm1d) and x.ndim == 3:
             # BatchNorm1d ожидает (B, C, L), поэтому транспонируем
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            # LayerNorm и Identity работают с (B, ..., Features)
            x = self.norm(x)
        return self.drop(F.glu(self.fc(x), dim=self.dim))


# --------------------------------------------------------------------------
# 2.  Feature- / AttentiveTransformer
# --------------------------------------------------------------------------
class FeatureTransformer(nn.Module):
    """Последовательность GLU блоков с residual connections."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int | None = None,
                 n_glu: int = 2,
                 shared: Optional[Sequence[nn.Module]] = None,
                 norm: str | None = None,
                 dropout: float = 0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.blocks = nn.ModuleList()
        shared = list(shared) if shared else []
        current_dim = input_dim # Размерность входа для первого блока

        # Добавляем общие блоки, обновляя current_dim
        for block in shared:
            if not isinstance(block, GLUBlock):
                 raise TypeError("Shared block must be an instance of GLUBlock")
            # Проверяем совместимость размерностей (грубо)
            # TODO: Более строгая проверка?
            # if block.fc.in_features != current_dim:
            #     raise ValueError(f"Shared block input dim {block.fc.in_features} != current dim {current_dim}")
            self.blocks.append(block)
            # Выход GLU блока - половина выхода fc слоя
            current_dim = block.fc.out_features // 2

        # Добавляем независимые блоки
        for i in range(n_glu):
            # Входная размерность первого независимого блока - выход последнего общего (или input_dim)
            # Выходная размерность - output_dim
            # Входная размерность последующих независимых блоков - output_dim
            block_input_dim = current_dim if i == 0 else output_dim
            self.blocks.append(
                GLUBlock(block_input_dim,
                         output_dim,
                         norm=norm,
                         p_dropout=dropout)
            )
            current_dim = output_dim # Обновляем размерность для следующего независимого блока

        self.scale = math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_x = x
        for i, blk in enumerate(self.blocks):
            block_input = current_x # Вход для текущего блока
            x_processed = blk(block_input)

            # Применяем residual connection: добавляем вход блока к выходу,
            # если их размерности совпадают.
            if x_processed.shape == block_input.shape:
                 current_x = (x_processed + block_input) * self.scale
            else:
                 # Если размерность изменилась (например, в первом общем блоке),
                 # остаточную связь не применяем.
                 current_x = x_processed

        return current_x # Возвращаем выход последнего блока


class AttentiveTransformer(nn.Module):
    """Вычисляет маску внимания для признаков."""
    # Принимает исходную размерность input_dim и размерность attention-фич n_a
    def __init__(self, input_dim: int, attention_dim: int, momentum: float = 0.1):
        super().__init__()
        # Линейный слой отображает attention features (n_a) -> input features (input_dim)
        self.fc = nn.Linear(attention_dim, input_dim)
        # BN применяется *после* линейного слоя к размерности input_dim
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, prior: torch.Tensor, x_att: torch.Tensor) -> torch.Tensor:
        # x_att имеет размерность (B, attention_dim = n_a)
        # prior имеет размерность (B, input_dim)

        # Отображаем attention features в пространство input_dim
        a = self.fc(x_att) # Размерность: (B, input_dim)

        # Применяем BN (обрабатываем 2D вход для BN1d)
        if a.ndim == 2:
            # BN ожидает (N, C) или (N, C, L), у нас (B, input_dim) -> OK
            a = self.bn(a) # Размерность: (B, input_dim)
        elif a.ndim == 3: # Не должно происходить, если x_att это (B, n_a)
             # Если все же 3D, BN ожидает (N, C, L), транспонируем
             a = self.bn(a.transpose(1, 2)).transpose(1, 2)
        else:
             raise ValueError(f"AttentiveTransformer ожидает 2D или 3D тензор для BN после fc, получено: {a.ndim}D")

        # Применяем prior scaling
        # Убедимся, что prior имеет совместимую размерность (обычно (B, input_dim))
        if prior.shape != a.shape:
             # Попробуем исправить, если последняя размерность prior = 1
              if prior.ndim == a.ndim and prior.size(-1) == 1:
                  prior = prior.expand_as(a)
              else:
                  raise ValueError(f"Размерности prior {prior.shape} и a {a.shape} не совпадают для умножения в AttentiveTransformer")

        a = a * prior # Размеры: (B, input_dim) * (B, input_dim) -> OK

        # Возвращаем маску softmax
        return torch.softmax(a, dim=-1) # Размерность: (B, input_dim)


# --------------------------------------------------------------------------
# 3.  Эмбеддинги для числовых и категориальных фич
# --------------------------------------------------------------------------
class NumericEmbedding(nn.Module):
    """Простое взвешивание числовых признаков."""
    def __init__(self, num_numeric: int, d_model: int):
        super().__init__()
        # Параметры для взвешивания каждого числового признака
        self.weight = nn.Parameter(torch.randn(num_numeric, d_model))
        self.bias   = nn.Parameter(torch.zeros(num_numeric, d_model))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (B, F_num) -> (B, F_num, 1) * (F_num, d) + (F_num, d) -> (B, F_num, d)
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CatEmbedding(nn.Module):
    """Эмбеддинги для категориальных признаков."""
    def __init__(self, cat_dims: Sequence[int], d_model: Union[int, Sequence[int]]):
        super().__init__()
        # Если d_model - целое число, используем его для всех категорий
        if isinstance(d_model, int):
            d_model = [d_model] * len(cat_dims)
        # Проверка совпадения длин списков размерностей словарей и эмбеддингов
        elif len(d_model) != len(cat_dims):
             raise ValueError(f"Количество размерностей эмбеддингов ({len(d_model)}) "
                              f"должно совпадать с количеством категориальных признаков ({len(cat_dims)})")

        # Создаем список слоев Embedding
        self.embs = nn.ModuleList(
            nn.Embedding(v, d) for v, d in zip(cat_dims, d_model)
        )
        # Суммарная размерность всех категориальных эмбеддингов
        self.out_dim = sum(d_model)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, F_cat)
        vecs = []
        for i, emb in enumerate(self.embs):
            # Проверяем и ограничиваем индексы перед передачей в Embedding
            safe_indices = torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)
            vecs.append(emb(safe_indices)) # (B, d_i)
        # Результат: (B, F_cat, d) - объединяем по новой размерности
        return torch.stack(vecs, dim=1)


# --------------------------------------------------------------------------
# 4.  TabNetCore – «сердце» модели
# --------------------------------------------------------------------------
class TabNetCore(nn.Module):
    """Основная архитектура TabNet с последовательными шагами внимания."""
    def __init__(self,
                 input_dim: int,                  # Размерность входа после эмбеддингов и flatten
                 output_dim: int,                 # Размерность выхода модели
                 n_steps: int = 3,                # Количество шагов принятия решений
                 decision_dim: int = 64,          # Общая размерность выхода FeatureTransformer (Nd + Na)
                 n_shared: int = 2,               # Количество общих GLU блоков в FeatureTransformer
                 n_independent: int = 2,          # Количество независимых GLU блоков в FeatureTransformer на каждом шаге
                 glu_dropout: float = 0.0,        # Dropout в GLU блоках
                 norm: str | None = None,         # Тип нормализации в GLU блоках ('batch', 'layer', None)
                 gamma: float = 1.5,              # Коэффициент релаксации для prior (из статьи)
                 att_momentum: float = 0.1):      # Momentum для BN в AttentiveTransformer
        super().__init__()
        if decision_dim % 2 != 0:
            raise ValueError("decision_dim должен быть четным для разделения на d и a.")

        self.n_d = decision_dim // 2
        self.n_a = decision_dim // 2
        self.gamma = gamma
        self.n_steps = n_steps

        # Создаем общие блоки, если они есть
        shared_blocks = []
        if n_shared > 0:
             # Первый общий блок преобразует input_dim -> decision_dim
             # Последующие общие блоки: decision_dim -> decision_dim
             current_shared_dim = input_dim
             for i in range(n_shared):
                 block = GLUBlock(current_shared_dim, decision_dim, norm=norm, p_dropout=glu_dropout)
                 shared_blocks.append(block)
                 current_shared_dim = decision_dim # Выход GLU - decision_dim

        self.shared_blocks = nn.ModuleList(shared_blocks)

        # Начальное преобразование: либо общие блоки, либо линейный слой
        if self.shared_blocks:
             # Используем общие блоки для начального преобразования
             # Размерность входа для шагов будет decision_dim
             step_input_dim = decision_dim
        else:
            # Если нет общих блоков, используем Linear + BN + ReLU
            self.initial_projection = nn.Sequential(
                nn.Linear(input_dim, decision_dim),
                nn.BatchNorm1d(decision_dim),
                nn.ReLU()
            )
            step_input_dim = decision_dim

        # Feature Transformers для каждого шага (только независимые блоки)
        self.step_ft = nn.ModuleList()
        for _ in range(n_steps):
            # Вход для FeatureTransformer шага - выход предыдущего шага (step_input_dim)
            # Выход - decision_dim
            self.step_ft.append(
                FeatureTransformer(step_input_dim, decision_dim,
                                   n_glu=n_independent,
                                   shared=None, # Только независимые блоки
                                   norm=norm,
                                   dropout=glu_dropout)
            )
            # Вход для следующего шага остается decision_dim
            step_input_dim = decision_dim

        # Attentive Transformers (отображают n_a -> input_dim)
        self.att = nn.ModuleList()
        for _ in range(n_steps):
             # Передаем исходную размерность input_dim и n_a
            self.att.append(AttentiveTransformer(input_dim, self.n_a, momentum=att_momentum))

        # Финальный линейный слой для агрегированных decision outputs (n_d)
        self.fc_out = nn.Linear(self.n_d, output_dim)
        # Входная нормализация (BatchNorm) перед первым шагом
        self.input_bn = nn.BatchNorm1d(input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает:
            - outputs: финальный выход модели (логиты)
            - agg_mask_loss: агрегированная потеря разреженности масок (для регуляризации)
        """
        # x: (B, input_dim = F_all * d_model)
        prior = torch.ones_like(x) # Начальный prior scale (B, input_dim)
        agg_decision_output = 0.0  # Агрегированный выход decision компонентов
        total_entropy = 0.0        # Суммарная энтропия масок для регуляризации

        # Нормализуем вход
        x_bn = self.input_bn(x) # (B, input_dim)

        # Применяем начальное преобразование (общие блоки или линейное)
        if self.shared_blocks:
            current_features = x_bn
            for block in self.shared_blocks:
                current_features = block(current_features)
            # current_features теперь (B, decision_dim)
        else:
            current_features = self.initial_projection(x_bn) # (B, decision_dim)

        masks = [] # Сохраняем маски для возможного анализа

        for i, (ft, att) in enumerate(zip(self.step_ft, self.att)):
            # 1. Применяем FeatureTransformer шага
            # Вход: current_features (B, decision_dim), Выход: y (B, decision_dim)
            y = ft(current_features)

            # 2. Разделяем выход на decision (d) и attention (a) части
            d, a = torch.split(y, [self.n_d, self.n_a], dim=-1) # d: (B, n_d), a: (B, n_a)

            # 3. Агрегируем decision output (применяем ReLU как в статье)
            agg_decision_output = agg_decision_output + F.relu(d)

            # 4. Вычисляем маску внимания (если не последний шаг)
            # Используем AttentiveTransformer, который отображает 'a' -> 'mask' размерности input_dim
            if i < self.n_steps: # Маска не нужна после последнего шага
                mask = att(prior, a) # mask: (B, input_dim) - Корректный размер!
                masks.append(mask)

                # 5. Обновляем prior scale для следующего шага
                # prior_{i+1} = prior_i * (gamma - mask_i)
                prior = prior * (self.gamma - mask) # Размеры (B, input_dim) совпадают

                # 6. Обновляем признаки для следующего шага
                # Используем выход 'd' (decision part) для следующего шага? Или 'y'?
                # В статье (рис 3а) показано, что d_i используется для финального агрегирования,
                # а выход FeatureTransformer (вероятно, 'y') идет на вход следующего шага.
                # Будем использовать 'y' для согласованности размерностей.
                current_features = y # (B, decision_dim)

                # 7. Вычисляем энтропию маски для регуляризации разреженности
                # Усредняем энтропию по батчу для этого шага
                entropy_step = -torch.sum(mask * torch.log(mask + 1e-10), dim=-1)
                total_entropy = total_entropy + torch.mean(entropy_step) # Суммируем средние энтропии шагов

        # 8. Финальный выход через линейный слой
        # Используем агрегированный decision output
        outputs = self.fc_out(agg_decision_output) # (B, output_dim)

        # 9. Вычисляем среднюю потерю энтропии по всем шагам
        if masks:
            avg_entropy_loss = total_entropy / self.n_steps # Делим суммарную среднюю энтропию на кол-во шагов
        else:
            avg_entropy_loss = torch.tensor(0.0, device=x.device)

        return outputs, avg_entropy_loss # Возвращаем логиты и потерю энтропии


# --------------------------------------------------------------------------
# 5.  TabNet – полный класс с эмбеддингами
# --------------------------------------------------------------------------
class TabNet(nn.Module):
    """Полная модель TabNet, объединяющая эмбеддинги и TabNetCore."""
    def __init__(self,
                 num_continuous: int,                 # Количество числовых признаков
                 cat_dims: Sequence[int],             # Размеры словарей категориальных признаков
                 cat_idx: Sequence[int],              # Индексы категориальных признаков в исходном тензоре
                 d_model: int = 8,                    # Размерность эмбеддингов (одинаковая для всех)
                 output_dim: int = 1,                 # Размерность выходного слоя
                 dropout_emb: float = 0.05,           # Dropout после эмбеддингов
                 att_momentum: float = 0.1,           # Momentum для BN в AttentiveTransformer
                 **core_kw):                          # Параметры для TabNetCore
        super().__init__()

        # Определяем индексы числовых признаков
        num_features_total = num_continuous + len(cat_idx)
        self.num_idx = [i for i in range(num_features_total) if i not in cat_idx]
        self.cat_idx = list(cat_idx) # Сохраняем как список

        # Создаем слои эмбеддингов
        self.num_emb = NumericEmbedding(len(self.num_idx), d_model) \
                       if self.num_idx else nn.Identity() # Identity если числовых нет
        self.cat_emb = CatEmbedding(cat_dims, d_model) \
                       if self.cat_idx else nn.Identity() # Identity если категориальных нет

        # Вычисляем размерность входа для TabNetCore
        # Каждый признак (числовой или категориальный) эмбеддится в d_model
        flat_dim = num_features_total * d_model
        if flat_dim == 0:
             raise ValueError("Невозможно создать TabNet без признаков.")

        self.core = TabNetCore(flat_dim, output_dim, att_momentum=att_momentum, **core_kw)
        self.drop = nn.Dropout(dropout_emb)

    def _split(self, X: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Разделяет входной тензор на числовые и категориальные части."""
        X_num = X[:, self.num_idx].float() if self.num_idx else None
        X_cat = X[:, self.cat_idx].long()  if self.cat_idx else None
        return X_num, X_cat

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход модели.

        Возвращает:
            - outputs: Финальный выход модели (логиты).
            - sparse_loss: Потеря разреженности (средняя энтропия масок).
        """
        X_num, X_cat = self._split(X)
        feats = []
        # Применяем эмбеддинги
        if X_num is not None and isinstance(self.num_emb, NumericEmbedding):
            feats.append(self.num_emb(X_num)) # (B, F_num, d)
        if X_cat is not None and isinstance(self.cat_emb, CatEmbedding):
            feats.append(self.cat_emb(X_cat)) # (B, F_cat, d)

        if not feats:
             raise RuntimeError("Нет признаков для обработки после эмбеддингов.")

        # Объединяем эмбеддинги по размерности признаков
        # feats_combined: (B, F_all, d)
        feats_combined = torch.cat(feats, dim=1)

        # Применяем Dropout и выравниваем в 2D тензор для TabNetCore
        # feats_flat: (B, F_all * d)
        feats_flat = self.drop(feats_combined).flatten(1)

        # Пропускаем через TabNetCore
        outputs, sparse_loss = self.core(feats_flat)

        return outputs, sparse_loss