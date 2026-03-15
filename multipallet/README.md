# X5 Tech — Smart 3D Packing

Соревновательный фреймворк задачи **3D Bin Packing** от X5 Tech (AI ЧЕМП).
Цель — оптимально упаковать коробки на паллеты с максимизацией плотности, соблюдением физических ограничений и учётом хрупкости/ориентации товаров.

---

## Содержание

1. [Задача](#задача)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Описание скриптов](#описание-скриптов)
5. [Форматы данных](#форматы-данных)
6. [Система оценки](#система-оценки)
7. [Последовательность работы](#последовательность-работы)

---

## Задача

Дана паллета с ограничениями по габаритам и весу. Дан набор коробок с атрибутами:

| Атрибут | Описание |
|---------|----------|
| `strict_upright` | Если `true` — коробку нельзя класть на бок, только вращение вокруг вертикальной оси |
| `fragile` | Если `true` — поверх нельзя класть нехрупкий груз суммарным весом >2 кг |
| `stackable` | Если `false` — поверх этой коробки ничего нельзя ставить |

Задача — разместить максимально возможное количество коробок, не нарушая:
- геометрических ограничений (не выходить за пределы паллеты)
- физических ограничений (поддержка ≥60% площади основания, вес паллеты)
- правил ориентации и хрупкости

Задача относится к классу **NP-трудных (3D Bin Packing Problem)**. В репозитории реализованы два подхода: жадный алгоритм и LNS (Large Neighborhood Search).

**Сценарии задач:**

| Сценарий | Описание |
|----------|----------|
| `heavy_water` | Тяжёлые нехрупкие коробки, упор на ограничение по весу |
| `fragile_tower` | Смесь крупных хрупких и нехрупких коробок |
| `liquid_tetris` | Все коробки строго вертикально, высокие + мелкие плоские |
| `random_mixed` | 4–7 случайных типов товаров в произвольном сочетании |

---

## Структура репозитория

```
X5_final/
│
├── solver.py                    # Жадный солвер (Extreme Points + гравитационный спуск)
├── lns_solver.py                # LNS солвер (Beam Search + Simulated Annealing)
├── evaluate_batch.py            # Пакетная оценка решений через validator
├── validate_lns_vs_greedy.py    # Живое сравнение LNS vs Greedy на датасете
├── plot_comparison.py           # Генерация всех 6 сравнительных графиков
├── select_best_pairs.py         # Отбор 5 лучших пар задач для детального сравнения
│
├── data/
│   ├── datasets/                # Готовые датасеты задач
│   │   ├── dataset_100.json     # 100 задач (25 на сценарий)
│   │   ├── dataset_200.json     # 200 задач (50 на сценарий) — основной
│   │   ├── dataset_1000.json    # 1000 задач
│   │   └── dataset_2000.json    # 2000 задач
│   │
│   ├── solutions/               # Предвычисленные решения
│   │   ├── greedy_solution_single.json  # Жадный, одна паллета на задачу
│   │   ├── greedy_sol_multi.json        # Жадный, неограниченное кол-во паллет
│   │   ├── lns_solution_single.json     # LNS, одна паллета на задачу
│   │   └── lns_sol_multi.json           # LNS, неограниченное кол-во паллет
│   │
│   ├── comparison_pairs/        # 10 JSON файлов с топ-5 показательными парами
│   │
│   └── scripts/
│       ├── generator.py         # Генератор задач (детерминированный, seed=42)
│       ├── validator.py         # Валидатор и оценщик решений
│       └── build_dataset.py     # Сборка датасета из generator
│
└── output/                      # Сгенерированные PNG-графики
    ├── cmp_1_final_score.png    # Итоговый балл по сценариям
    ├── cmp_2_volume_util.png    # Утилизация объёма
    ├── cmp_3_item_coverage.png  # Покрытие товаров
    ├── cmp_4_scatter.png        # Scatter: LNS vs Greedy по задачам
    ├── cmp_5_winrate.png        # Win/Tie/Loss по сценариям
    └── pallet_comparison.png    # Среднее кол-во паллет (multi-pallet режим)
```

---

## Быстрый старт

### Требования

- Python 3.10+
- Виртуальное окружение `x5_venv/` с пакетами: `numpy`, `matplotlib`

```bash
# Активация окружения (Windows bash / Git Bash)
source x5_venv/Scripts/activate
```

### Запуск солвера и оценка за 3 шага

```bash
# 1. Решить 10 задач жадным алгоритмом
python solver.py \
    --dataset data/datasets/dataset_100.json \
    --output  data/solutions/my_solution.json

# 2. Оценить решение
python evaluate_batch.py \
    data/datasets/dataset_100.json \
    data/solutions/my_solution.json

# 3. Построить графики сравнения с LNS
python plot_comparison.py \
    --greedy-single data/solutions/my_solution.json \
    --lns-single    data/solutions/lns_solution_single.json \
    --output-dir    output/
```

---

## Описание скриптов

### `solver.py` — Жадный солвер

Реализует эвристику **Extreme Points + Bottom-Left-Fill**:
- Перебирает кандидатные точки размещения (крайние точки пространства)
- Проверяет все 6 вариантов вращения коробки
- Сортирует коробки по убыванию объёма (FFD-эвристика)
- Использует пространственный хэш-грид для быстрых проверок коллизий (O(k) вместо O(n))

```bash
# Одна паллета на задачу
python solver.py --dataset data/datasets/dataset_200.json --output data/solutions/out.json

# Мульти-паллетный режим (открывает новые паллеты по мере необходимости)
python solver.py --dataset data/datasets/dataset_200.json --output data/solutions/out_multi.json --multi-pallet

# Одна задача из файла
python solver.py --input request_heavy_water.json --output solution.json
```

**Время:** ~50–300 мс/задача | **Версия:** `greedy-1.0` / `greedy-1.0-multi`

---

### `lns_solver.py` — LNS солвер

Реализует **Large Neighborhood Search** с улучшениями:
- **Фаза 0:** Multi-start greedy с 3-уровневой сортировкой (обычные → хрупкие → unstackable)
- **Фаза 1:** LNS — итеративное уничтожение части решения и восстановление через Beam Search
- **Принятие:** Simulated Annealing (вероятностное принятие ухудшений)
- **5 операторов уничтожения** с адаптивными весами (какой оператор лучше работает — тот чаще выбирается)
- **ILS-kicks** для выхода из локальных оптимумов

```bash
# Стандартный режим (~9 с/задача)
python lns_solver.py --dataset data/datasets/dataset_200.json --output data/solutions/out_lns.json

# Высококачественный режим (~28 с/задача, beam_width=8)
python lns_solver.py --dataset data/datasets/dataset_200.json --output data/solutions/out_lns_hq.json --hq

# Мульти-паллетный режим (~30 с/задача, beam_width=16)
python lns_solver.py --dataset data/datasets/dataset_200.json --output data/solutions/out_lns_multi.json --multi-pallet
```

**Время:** 0.7–30 с/задача | **Версии:** `lns-1.0`, `lns-hq-1.0`, `lns-multi-1.0`

---

### `evaluate_batch.py` — Пакетная оценка

Читает файл задач и файл решений, прогоняет каждое решение через валидатор, выводит таблицу метрик с разбивкой по сценариям.

```bash
python evaluate_batch.py <requests.json> <responses.json>

# Примеры
python evaluate_batch.py data/datasets/dataset_200.json data/solutions/greedy_solution_single.json
python evaluate_batch.py data/datasets/dataset_200.json data/solutions/lns_solution_single.json
```

Поддерживает все форматы: одна задача, список задач, мульти-паллетные решения (вложенные списки).

---

### `validate_lns_vs_greedy.py` — Живое сравнение

Берёт задачи из датасета, **запускает оба солвера** на каждой задаче прямо сейчас и выводит построчное сравнение. Завершается с кодом `0` если LNS ≥ Greedy в среднем, иначе `1`.

```bash
# Сравнение на 10 задачах (быстро, ~1.5 мин)
python validate_lns_vs_greedy.py --n 10

# Полный датасет
python validate_lns_vs_greedy.py --dataset data/datasets/dataset_200.json

# Мульти-паллетный режим
python validate_lns_vs_greedy.py --dataset data/datasets/dataset_200.json --multi

# HQ-режим (~30 с/задача)
python validate_lns_vs_greedy.py --n 20 --hq
```

---

### `plot_comparison.py` — Визуализация

Генерирует 6 сравнительных графиков в папку `output/`. По умолчанию использует предвычисленные решения из `data/solutions/`.

```bash
# Все графики с настройками по умолчанию
python plot_comparison.py

# Свои файлы решений
python plot_comparison.py \
    --dataset       data/datasets/dataset_200.json \
    --greedy-single data/solutions/greedy_solution_single.json \
    --lns-single    data/solutions/lns_solution_single.json \
    --greedy-multi  data/solutions/greedy_sol_multi.json \
    --lns-multi     data/solutions/lns_sol_multi.json \
    --output-dir    output/
```

| График | Файл | Описание |
|--------|------|----------|
| 1 | `cmp_1_final_score.png` | Средний итоговый балл по сценариям |
| 2 | `cmp_2_volume_util.png` | Утилизация объёма паллеты |
| 3 | `cmp_3_item_coverage.png` | Покрытие товаров (размещено / запрошено) |
| 4 | `cmp_4_scatter.png` | Scatter: LNS score vs Greedy score на каждую задачу |
| 5 | `cmp_5_winrate.png` | Доля побед / ничьих / поражений LNS |
| 6 | `pallet_comparison.png` | Среднее кол-во паллет (multi-pallet режим) |

---

### `select_best_pairs.py` — Отбор пар для анализа

Находит 5 задач, где LNS наиболее заметно превосходит Greedy (по кол-ву паллет и баллу), и экспортирует пары решений в `data/comparison_pairs/`.

```bash
python select_best_pairs.py
```

---

### `data/scripts/build_dataset.py` — Генерация датасета

```bash
# Сгенерировать dataset_100.json (100 задач, детерминированно)
python data/scripts/build_dataset.py
```

---

## Форматы данных

### Входной JSON (заявка на упаковку)

```json
{
  "task_id": "heavy_water_0000",
  "pallet": {
    "length_mm": 1200,
    "width_mm": 800,
    "max_height_mm": 1800,
    "max_weight_kg": 1500.0
  },
  "boxes": [{
    "sku_id": "SKU-WATER-7890",
    "length_mm": 275,
    "width_mm": 194,
    "height_mm": 330,
    "weight_kg": 9.2,
    "quantity": 10,
    "strict_upright": true,
    "fragile": false,
    "stackable": true
  }]
}
```

### Выходной JSON (решение)

```json
{
  "task_id": "heavy_water_0000",
  "solver_version": "lns-1.0",
  "solve_time_ms": 9140,
  "placements": [{
    "sku_id": "SKU-WATER-7890",
    "instance_index": 0,
    "position": { "x_mm": 0, "y_mm": 0, "z_mm": 0 },
    "dimensions_placed": { "length_mm": 275, "width_mm": 194, "height_mm": 330 },
    "rotation_code": "LWH"
  }],
  "unplaced": [{
    "sku_id": "SKU-SUGAR-8808",
    "quantity_unplaced": 5,
    "reason": "no_space"
  }]
}
```

`rotation_code` задаёт порядок осей после вращения: `LWH`, `LHW`, `WLH`, `WHL`, `HLW`, `HWL`.
`position` — координата нижнего левого ближнего угла короба (мм).

### Мульти-паллетное решение

Список списков: внешний — по задачам, внутренний — по паллетам одной задачи:

```json
[
  [
    { "task_id": "heavy_water_0000", "solver_version": "lns-multi-1.0", "placements": [...], ... },
    { "task_id": "heavy_water_0000", "solver_version": "lns-multi-1.0", "placements": [...], ... }
  ],
  ...
]
```

---

## Система оценки

### Жёсткие ограничения (нарушение = 0 баллов за задачу)

| # | Ограничение | Описание |
|---|-------------|----------|
| 1 | **Границы** | Все коробки полностью внутри паллеты |
| 2 | **Коллизии** | Объём пересечений = 0 |
| 3 | **Опора** | ≥60% площади основания опирается на паллету или другой короб |
| 4 | **Ориентация** | `strict_upright: true` — только вращение вокруг Z-оси |
| 5 | **Вес** | Суммарный вес всех коробок ≤ `max_weight_kg` паллеты |

### Мягкие метрики (итоговый балл 0–1)

| Метрика | Вес | Детали |
|---------|-----|--------|
| Утилизация объёма | **50%** | Суммарный объём коробов / объём паллеты |
| Покрытие товаров | **30%** | Размещено штук / запрошено штук |
| Хрупкость | **10%** | Штраф, если над `fragile`-коробом нехрупкий груз >2 кг |
| Время решения | **10%** | <1 с → 1.0; 1–5 с → 0.7; 5–30 с → 0.3; >30 с → 0.0 |

---

## Последовательность работы

Ниже — полная последовательность от нуля до графиков:

```bash
source x5_venv/Scripts/activate

# ── Шаг 1. Сгенерировать датасет ────────────────────────────────────────────
python data/scripts/build_dataset.py
# → data/datasets/dataset_100.json

# ── Шаг 2. Запустить солверы ─────────────────────────────────────────────────
# Жадный (быстро, ~30 с на 100 задач)
python solver.py \
    --dataset data/datasets/dataset_100.json \
    --output  data/solutions/my_greedy.json

# LNS (медленнее, ~15 мин на 100 задач)
python lns_solver.py \
    --dataset data/datasets/dataset_100.json \
    --output  data/solutions/my_lns.json

# ── Шаг 3. Оценить решения ───────────────────────────────────────────────────
python evaluate_batch.py data/datasets/dataset_100.json data/solutions/my_greedy.json
python evaluate_batch.py data/datasets/dataset_100.json data/solutions/my_lns.json

# ── Шаг 4. Сравнить LNS vs Greedy вживую (на малом числе задач) ─────────────
python validate_lns_vs_greedy.py \
    --dataset data/datasets/dataset_100.json \
    --n 10

# ── Шаг 5. Построить графики ─────────────────────────────────────────────────
python plot_comparison.py \
    --dataset       data/datasets/dataset_100.json \
    --greedy-single data/solutions/my_greedy.json \
    --lns-single    data/solutions/my_lns.json \
    --output-dir    output/
# → output/cmp_1_final_score.png ... output/cmp_5_winrate.png

# ── Шаг 6 (опционально). Мульти-паллетный режим ──────────────────────────────
python solver.py   --dataset data/datasets/dataset_100.json --output data/solutions/my_greedy_multi.json --multi-pallet
python lns_solver.py --dataset data/datasets/dataset_100.json --output data/solutions/my_lns_multi.json --multi-pallet

python plot_comparison.py \
    --dataset      data/datasets/dataset_100.json \
    --greedy-single data/solutions/my_greedy.json \
    --lns-single    data/solutions/my_lns.json \
    --greedy-multi  data/solutions/my_greedy_multi.json \
    --lns-multi     data/solutions/my_lns_multi.json \
    --output-dir    output/
# → output/pallet_comparison.png (добавляется график кол-ва паллет)

# ── Шаг 7 (опционально). Отобрать лучшие пары для презентации ───────────────
python select_best_pairs.py
# → data/comparison_pairs/{task_id}_greedy.json + _lns.json (5 пар)
```

### Результаты на `dataset_200.json` (предвычисленные решения)

| Метрика | Жадный | LNS | Улучшение |
|---------|--------|-----|-----------|
| Средний балл | ~0.70 | ~0.73 | +4% |
| Утилизация объёма | ~0.62 | ~0.67 | +8% |
| Покрытие товаров | ~0.55 | ~0.60 | +9% |
| Ср. паллет/задача (multi) | ~4.6 | ~3.2 | −31% |
