# 3D Bin Packing — X5 Tech x ITMO

Решение задачи оптимальной 3D-укладки товаров на паллету для фуд-ритейла.

## Описание кейса

**Задача:** разместить максимальное количество коробок на паллете с учетом физических ограничений:
- Габариты паллеты (длина, ширина, максимальная высота)
- Грузоподъемность (максимальный вес)
- Поддержка снизу (не менее 60% площади основания)
- Хрупкость — на хрупкие предметы нельзя ставить тяжелые (>2 кг)
- Штабелируемость — на нештабелируемые нельзя ставить ничего
- Ориентация — некоторые предметы только вертикально (strict_upright)

**Метрика оценки:**
```
score = 0.50 * volume_utilization + 0.30 * item_coverage + 0.10 * fragility_score + 0.10 * time_score
```

## Структура проекта

```
├── solvers/                    # Солверы
│   ├── lns_solver.py           # LNS (Large Neighborhood Search) — основной
│   ├── solve_x5.py             # Maxrects 2D packer
│   ├── greedy_algorithm.py     # Жадный алгоритм
│   └── gan_ga_solver.py        # GAN (Generative Adversarial Networks)
├── datasets/                   # Входные датасеты (JSON)
├── results/                    # Результаты солверов
├── vizualizator/               # 3D-визуализация
│   ├── visualizator.py         # Просмотр 1-3 паллет
│   └── vizualizator_multipallet.py  # Сетка до 9 паллет
├── dataset_generation/         # Генерация датасетов
├── benchmark_report.py         # Генерация CSV-отчета
├── plot_benchmark.py           # Построение графиков
├── benchmark.sh                # Полный запуск бенчмарка
└── validator.py                # Валидация и скоринг
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрый старт

### 1. Запуск LNS-солвера на датасете

```bash
PYTHONPATH=. python3 solvers/lns_solver.py datasets/dataset_100.json results/lns_solver/dataset_100.json
```

Для нескольких датасетов:
```bash
PYTHONPATH=. python3 solvers/lns_solver.py datasets/dataset_200_2.json results/lns_solver/dataset_200_2.json
PYTHONPATH=. python3 solvers/lns_solver.py datasets/dataset_200_3.json results/lns_solver/dataset_200_3.json
PYTHONPATH=. python3 solvers/lns_solver.py datasets/dataset_150.json results/lns_solver/dataset_150.json
```

### 2. Генерация CSV-отчета

Собирает метрики из всех результатов в `results/` в единый CSV:

```bash
python3 benchmark_report.py results/ benchmark.csv
```

### 3. Построение графиков

```bash
python3 plot_benchmark.py benchmark.csv
```

Создает `benchmark_plot.png` — столбчатая диаграмма среднего score по солверам и датасетам.

### 4. Полный бенчмарк (все солверы на всех датасетах)

```bash
./benchmark.sh
```

Или только один солвер:
```bash
./benchmark.sh lns_solver
```

### 5. Визуализация результатов

Открывает интерактивный 3D-просмотр укладки:

```bash
cd vizualizator/
python3 visualizator.py
```

По умолчанию отображает результаты LNS на `dataset_100.json`. Для другого датасета отредактируйте пути `INPUT_JSON_PATH` и `RESULT_JSON_PATH` в файле.

Мультипаллетный вид (сетка до 9 паллет одновременно):
```bash
cd vizualizator/
python3 vizualizator_multipallet.py
```

Для смены датасета в мультипаллетном визуализаторе отредактируйте пути в файле `vizualizator_multipallet.py`:
```python
INPUT_JSON_PATH = "../datasets/dataset_200_3.json"
RESULT_JSON_PATH = "../results/lns_solver/dataset_200_3.json"
```

**Управление визуализатором:**
- Чекбоксы слоев — включение/выключение видимости по слоям
- Previous / Next — переключение между задачами
- Стрелки вверх/вниз — прокрутка слоев (если > 15)
- «Show all» — показать/скрыть все слои сразу
- Легенда — типы коробок с пометками (хрупкие, вертикальные)

## Пайплайн от датасета до графика

```
datasets/*.json
    │
    ▼
solvers/lns_solver.py  ──►  results/lns_solver/*.json
    │
    ▼
benchmark_report.py    ──►  benchmark.csv
    │
    ▼
plot_benchmark.py      ──►  benchmark_plot.png

results/ + datasets/
    │
    ▼
vizualizator/visualizator.py  ──►  интерактивный 3D-просмотр
```
