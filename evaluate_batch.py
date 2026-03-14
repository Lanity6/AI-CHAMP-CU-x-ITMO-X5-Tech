"""
evaluate_batch.py — батч-оценка решений упаковки.

Принимает два JSON-файла:
  - requests: один task или список tasks (как dataset_1000.json)
  - responses: один response или список responses

Использование:
  python evaluate_batch.py <requests.json> <responses.json>
  python evaluate_batch.py data/dataset_1000.json my_solutions.json
"""

import json
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Добавляем путь к validator.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "data" / "scripts"))
from validator import evaluate_solution


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_to_list(data) -> list:
    return data if isinstance(data, list) else [data]


def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.1%}"


def print_task_result(task_id: str, result: dict, idx: int, total: int):
    prefix = f"[{idx}/{total}]"
    if not result["valid"]:
        print(f"{prefix} {task_id:30s}  INVALID - {result['error']}")
        return

    m = result["metrics"]
    score = result["final_score"]
    print(
        f"{prefix} {task_id:30s}  "
        f"score={score:.4f}  "
        f"vol={m['volume_utilization']:.3f}  "
        f"cov={m['item_coverage']:.3f}  "
        f"frag={m['fragility_score']:.3f}  "
        f"time={m['time_score']:.1f}"
    )


def print_summary(valid_results: list, invalid_count: int, total: int):
    print("\n" + "=" * 72)
    print(f"  ИТОГ: {total} задач | валидных: {len(valid_results)} | невалидных: {invalid_count}")
    print("=" * 72)

    if not valid_results:
        print("  Нет валидных результатов.")
        return

    def avg(key):
        return sum(r["metrics"][key] for r in valid_results) / len(valid_results)

    def avg_score():
        return sum(r["final_score"] for r in valid_results) / len(valid_results)

    vol   = avg("volume_utilization")
    cov   = avg("item_coverage")
    frag  = avg("fragility_score")
    time_ = avg("time_score")
    score = avg_score()

    print(f"\n  Среднее по валидным задачам ({len(valid_results)} шт.):")
    print(f"  Итоговый балл     {bar(score)}")
    print(f"  Утилизация объёма {bar(vol)}")
    print(f"  Покрытие товаров  {bar(cov)}")
    print(f"  Хрупкость         {bar(frag)}")
    print(f"  Время             {bar(time_)}")

    scores = [r["final_score"] for r in valid_results]
    scores.sort()
    n = len(scores)
    p25 = scores[int(n * 0.25)]
    p50 = scores[int(n * 0.50)]
    p75 = scores[int(n * 0.75)]

    print(f"\n  Распределение final_score:")
    print(f"    min={scores[0]:.4f}  p25={p25:.4f}  median={p50:.4f}  "
          f"p75={p75:.4f}  max={scores[-1]:.4f}")

    # Гистограмма по диапазонам
    buckets = [0] * 10
    for s in scores:
        i = min(int(s * 10), 9)
        buckets[i] += 1
    print("\n  Гистограмма (шаг 0.1):")
    for i, count in enumerate(buckets):
        lo, hi = i / 10, (i + 1) / 10
        bar_str = "#" * count
        print(f"    [{lo:.1f}-{hi:.1f})  {bar_str:30s} {count}")

    if invalid_count:
        print(f"\n  [!] {invalid_count} задач невалидны (0 баллов в итоговом рейтинге).")

    # Итоговый балл с учётом невалидных (как будто 0)
    total_avg = sum(r["final_score"] for r in valid_results) / total if total > 0 else 0.0
    print(f"\n  Средний балл по ВСЕМ задачам (включая невалидные): {total_avg:.4f}")
    print("=" * 72)


def get_scenario_type(task_id: str) -> str:
    """
    Извлекает тип сценария из task_id.
    Форматы: 'heavy_water_0001' -> 'heavy_water', 'task_fragile_tower' -> 'fragile_tower',
    'task_heavy_water' -> 'heavy_water'. Если не распознан — 'unknown'.
    """
    known = ["heavy_water", "fragile_tower", "liquid_tetris", "random_mixed"]
    for sc in known:
        if sc in task_id:
            return sc
    return "unknown"


def print_scenario_breakdown(results_with_ids: list, invalid_by_scenario: dict, total_by_scenario: dict):
    """Печатает сводную таблицу метрик по каждому типу сценария."""
    # Группируем валидные результаты по сценарию
    by_scenario: dict = {}
    for task_id, result in results_with_ids:
        sc = get_scenario_type(task_id)
        by_scenario.setdefault(sc, []).append(result)

    all_scenarios = sorted(set(list(by_scenario.keys()) + list(invalid_by_scenario.keys())))
    if not all_scenarios:
        return

    print("\n" + "=" * 72)
    print("  РАЗБИВКА ПО ТИПАМ ЗАДАЧ")
    print("=" * 72)

    header = f"  {'Сценарий':<18} {'N':>5} {'inv':>4}  {'score':>6}  {'vol':>6}  {'cov':>6}  {'frag':>6}  {'time':>6}"
    print(header)
    print("  " + "-" * 68)

    for sc in all_scenarios:
        recs = by_scenario.get(sc, [])
        inv  = invalid_by_scenario.get(sc, 0)
        tot  = total_by_scenario.get(sc, len(recs) + inv)
        n    = len(recs)

        if n == 0:
            print(f"  {sc:<18} {tot:>5} {inv:>4}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
            continue

        def avg(key):
            return sum(r["metrics"][key] for r in recs) / n

        score = sum(r["final_score"] for r in recs) / n
        vol   = avg("volume_utilization")
        cov   = avg("item_coverage")
        frag  = avg("fragility_score")
        time_ = avg("time_score")

        print(f"  {sc:<18} {tot:>5} {inv:>4}  {score:>6.4f}  {vol:>6.3f}  {cov:>6.3f}  {frag:>6.3f}  {time_:>6.3f}")

    print("=" * 72)


def main():
    if len(sys.argv) != 3:
        print("Использование: python evaluate_batch.py <requests.json> <responses.json>")
        sys.exit(1)

    req_path, resp_path = sys.argv[1], sys.argv[2]

    print(f"Загружаем запросы:  {req_path}")
    print(f"Загружаем ответы:   {resp_path}")

    requests_raw  = normalize_to_list(load_json(req_path))
    responses_raw = normalize_to_list(load_json(resp_path))

    # Индексируем по task_id
    requests_by_id  = {r["task_id"]: r for r in requests_raw}
    responses_by_id = {r["task_id"]: r for r in responses_raw}

    # Пересечение по task_id
    common_ids = sorted(set(requests_by_id) & set(responses_by_id))
    only_req   = set(requests_by_id) - set(responses_by_id)
    only_resp  = set(responses_by_id) - set(requests_by_id)

    if only_req:
        print(f"\n[!] {len(only_req)} задач без ответа (пропускаем): {sorted(only_req)[:5]}{'...' if len(only_req) > 5 else ''}")
    if only_resp:
        print(f"[!] {len(only_resp)} ответов без запроса (пропускаем): {sorted(only_resp)[:5]}{'...' if len(only_resp) > 5 else ''}")

    if not common_ids:
        print("Нет совпадающих task_id между файлами.")
        sys.exit(1)

    print(f"\nОцениваем {len(common_ids)} задач...\n")
    print("-" * 72)

    valid_results = []
    valid_results_with_ids = []  # (task_id, result)
    invalid_count = 0
    invalid_by_scenario: dict = {}
    total_by_scenario: dict = {}
    total = len(common_ids)

    for idx, task_id in enumerate(common_ids, 1):
        req  = requests_by_id[task_id]
        resp = responses_by_id[task_id]
        result = evaluate_solution(req, resp)
        print_task_result(task_id, result, idx, total)

        sc = get_scenario_type(task_id)
        total_by_scenario[sc] = total_by_scenario.get(sc, 0) + 1

        if result["valid"]:
            valid_results.append(result)
            valid_results_with_ids.append((task_id, result))
        else:
            invalid_count += 1
            invalid_by_scenario[sc] = invalid_by_scenario.get(sc, 0) + 1

    print_summary(valid_results, invalid_count, total)
    print_scenario_breakdown(valid_results_with_ids, invalid_by_scenario, total_by_scenario)


if __name__ == "__main__":
    main()
