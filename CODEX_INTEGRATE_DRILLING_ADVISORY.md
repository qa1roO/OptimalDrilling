# Codex task: подключить drilling advisory ML-модели к симулятору

## Цель

Нужно подключить к существующему симулятору буровой установки ML advisory-layer, который на каждом шаге replay/simulation:

1. получает текущую telemetry;
2. определяет текущую энергоёмкость породы / drilling regime;
3. строит локальную future-ROP поверхность:

```text
pressure_axis × pressure_rotation → predicted target_speed_near5
```

4. показывает на 3D-графике:
   - текущую точку оператора;
   - рекомендованную точку;
   - поверхность текущего класса энергоёмкости;
   - predicted uplift;
5. при смене энергоёмкости перестраивает поверхность под новый energy type.

На первом этапе нужен **replay/advisory mode**, не closed-loop. То есть симулятор проигрывает историческую скважину из `united_rock_energy_segment_quantile.csv`, а ML-модель только советует, куда стоило бы сместить `pressure_axis / pressure_rotation`.

---

## Текущие файлы проекта

Рабочий набор данных и артефактов:

```text
C:.
│   rock_energy_segment_quantile_config.json
│   rock_energy_segment_quantile_surfaces_only.ipynb
│   train_drilling_advisory_light_penalty_final.ipynb
│   united_rock_energy_segment_quantile.csv
│
├───drilling_advisory_light_penalty_artifacts
│       feature_config.json
│       offline_recommendations_light_penalty.csv
│       optimizer_config.json
│       optimizer_summary.csv
│       rotation_model_near5.joblib
│       speed_model_near5.joblib
│       surface_ranges_by_energy_type.json
│       training_report.json
│       uplift_by_energy_type.csv
│
├───plotly_surfaces_html
│       surface_hard_high_energy.html
│       surface_medium_high_energy.html
│       surface_medium_low_energy.html
│       surface_soft_low_energy.html
│
└───rock_energy_segment_quantile_artifacts
        expected_speed_from_controls_model.joblib
```

Для подключения к симулятору главные файлы:

```text
united_rock_energy_segment_quantile.csv

drilling_advisory_light_penalty_artifacts/
    rotation_model_near5.joblib
    speed_model_near5.joblib
    feature_config.json
    optimizer_config.json
    surface_ranges_by_energy_type.json
```

Опционально:

```text
rock_energy_segment_quantile_artifacts/
    expected_speed_from_controls_model.joblib
```

Этот файл нужен только если потом будет online-расчёт `hardness_score_smooth` и `rock_energy_type_final` из сырой telemetry. В первом replay-варианте они уже есть в `united_rock_energy_segment_quantile.csv`.

---

## ML-логика

Финальная модель работает так:

```text
current telemetry + history + candidate pressure_axis/pressure_rotation
    ↓
rotation_model_near5
    ↓
predicted average rotation over next 5 points
    ↓
speed_model_near5
    ↓
predicted average speed over next 5 points
    ↓
light-penalty optimizer
    ↓
recommended pressure_axis / pressure_rotation
```

Target модели:

```text
target_rotation_near5 = average rotation over next 5 points
target_speed_near5    = average speed over next 5 points
```

То есть модель не предсказывает текущую `speed`, а оценивает ближайший future ROP.

---

## Что нужно добавить в репозиторий

Добавить новый файл:

```text
simulator/app/data/advisory_engine.py
```

В нём реализовать класс:

```python
class AdvisoryEngine:
    def __init__(self, artifact_dir: str | Path):
        ...

    def update(self, telemetry_row: dict) -> dict | None:
        ...

    def get_recommendation(self) -> dict | None:
        ...
```

Дополнительно можно добавить:

```text
simulator/app/data/advisory_types.py
```

с dataclass-структурами, если это удобно.

---

## Что должен делать AdvisoryEngine

### 1. Загружать артефакты

В `__init__` загрузить:

```python
import json
import joblib
from pathlib import Path

artifact_dir = Path(artifact_dir)

rotation_model = joblib.load(artifact_dir / "rotation_model_near5.joblib")
speed_model = joblib.load(artifact_dir / "speed_model_near5.joblib")

feature_config = json.load(open(artifact_dir / "feature_config.json", encoding="utf-8"))
optimizer_config = json.load(open(artifact_dir / "optimizer_config.json", encoding="utf-8"))
surface_ranges = json.load(open(artifact_dir / "surface_ranges_by_energy_type.json", encoding="utf-8"))
```

Из `feature_config.json` взять:

```text
base_numeric_features
categorical_features
speed_numeric_features
energy_type_column
```

Из `optimizer_config.json` взять:

```text
grid_size_default
max_delta_frac_default
change_penalty_weight
boundary_penalty_weight
boundary_start
```

---

### 2. Хранить rolling buffer telemetry

Модель использует lag/rolling признаки:

```text
pressure_axis_lag1
pressure_axis_lag3
pressure_axis_lag6
pressure_rotation_lag1
...
speed_roll_mean_12
rotation_roll_mean_12
hardness_score_smooth_roll_mean_12
...
```

Поэтому `AdvisoryEngine` должен хранить последние telemetry rows.

Рекомендуемый размер буфера:

```python
buffer_size = 60
```

Минимум для стабильной работы:

```python
30
```

Если точек меньше 30, можно возвращать `None` или recommendation=current controls.

---

### 3. Принимать telemetry row

На каждом шаге симулятор должен передавать в engine словарь с полями:

```text
processing_time
well_id
pressure_axis
pressure_rotation
rotation
speed
hardness_score_smooth
rock_energy_type_final
```

Пример:

```python
telemetry = {
    "processing_time": current_time,
    "well_id": well_id,
    "pressure_axis": p_ax,
    "pressure_rotation": p_rot,
    "rotation": rotation,
    "speed": speed,
    "hardness_score_smooth": hardness,
    "rock_energy_type_final": energy_type,
}
```

В replay mode эти поля уже берутся из:

```text
united_rock_energy_segment_quantile.csv
```

---

## Feature engineering внутри AdvisoryEngine

Нужно повторить feature engineering из финального notebook `train_drilling_advisory_light_penalty_final.ipynb`.

### Базовые признаки

Для последней строки буфера:

```python
total_pressure = pressure_axis + pressure_rotation

pressure_balance = pressure_axis / (total_pressure + EPS)

axis_over_rot_pressure = pressure_axis / (pressure_rotation + EPS)

rot_pressure_over_axis = pressure_rotation / (pressure_axis + EPS)

rotation_efficiency = rotation / (pressure_rotation + EPS)

axis_x_rotation = pressure_axis * rotation

rot_pressure_x_rotation = pressure_rotation * rotation

energy_input_proxy = pressure_axis + pressure_rotation * rotation

log_energy_input_proxy = np.log1p(energy_input_proxy)
```

### Временной шаг

```python
dt = current_processing_time - previous_processing_time
```

Если нет предыдущей точки:

```python
dt = median_dt или 0
```

Для replay можно брать фактическую разницу по времени.

### Lag-признаки

Для колонок:

```python
history_cols = [
    "pressure_axis",
    "pressure_rotation",
    "rotation",
    "speed",
    "hardness_score_smooth",
    "energy_input_proxy",
    "pressure_balance",
]
```

Нужны lag:

```python
lag1
lag3
lag6
lag12
```

Например:

```python
pressure_axis_lag1
pressure_axis_lag3
pressure_axis_lag6
pressure_axis_lag12
```

### Rolling-признаки

Для тех же `history_cols` нужны:

```python
roll_mean_6
roll_std_6
roll_mean_12
roll_std_12
roll_mean_30
roll_std_30
```

В текущей ML-модели реально используются в основном `roll_mean_12` и `roll_std_12`, но лучше считать все, как в notebook.

Важно: rolling-признаки должны считаться только по прошлым точкам, без текущей точки, как в training pipeline:

```python
shifted = series.shift(1)
rolling_mean = shifted.rolling(w).mean()
rolling_std = shifted.rolling(w).std()
```

В online buffer это значит: rolling для текущего шага считаем по предыдущим значениям.

### Diff-признаки

Для колонок:

```python
["pressure_axis", "pressure_rotation", "rotation", "speed", "hardness_score_smooth"]
```

нужны:

```python
diff1 = current - previous
rel_diff1 = diff1 / (abs(previous) + EPS)
```

В финальной модели используются `diff1`, но можно считать оба.

---

## Candidate grid

Для каждой текущей точки строится grid по `pressure_axis` и `pressure_rotation`.

Использовать настройки:

```python
GRID_SIZE = optimizer_config["grid_size_default"]  # обычно 21
MAX_DELTA_FRAC = optimizer_config["max_delta_frac_default"]  # обычно 0.08
```

### Локальные границы

```python
local_ax_low  = current_p_ax * (1 - MAX_DELTA_FRAC)
local_ax_high = current_p_ax * (1 + MAX_DELTA_FRAC)

local_rot_low  = current_p_rot * (1 - MAX_DELTA_FRAC)
local_rot_high = current_p_rot * (1 + MAX_DELTA_FRAC)
```

### Границы по energy type

Берутся из:

```text
surface_ranges_by_energy_type.json
```

Например:

```python
r = surface_ranges[current_energy_type]

p_ax_low = r["pressure_axis_q05"]
p_ax_high = r["pressure_axis_q95"]

p_rot_low = r["pressure_rotation_q05"]
p_rot_high = r["pressure_rotation_q95"]
```

### Итоговые границы

```python
p_ax_min = max(p_ax_low, local_ax_low)
p_ax_max = min(p_ax_high, local_ax_high)

p_rot_min = max(p_rot_low, local_rot_low)
p_rot_max = min(p_rot_high, local_rot_high)
```

Если диапазон схлопнулся, fallback:

```python
p_ax_min = local_ax_low
p_ax_max = local_ax_high

p_rot_min = local_rot_low
p_rot_max = local_rot_high
```

### Построение сетки

```python
p_ax_grid = np.linspace(p_ax_min, p_ax_max, GRID_SIZE)
p_rot_grid = np.linspace(p_rot_min, p_rot_max, GRID_SIZE)

PA, PR = np.meshgrid(p_ax_grid, p_rot_grid)
```

Собрать DataFrame:

```python
grid = pd.DataFrame({
    "pressure_axis": PA.ravel(),
    "pressure_rotation": PR.ravel(),
})
```

Все остальные признаки состояния копируются из текущего feature row.

После подстановки candidate `pressure_axis / pressure_rotation` нужно пересчитать derived-признаки:

```text
total_pressure
pressure_balance
axis_over_rot_pressure
rot_pressure_over_axis
rotation_efficiency
axis_x_rotation
rot_pressure_x_rotation
energy_input_proxy
log_energy_input_proxy
```

---

## Prediction

Для каждой строки grid:

### 1. Предсказать candidate target rotation

```python
grid["candidate_target_rotation"] = rotation_model.predict(
    grid[base_numeric_features + categorical_features]
)
```

### 2. Предсказать candidate target speed

```python
grid["pred_target_speed"] = speed_model.predict(
    grid[speed_numeric_features + categorical_features]
)
```

---

## Light penalty optimizer

Нужно реализовать тот же score, что в финальном notebook.

Параметры:

```python
CHANGE_PENALTY_WEIGHT = 0.010
BOUNDARY_PENALTY_WEIGHT = 0.020
BOUNDARY_START = 0.85
MAX_DELTA_FRAC = 0.08
```

Для каждой candidate-точки:

```python
delta_pressure_axis_frac =
    candidate_pressure_axis / current_pressure_axis - 1

delta_pressure_rotation_frac =
    candidate_pressure_rotation / current_pressure_rotation - 1
```

Нормированные изменения:

```python
axis_delta_norm = abs(delta_pressure_axis_frac) / MAX_DELTA_FRAC
rot_delta_norm = abs(delta_pressure_rotation_frac) / MAX_DELTA_FRAC
```

Штраф за величину изменения:

```python
change_penalty =
    CHANGE_PENALTY_WEIGHT
    * current_pred_speed
    * (axis_delta_norm**2 + rot_delta_norm**2)
    / 2
```

Штраф за близость к границе:

```python
axis_edge = clip((axis_delta_norm - BOUNDARY_START) / (1 - BOUNDARY_START), 0, 1)
rot_edge = clip((rot_delta_norm - BOUNDARY_START) / (1 - BOUNDARY_START), 0, 1)

boundary_penalty =
    BOUNDARY_PENALTY_WEIGHT
    * current_pred_speed
    * (axis_edge**2 + rot_edge**2)
    / 2
```

Финальный score:

```python
optimizer_score =
    pred_target_speed
    - change_penalty
    - boundary_penalty
```

Выбираем:

```python
best_idx = grid["optimizer_score"].argmax()
best = grid.iloc[best_idx]
```

---

## Как посчитать current predicted speed

Перед выбором recommendation нужно посчитать прогноз для текущих операторских параметров.

```python
current_grid = pd.DataFrame([current_feature_row])
current_grid["candidate_target_rotation"] = rotation_model.predict(
    current_grid[base_numeric_features + categorical_features]
)

current_pred_speed = speed_model.predict(
    current_grid[speed_numeric_features + categorical_features]
)[0]
```

Then:

```python
predicted_uplift_pct =
    100 * (best["pred_target_speed"] / current_pred_speed - 1)
```

---

## Формат результата AdvisoryEngine

`AdvisoryEngine.update(...)` должен возвращать структуру:

```python
{
    "energy_type": "medium_high_energy",

    "current": {
        "pressure_axis": 20000.0,
        "pressure_rotation": 15500.0,
        "rotation": 130.0,
        "speed": 0.018,
        "predicted_target_speed": 0.0184,
    },

    "recommended": {
        "pressure_axis": 19350.0,
        "pressure_rotation": 16100.0,
        "predicted_target_rotation": 135.2,
        "predicted_target_speed": 0.0189,
        "optimizer_score": 0.0188,
        "predicted_uplift_pct": 2.7,
        "delta_pressure_axis_pct": -3.2,
        "delta_pressure_rotation_pct": 3.9,
    },

    "surface": {
        "x": PA,                 # 2D numpy array
        "y": PR,                 # 2D numpy array
        "z": Z_speed,            # 2D numpy array predicted target speed
        "score": Z_score,        # 2D numpy array optimizer score
    }
}
```

Если UI не может принимать numpy arrays, преобразовать в lists:

```python
PA.tolist()
PR.tolist()
Z.tolist()
Z_score.tolist()
```

---

## Как подключить к существующему UI

В текущем симуляторе уже есть 3D-график/панель. Нужно заменить или дополнить старую synthetic surface-логику:

### Старый смысл

```text
cluster_id → synthetic ClusterProfile → synthetic surface
```

### Новый смысл

```text
rock_energy_type_final → ML future-speed surface → recommended point
```

То есть `cluster_id` можно временно переиспользовать как energy type id:

```python
ENERGY_TYPE_TO_ID = {
    "soft_low_energy": 0,
    "medium_low_energy": 1,
    "medium_high_energy": 2,
    "hard_high_energy": 3,
}
```

Но лучше в UI явно хранить:

```python
energy_type: str
```

---

## Что рисовать на 3D-графике

Ось X:

```text
pressure_axis
```

Ось Y:

```text
pressure_rotation
```

Ось Z:

```text
predicted target_speed_near5
```

Нужно нарисовать:

### 1. Surface

```python
x = recommendation["surface"]["x"]
y = recommendation["surface"]["y"]
z = recommendation["surface"]["z"]
```

### 2. Current point

```python
current_x = current pressure_axis
current_y = current pressure_rotation
current_z = current predicted_target_speed
```

### 3. Recommended point

```python
recommended_x = recommended pressure_axis
recommended_y = recommended pressure_rotation
recommended_z = recommended predicted_target_speed
```

### 4. Optional arrow

Стрелка:

```text
current point → recommended point
```

Если текущий UI не поддерживает стрелку в 3D, можно провести линию между двумя точками.

---

## Что показывать в UI рядом с графиком

Минимальный набор:

```text
Energy type: medium_high_energy
Current p_ax: ...
Current p_rot: ...
Current speed: ...
Predicted near5 speed: ...
Recommended p_ax: ...
Recommended p_rot: ...
Expected uplift: +2.3%
```

Также полезно:

```text
delta p_ax: -3.2%
delta p_rot: +3.9%
```

---

## Как работает смена энергоёмкости

На каждом шаге смотреть:

```python
current_energy_type = telemetry_row["rock_energy_type_final"]
```

Если:

```python
current_energy_type != previous_energy_type
```

то:

```text
1. UI обновляет заголовок energy type.
2. AdvisoryEngine использует другие ranges из surface_ranges_by_energy_type.json.
3. Новая surface строится автоматически.
```

Отдельно вручную переключать модель не нужно. Модели общие, а energy type передаётся как categorical feature.

---

## Replay mode

На первом этапе реализовать replay/advisory mode.

Источник данных:

```text
united_rock_energy_segment_quantile.csv
```

Принцип:

```text
1. Симулятор читает строки CSV по выбранной well_id.
2. Бур визуально движется по скважине.
3. Графики rotation/speed показывают фактические значения оператора.
4. AdvisoryEngine параллельно показывает:
   - surface;
   - current point;
   - recommended point;
   - predicted uplift.
5. Симулятор НЕ меняет speed и НЕ применяет recommendation к физике.
```

Это важно: на первом этапе recommendation — advisory, а не closed-loop управление.

---

## Closed-loop mode НЕ делать сейчас

Пока не нужно делать режим, где recommended `pressure_axis / pressure_rotation` реально меняют движение бура.

Причина: нужно отдельно решить, кто является источником истины для новых `speed/rotation`:

```text
1. исторический датасет;
2. ML speed_model;
3. физическая модель симулятора;
4. гибрид.
```

Сейчас задача — подключить advisory visualization.

---

## Куда положить артефакты в репозитории

Рекомендуемая структура:

```text
simulator/
    app/
        data/
            advisory_engine.py
        ml_artifacts/
            drilling_advisory_light_penalty_artifacts/
                feature_config.json
                optimizer_config.json
                rotation_model_near5.joblib
                speed_model_near5.joblib
                surface_ranges_by_energy_type.json
```

CSV для replay:

```text
simulator/app/data/united_rock_energy_segment_quantile.csv
```

или:

```text
simulator/data/united_rock_energy_segment_quantile.csv
```

Главное — чтобы путь был явно задан в config.

---

## Acceptance criteria

После подключения должно работать следующее:

1. Симулятор запускается без ошибок.
2. Replay-режим читает `united_rock_energy_segment_quantile.csv`.
3. После накопления минимум 30 точек buffer появляется recommendation.
4. В UI отображается текущий `rock_energy_type_final`.
5. 3D surface строится из ML-моделей, а не из synthetic formula.
6. На surface есть current point.
7. На surface есть recommended point.
8. Отображается predicted uplift percent.
9. При смене `rock_energy_type_final` поверхность перестраивается.
10. Если данных в buffer недостаточно, UI показывает “warming up” или recommendation=None.

---

## Минимальный псевдокод

```python
engine = AdvisoryEngine(
    artifact_dir="simulator/app/ml_artifacts/drilling_advisory_light_penalty_artifacts"
)

for telemetry_row in replay_rows:
    recommendation = engine.update(telemetry_row)

    update_time_series(
        rotation=telemetry_row["rotation"],
        speed=telemetry_row["speed"],
    )

    if recommendation is not None:
        update_3d_surface(
            x=recommendation["surface"]["x"],
            y=recommendation["surface"]["y"],
            z=recommendation["surface"]["z"],
            current_point=recommendation["current"],
            recommended_point=recommendation["recommended"],
            energy_type=recommendation["energy_type"],
            uplift_pct=recommendation["recommended"]["predicted_uplift_pct"],
        )
    else:
        show_status("warming up advisory model")
```

---

## Важное замечание

`plotly_surfaces_html/` — это статические поверхности для демонстрации. Для live-simulator их не нужно использовать как источник истины. В симуляторе поверхность должна строиться динамически на каждом шаге через:

```text
rotation_model_near5.joblib
speed_model_near5.joblib
```

Потому что surface зависит не только от energy type, но и от текущих:

```text
rotation
speed
hardness_score_smooth
history
pressure history
```
