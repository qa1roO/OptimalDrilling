# OptimalDrilling Simulator Overview

Этот документ описывает только desktop-симулятор из папки `simulator/`.
Его нужно обновлять при изменениях в архитектуре, сигналах, графиках,
визуальной сцене или путях к данным.

## Назначение

`OptimalDrilling/simulator` - desktop-приложение на Python для наглядной
демонстрации бурения скважины. Симулятор не является точной физической моделью
буровой установки. Его задача - показать прохождение слоев породы, изменение
параметров бурения и реакцию model-driven логики на графиках.

Основной стек:

- Python
- PySide6
- pyqtgraph
- pyqtgraph.opengl
- numpy
- pandas
- scikit-learn
- joblib

Точка входа:

```text
simulator/run.py
```

`run.py` импортирует `run()` из `app.main`. В `app/main.py` создается
`QApplication`, окно `SimulatorMainWindow`, затем окно открывается через
`showMaximized()`.

## Архитектура окна

Главное окно находится в:

```text
simulator/app/window.py
```

Окно состоит из двух панелей:

```text
левая панель  -> SideViewWidget
правая панель -> Performance3DWidget
```

В `window.py` создаются оба виджета и связываются сигналами:

```python
self.side_view.drilling_sample.connect(self.performance_chart.append_drilling_point)
self.side_view.drilling_cycle_started.connect(self.performance_chart.on_drilling_cycle_started)
```

Левая сцена управляет live-процессом бурения, правая панель строит графики по
сигналам из левой сцены.

## Левая часть: SideViewWidget

Основной файл:

```text
simulator/app/scene/side_view_widget.py
```

`SideViewWidget` отвечает за левую live-сцену симулятора: хранит текущую
глубину, определяет активный слой, двигает буровой инструмент и отправляет
данные в правую панель графиков.

Геометрия сцены задается константами в начале файла, например:

```python
SCENE_W = 900.0
SCENE_H = 1220.0
MAST_X = 310.0
SECTION_TOP = TRACK_Y + TRACK_H
MAX_DEPTH_M = 40.0
TIMER_INTERVAL_MS = 40
```

Текущая глубина бурения хранится в:

```python
self.depth_m
```

Визуальная позиция долота вычисляется через:

```python
_depth_to_section_y(depth_m)
_bit_y()
_bit_tip_y()
```

Каретка движется по мачте через:

```python
_car_y()
```

Обновление кинематики происходит в:

```python
_sync()
```

В `_sync()` синхронизируются:

- позиция каретки;
- вал мотора;
- бурильная труба;
- пробуренная скважина;
- долото;
- индикатор глубины;
- локальные подписи `rotation`, `axial`, `speed`.

## Геология и слои

Слои генерируются в:

```text
simulator/app/scene/rock_layer_generator_stub.py
```

Структура слоя:

```python
@dataclass(frozen=True)
class RockLayer:
    name: str
    thickness: float
    color_hex: str
    energy_type: str
```

Регионы:

```text
Europe Basin
Far East Volcanic
Brazil Shield
```

Каждый регион содержит набор пород. Для каждого слоя случайно задается толщина,
а `energy_type` определяется по типу породы.

Используемые energy sections:

```text
soft_low_energy
medium_low_energy
medium_high_energy
hard_high_energy
```

Визуальная геология больше не генерирует старые случайные `cluster_id 0..6`.
Эти id относятся к legacy kmeans-логике и не должны использоваться как основная
классификация в replay/advisory режиме.

## Скорость бурения в сцене

Скорость проходки зависит от текущей породы:

```python
ROCK_DRILL_SPEED_MPS = {
    "TopSoil": 0.030,
    "Clay": 0.022,
    "Sandstone": 0.014,
    "Shale": 0.011,
    "Limestone": 0.009,
    "Granite": 0.005,
    "Basalt": 0.004,
}
```

Текущая скорость вычисляется в:

```python
_current_drilling_speed()
```

Формула использует базовую скорость слоя и небольшую синусоидальную вариацию:

```python
base_speed * (1.0 + 0.08 * sin(...))
```

Глубина увеличивается на каждом тике:

```python
depth_step = current_speed * DRILLING_SPEED_TO_DEPTH_STEP
self.depth_m += depth_step
```

## Live-сигнал из сцены

На каждом тике, пока бурение идет вниз, `SideViewWidget` испускает сигнал:

```python
drilling_sample = Signal(float, float, float, float, int)
```

Порядок данных:

```text
time_s
rotation
speed
depth_m
energy_type_id
```

Emit находится в `_on_tick()`:

```python
self.drilling_sample.emit(
    self.elapsed_time_s,
    self.rot_speed_rpm,
    current_speed,
    self.depth_m,
    self._current_layer_energy_type_id(),
)
```

`energy_type_id` нужен только для совместимости старого fallback-потока
`append_drilling_point`. Основной replay/advisory поток использует строковый
`rock_energy_type_final`.

При завершении цикла бурения и генерации новой геологии испускается:

```python
drilling_cycle_started
```

График использует этот сигнал для reset.

## Правая часть: Performance3DWidget

Основной файл:

```text
simulator/app/charts/performance_3d_widget_stub.py
```

Несмотря на название `stub`, это рабочий live-виджет графиков.

Виджет содержит:

- верхний 2D-график `Rotation by depth`;
- нижний 2D-график `Speed by depth`;
- правый 3D-график с условной поверхностью кластера.

2D-графики строятся через `pyqtgraph.PlotWidget`.

3D-график строится через:

```python
pyqtgraph.opengl.GLViewWidget
GLAxisItem
GLGridItem
GLSurfacePlotItem
GLLinePlotItem
GLScatterPlotItem
GLTextItem
```

Ось X у 2D-графиков - глубина:

```python
depth_axis_max_m = 42.0
plot.setLabel("bottom", "depth", units="m")
plot.setXRange(0.0, self.depth_axis_max_m)
```

## Прием live-данных графиком

Точка входа для данных:

```python
append_drilling_point(time_s, rotation, speed, depth_m, energy_type_id)
```

В legacy fallback-режиме при смене `energy_type_id` запускается новый сегмент:

```python
_start_layer_segment(...)
```

Если energy section та же, точка продолжает текущий сегмент.

Live-точки хранятся в:

```python
self._live_points
self._live_depths_m
self._surface_segment_points
```

## Model-driven логика графиков

В legacy fallback-режиме график не просто рисует сырые значения из сцены. При
получении числового `energy_type_id` он может использовать старый `ClusterProfile`:

```python
load_cluster_profiles()
```

Профиль содержит:

```python
avg_rotation
avg_speed
optimal_rotation
optimal_speed
```

Для первого слоя стартовый `rotation` фиксирован:

```python
_initial_rotation_rpm = 100.0
```

Для следующих слоев стартовая точка берется из конца предыдущего сегмента,
чтобы траектория была непрерывной.

## Плавность графиков

Плавность делается в:

```python
_interpolated_profile_point()
```

Основная идея:

```python
k = пройденная_глубина_в_сегменте / transition_depth
smooth_k = k * k * (3.0 - 2.0 * k)
```

Это smoothstep-интерполяция. Она дает мягкий старт, плавное движение и мягкое
приближение к оптимуму.

Для `speed` используется:

```python
_transition_depth_m = 3.0
```

Для `rotation` есть отдельная глубина восстановления:

```python
_active_rotation_transition_depth_m
```

Она может увеличиваться, если в предыдущем кластере был сильный провал rotation:

```python
_previous_rotation_drop_rpm
_rotation_growth_base_slowdown = 1.35
_rotation_drop_slowdown_per_10rpm = 0.28
```

Если rotation раньше просел, следующий рост растягивается по большей глубине и
выглядит более реалистично.

## Логика speed

Speed на графике приводится к live-масштабу симулятора. Это важно, потому что
значения скорости из датасета могут быть в другом масштабе.

Целевой live-speed считается в:

```python
_target_live_speed()
```

Используется отношение:

```python
optimal_speed / avg_speed
```

Коэффициент ограничивается:

```python
1.08 .. 1.35
```

И применяется к стартовой live-скорости. Итог также ограничивается:

```python
0.003 .. 0.035
```

Это сделано, чтобы график speed после резкой просадки в новом кластере плавно
выходил к улучшенному режиму, а не падал к странному датасетному значению.

## 3D-график

3D-график сейчас строит условную поверхность кластера, а не настоящую
модельную плоскость из ML-пайплайна.

При смене кластера вызывается:

```python
_draw_cluster_surface()
```

Поверхность создается синтетически:

```python
z = surface_height * (...)
z += sin(...)
z += cos(...)
```

На поверхности есть:

- красная target-точка;
- траектория текущего сегмента;
- точки live-траектории;
- подписи `rotation`, `speed`, `surface`;
- сетка и оси.

Координаты live-точек переводятся в координаты 3D-сцены через:

```python
_rotation_to_surface_x()
_speed_to_surface_y()
_surface_z()
```

## Data-слой

Основной файл:

```text
simulator/app/data/drilling_series.py
```

Он отвечает за:

- чтение `dataset_1.csv` и `dataset_2.csv`;
- загрузку `scaler` и `kmeans`;
- предсказание кластеров;
- подсчет средних значений по кластерам;
- поиск оптимальных значений по кластерам;
- подготовку `ClusterProfile`.

Ключевые структуры:

```python
DrillingPoint
ClusterProfile
```

Пути в текущем состоянии репозитория:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_CORE_DIR = SIMULATOR_ROOT / "simulator_core"

DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_PATHS = (
    DATASETS_DIR / "dataset_1.csv",
    DATASETS_DIR / "dataset_2.csv",
)

KMEANS_PATH = SIMULATOR_CORE_DIR / "kmeans_k7.joblib"
SCALER_PATH = SIMULATOR_CORE_DIR / "scaler_k7.joblib"
OPTIMALS_PATH = SIMULATOR_CORE_DIR / "optimals.csv"
```

Все основные артефакты, влияющие на логику графиков, хранятся внутри
репозитория и доступны после `git pull`.

## Модельные артефакты

В текущем `main` внутри репозитория есть:

```text
simulator/simulator_core/kmeans_k7.joblib
simulator/simulator_core/scaler_k7.joblib
simulator/simulator_core/optimals.csv
```

`optimals.csv` хранится рядом с `kmeans` и `scaler`, чтобы симулятор не зависел
от локальных файлов конкретного разработчика.

## Датасеты

Используются:

```text
datasets/dataset_1.csv
datasets/dataset_2.csv
```

Фичи кластеризации:

```python
FEATURE_COLUMNS = (
    "pressure_axis",
    "pressure_rotation",
    "rotation",
    "speed",
)
```

Кластеризация:

```text
features -> scaler.transform(features) -> kmeans.predict(...)
```

`load_cluster_profiles()` собирает профиль каждого кластера:

```text
average rotation/speed
optimal rotation/speed
```

Если есть `optimals.csv`, код читает `opt_pax` / `opt_prot`, затем ищет
ближайшую строку в датасетах и берет из нее `rotation` и `speed`.

Если `optimals.csv` нет, оптимум выбирается как строка с максимальной `speed` в
каждом кластере.

## Reset-логика

Когда бур достигает целевой глубины, он начинает обратный ход наверх. После
возврата к нулевой глубине генерируется новая геология:

```python
self.region, self.layers = generate_rock_layers()
```

Затем вызывается:

```python
self.drilling_cycle_started.emit()
```

Правый график принимает этот сигнал и вызывает:

```python
reset_live_mode()
```

При reset очищаются live-точки, surface-items, target-lines и графики. Оси и
подписи сцены остаются.

## Текущий UI

Окно темное, две панели:

```text
Rig Side View        слева
Performance Dashboard справа
```

В `window.py` задан общий stylesheet для темного интерфейса. Окно открывается
на весь экран через `showMaximized()`.

## Главная цепочка данных

```text
rock_layer_generator_stub.py
    -> создает визуальные слои с energy_type

SideViewWidget
    -> определяет текущий слой по depth_m
    -> считает current_speed, rotation, axial
    -> двигает бур и трубу
    -> emit replay_sample(telemetry_row, depth_m) в replay/advisory mode
    -> emit drilling_sample(..., energy_type_id) только в fallback mode

Performance3DWidget
    -> принимает replay/advisory telemetry
    -> строит 2D depth-графики по фактическим rotation/speed
    -> строит 3D ML future-speed surface
    -> показывает current point, recommended point и uplift

drilling_series.py
    -> читает датасеты
    -> грузит scaler/kmeans
    -> собирает avg/optimal профили кластеров
```

## Replay / Advisory mode

Текущий симулятор дополнен replay/advisory режимом. В этом режиме левая сцена
использует историческую telemetry из:

```text
simulator/app/data/united_rock_energy_segment_quantile.csv
```

`SideViewWidget` читает первый валидный contiguous `well_id` из CSV и не
сканирует весь файл целиком при старте. Это важно, потому что replay CSV большой.
На каждом тике сцена берет следующую строку telemetry, двигает бур по фактической
`speed`, показывает фактический `rotation` и отправляет в графики новый сигнал:

```python
replay_sample = Signal(dict, float)
```

Порядок данных:

```text
telemetry_row
depth_m
```

Обычный synthetic-сигнал `drilling_sample` сохранен как fallback, если replay CSV
недоступен.

## AdvisoryEngine

Новый data-layer находится в:

```text
simulator/app/data/advisory_engine.py
```

Основной класс:

```python
class AdvisoryEngine:
    def update(self, telemetry_row: dict) -> dict | None:
        ...

    def get_recommendation(self) -> dict | None:
        ...
```

Engine хранит rolling buffer telemetry, повторяет feature engineering из
training pipeline, строит candidate grid по `pressure_axis` и
`pressure_rotation`, затем использует две модели:

```text
rotation_model_near5.joblib
speed_model_near5.joblib
```

После накопления минимум 30 telemetry points engine возвращает recommendation:

```text
energy_type
current point
recommended point
predicted uplift
future-speed surface
optimizer score surface
```

Если данных меньше 30, UI показывает warming-up status.

## Advisory artifacts

Артефакты advisory-модели лежат внутри simulator:

```text
simulator/app/ml_artifacts/drilling_advisory_light_penalty_artifacts/
    feature_config.json
    optimizer_config.json
    surface_ranges_by_energy_type.json
    rotation_model_near5.joblib
    speed_model_near5.joblib
```

Для загрузки этих моделей требуется `lightgbm`, поэтому он добавлен в:

```text
simulator/requirements.txt
```

Если `lightgbm` или артефакты отсутствуют, симулятор не должен падать: правая
панель показывает статус, что advisory model unavailable.

Даже когда advisory-модель недоступна или buffer еще прогревается, 3D-панель
не остается пустой: она строит preview-плоскость текущего `energy_type` по
диапазонам из `surface_ranges_by_energy_type.json` и показывает текущую
операторскую точку. После появления полноценной ML-рекомендации preview
заменяется ML future-speed surface.

## Advisory 3D surface

В advisory mode старая synthetic 3D surface заменяется ML-поверхностью:

```text
X = pressure_axis
Y = pressure_rotation
Z = predicted target_speed_near5
```

На 3D-графике отображаются:

- ML future-speed surface;
- текущая точка оператора;
- рекомендованная точка;
- линия от текущей точки к рекомендованной;
- predicted uplift percent в status label.

В warm-up/fallback состоянии отображаются:

- preview-плоскость текущего `energy_type`;
- текущая точка оператора, которая обновляется по мере бурения.

2D-графики в этом режиме показывают фактические replay значения:

```text
Rotation by depth -> фактический rotation из CSV
Speed by depth    -> фактический speed из CSV
```

Красные target-lines на 2D-графиках показывают текущие recommended
`predicted_target_rotation` и `predicted_target_speed`.

## Ограничения при изменениях

Анализировать и менять в первую очередь только папку:

```text
simulator/
```

ML-артефакты, датасеты и обучение модели лучше считать read-only. Симулятор
использует готовые `scaler`, `kmeans`, датасеты и оптимумы, но не должен
переобучать модель.

При изменениях важно не ломать:

- сигнал `SideViewWidget.drilling_sample`;
- сигнатуру `Performance3DWidget.append_drilling_point`;
- reset графика при `drilling_cycle_started`;
- масштаб оси глубины `0..42 м`;
- live-непрерывность между слоями;
- локальные подписи на буровой сцене.
