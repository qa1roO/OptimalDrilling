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

`SideViewWidget` рисует боковой вид буровой установки через
`pyqtgraph.GraphicsLayoutWidget` и `QGraphics*Item`.

На сцене есть:

- буровая установка;
- гусеничное основание;
- кабина;
- мачта;
- каретка;
- вращатель / мотор;
- бурильная труба;
- шарошка / долото;
- скважина;
- слои породы;
- индикатор глубины;
- текстовые параметры;
- локальные лейблы около рабочих органов.

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
    cluster_id: int
```

Регионы:

```text
Europe Basin
Far East Volcanic
Brazil Shield
```

Каждый регион содержит набор пород. Для каждого слоя случайно задается толщина
и случайный `cluster_id` из диапазона `0..6`.

`cluster_id` связывает визуальный слой с model-driven логикой графиков.

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
cluster_id
```

Emit находится в `_on_tick()`:

```python
self.drilling_sample.emit(
    self.elapsed_time_s,
    self.rot_speed_rpm,
    current_speed,
    self.depth_m,
    self._current_layer_cluster_id(),
)
```

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
append_drilling_point(time_s, rotation, speed, depth_m, cluster_id)
```

Если `cluster_id` изменился, запускается новый сегмент:

```python
_start_layer_segment(...)
```

Если кластер тот же, точка продолжает текущий сегмент.

Live-точки хранятся в:

```python
self._live_points
self._live_depths_m
self._surface_segment_points
```

## Model-driven логика графиков

График не просто рисует сырые значения из сцены. При получении `cluster_id` он
берет профиль кластера:

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
```

В текущем `main` `OPTIMALS_PATH` все еще указывает на абсолютный локальный путь:

```python
OPTIMALS_PATH = Path(r"C:\Users\stas2\Downloads\optimals.csv")
```

Это потенциальная проблема переносимости. Если файла нет у другого пользователя,
код не падает сразу, но использует fallback: ищет оптимумы по датасетам, выбирая
максимальную speed внутри кластера. Из-за этого графики у разных пользователей
могут отличаться.

## Модельные артефакты

В текущем `main` внутри репозитория есть:

```text
simulator/simulator_core/kmeans_k7.joblib
simulator/simulator_core/scaler_k7.joblib
```

`optimals.csv` в текущем `main` не лежит рядом с ними, хотя логически должен
быть там же.

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
    -> создает слои с cluster_id

SideViewWidget
    -> определяет текущий слой по depth_m
    -> считает current_speed, rotation, axial
    -> двигает бур и трубу
    -> emit(time_s, rotation, speed, depth_m, cluster_id)

Performance3DWidget
    -> принимает live-сигнал
    -> при смене cluster_id берет ClusterProfile
    -> строит плавный переход к optimal_rotation / target_live_speed
    -> обновляет 2D depth-графики и 3D поверхность

drilling_series.py
    -> читает датасеты
    -> грузит scaler/kmeans
    -> собирает avg/optimal профили кластеров
```

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

## Известный риск текущего main

В текущем состоянии репозитория есть риск несовпадения поведения у разных
пользователей из-за:

```python
OPTIMALS_PATH = Path(r"C:\Users\stas2\Downloads\optimals.csv")
```

Если цель - одинаковый запуск после `git pull`, нужно перенести `optimals.csv`
в `simulator/simulator_core` и заменить путь на:

```python
OPTIMALS_PATH = SIMULATOR_CORE_DIR / "optimals.csv"
```

Такой фикс ранее был подготовлен в отдельной ветке `simulator-core-optimals`, но
в текущую `main` он не влит.
