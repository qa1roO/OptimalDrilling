# OptimalDrilling: актуальная логика репозитория

Документ описывает текущую архитектуру `OptimalDrilling` после интеграции replay/advisory-режима, сглаживания speed-графика и исправления 3D-поверхностей.

Главная рабочая часть проекта находится в:

```text
simulator/
```

Запуск:

```bash
cd simulator
python run.py
```

---

## 1. Назначение проекта

`OptimalDrilling` демонстрирует процесс бурения и работу advisory-системы:

```text
исторические данные бурения
    ->
разметка rock_energy_type_final
    ->
обученные advisory-модели
    ->
desktop-симулятор
    ->
2D-графики rotation/speed + 3D-поверхность режима + рекомендации
```

Симулятор не является точной физической моделью буровой установки. Его задача:

```text
1. визуально показать проходку по слоям породы;
2. проигрывать реальные replay-строки из подготовленного CSV;
3. показывать текущий energy type;
4. рисовать поверхность pressure_axis x pressure_rotation -> speed;
5. показывать current/recommended точки на этой поверхности;
6. показывать expected uplift и дельты давления.
```

---

## 2. Основные папки

```text
OptimalDrilling/
├── datasets/
├── notebooks/
│   ├── rock_energy_segment_quantile_surfaces_only.ipynb
│   ├── train_drilling_advisory_light_penalty_final.ipynb
│   ├── united_rock_energy_segment_quantile.csv
│   ├── plotly_surfaces_html/
│   └── drilling_advisory_light_penalty_artifacts/
├── simulator/
│   ├── app/
│   │   ├── charts/
│   │   ├── data/
│   │   ├── ml_artifacts/
│   │   ├── scene/
│   │   ├── main.py
│   │   └── window.py
│   └── run.py
└── REPOSITORY_LOGIC.md
```

---

## 3. Точка входа

Файл:

```text
simulator/run.py
```

Содержит только запуск Qt-приложения:

```python
from app.main import run

if __name__ == "__main__":
    raise SystemExit(run())
```

Дальше цепочка такая:

```text
run.py
    ->
app.main.run()
    ->
QApplication
    ->
SimulatorMainWindow
    ->
SideViewWidget + Performance3DWidget
```

---

## 4. Главное окно

Файл:

```text
simulator/app/window.py
```

Главное окно состоит из двух панелей:

```text
левая панель  -> SideViewWidget
правая панель -> Performance3DWidget
```

Сигналы:

```python
self.side_view.drilling_sample.connect(
    self.performance_chart.append_drilling_point
)

self.side_view.replay_sample.connect(
    self.performance_chart.append_advisory_telemetry
)

self.side_view.drilling_cycle_started.connect(
    self.performance_chart.on_drilling_cycle_started
)
```

`SideViewWidget` отвечает за сцену бурения и telemetry.  
`Performance3DWidget` отвечает за 2D/3D-графики, advisory engine, поверхности и маркеры.

---

## 5. Левая сцена: `SideViewWidget`

Файл:

```text
simulator/app/scene/side_view_widget.py
```

Виджет:

```text
1. рисует буровую установку и слои;
2. хранит depth_m;
3. обновляет положение долота;
4. читает replay CSV, если он доступен;
5. отправляет telemetry в правый виджет;
6. при завершении цикла бурения испускает drilling_cycle_started.
```

Replay-данные берутся из:

```text
simulator/app/data/united_rock_energy_segment_quantile.csv
```

Минимальный набор колонок replay:

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

---

## 6. Два runtime-режима

### Replay/advisory mode

Основной режим, если replay CSV доступен.

```text
CSV row
    ->
SideViewWidget.replay_sample.emit(row, depth_m)
    ->
Performance3DWidget.append_advisory_telemetry(row, depth_m)
    ->
AdvisoryEngine.update(row)
    ->
recommendation + current/recommended markers
```

В этом режиме 2D-графики показывают фактические `rotation` и `speed` из replay.

### Legacy/fallback mode

Если replay/advisory-данные недоступны, используется старый synthetic поток:

```text
SideViewWidget.drilling_sample
    ->
Performance3DWidget.append_drilling_point
    ->
legacy ClusterProfile / synthetic surface
```

Этот режим нужен, чтобы симулятор не падал без ML-данных.

---

## 7. Правая панель: `Performance3DWidget`

Файл:

```text
simulator/app/charts/performance_3d_widget_stub.py
```

Несмотря на `stub` в имени, это рабочий виджет графиков.

Он содержит:

```text
1. Rotation by depth;
2. Speed by depth;
3. 3D-view: surface + current point + recommended point + line;
4. status label с energy/uplift/delta.
```

Используемые компоненты:

```text
pyqtgraph.PlotWidget
pyqtgraph.opengl.GLViewWidget
GLSurfacePlotItem
GLScatterPlotItem
GLLinePlotItem
GLTextItem
```

---

## 8. 2D-графики

Ось X:

```text
depth, m
```

Графики:

```text
Rotation by depth -> raw replay rotation
Speed by depth    -> smoothed replay speed
```

Важно: replay speed не заменяется predicted speed.  
Для отображения speed используется rolling mean:

```python
self.speed_curve.setData(
    depth_values,
    self._rolling_mean(speeds, self._speed_smoothing_window)
)
```

Это только визуальное сглаживание линии. В telemetry, advisory и status остаются фактические replay-значения.

---

## 9. Notebook-поверхности

Текущая семантика поверхностей: фоновые 3D-поверхности строятся эмпирически по фактическим строкам внутри каждого уровня энергоёмкости. Пространство `pressure_axis x pressure_rotation` разбивается на ячейки, высота задаётся медианной сглаженной скоростью в ячейке, ячейки с малым числом наблюдений считаются ненадёжными, заполняются интерполяцией и затем поверхность сглаживается взвешенным 2D Gaussian-фильтром. Эти поверхности используются только для визуализации; рекомендации и uplift рассчитываются отдельно near_5 LightGBM-моделями.

Финальные поверхности регенерируются в notebook:

```text
notebooks/rock_energy_segment.ipynb
```

Логика notebook:

```text
для каждого rock_energy_type_final:
    взять фактические строки этого energy type
    выбрать surface target из speed_roll_median_30, speed_roll_mean_30, speed_roll_median_12, speed
    построить сетку pressure_axis x pressure_rotation
    посчитать median speed target и count в каждой ячейке
    убрать low-support ячейки, заполнить пропуски интерполяцией
    сгладить поверхность weighted 2D Gaussian filter
    сохранить Plotly surface в HTML
```

Диагностический отчёт по заполненности, clipping и диапазонам сохраняется в:

```text
notebooks/rock_energy_segment_reports/surface_empirical_report.csv
```

HTML-файлы:

```text
notebooks/plotly_surfaces_html/surface_soft_low_energy.html
notebooks/plotly_surfaces_html/surface_medium_low_energy.html
notebooks/plotly_surfaces_html/surface_medium_high_energy.html
notebooks/plotly_surfaces_html/surface_hard_high_energy.html
```

---

## 10. Загрузка notebook-поверхностей в симулятор

Файл:

```text
simulator/app/data/notebook_surfaces.py
```

Он извлекает из Plotly HTML только данные первой surface-trace:

```text
x -> pressure_axis grid
y -> pressure_rotation grid
z -> speed grid
```

Функции:

```python
load_notebook_surface(energy_type, repository_root)
load_notebook_surfaces(repository_root)
```

Важно: симулятор не встраивает HTML и не использует браузер для live-rendering.  
HTML используется как контейнер сохранённых массивов `x/y/z`.

---

## 11. 3D-поверхность в advisory mode

Текущая логика:

```text
rock_energy_type_final
    ->
Performance3DWidget берёт notebook surface для этого energy type
    ->
рисует GLSurfacePlotItem
```

Поверхность:

```text
X = pressure_axis
Y = pressure_rotation
Z = speed
```

Если notebook surface для energy type не найдена, используется fallback:

```text
recommendation["surface"] из AdvisoryEngine
```

Но основной ожидаемый путь сейчас:

```text
notebooks/plotly_surfaces_html/*.html
    ->
notebook_surfaces.py
    ->
Performance3DWidget._notebook_surfaces
    ->
GLSurfacePlotItem
```

---

## 12. Current/recommended точки на 3D-графике

Раньше точки могли “летать” отдельно от поверхности, потому что их `z` нормализовался по predicted speed независимо от surface.

Текущая логика исправлена:

```text
1. pressure_axis / pressure_rotation точки clamp-ятся в диапазон текущей surface;
2. x/y переводятся в координаты сцены через bounds текущей surface;
3. z берётся не из predicted speed напрямую;
4. z вычисляется интерполяцией по текущей surface:
   _surface_scene_z_at(pressure_axis, pressure_rotation)
```

То есть current и recommended точки лежат на той же поверхности, которая отрисована в 3D.

Маркеры:

```text
current marker
recommended marker
line current -> recommended
```

Хранятся отдельно от surface:

```python
self._surface_items = []
self._marker_items = []
```

Surface пересоздаётся только при смене energy type или source.  
Markers обновляются каждый telemetry-step.

---

## 13. AdvisoryEngine

Файл:

```text
simulator/app/data/advisory_engine.py
```

Главный класс:

```python
class AdvisoryEngine:
    def update(self, telemetry_row: dict) -> dict | None:
        ...

    def get_recommendation(self) -> dict | None:
        ...

    def reset(self) -> None:
        ...
```

Он:

```text
1. хранит rolling buffer telemetry rows;
2. строит признаки как при обучении;
3. создаёт candidate grid pressure_axis / pressure_rotation;
4. предсказывает candidate_target_rotation;
5. предсказывает pred_target_speed;
6. считает optimizer_score;
7. выбирает recommended point;
8. возвращает recommendation.
```

Минимальный размер буфера:

```python
min_buffer_size = 30
```

Пока данных меньше, UI показывает warming up.

---

## 14. Advisory artifacts

Путь:

```text
simulator/app/ml_artifacts/drilling_advisory_light_penalty_artifacts/
```

Файлы:

```text
rotation_model_near5.joblib
speed_model_near5.joblib
feature_config.json
optimizer_config.json
surface_ranges_by_energy_type.json
```

`rotation_model_near5.joblib` предсказывает near5 rotation.  
`speed_model_near5.joblib` предсказывает near5 speed.  
`feature_config.json` задаёт порядок признаков.  
`optimizer_config.json` задаёт grid/penalty параметры.  
`surface_ranges_by_energy_type.json` задаёт исторические ranges по energy type.

---

## 15. Формат recommendation

`AdvisoryEngine.update(...)` возвращает:

```python
{
    "energy_type": "medium_high_energy",
    "current": {
        "pressure_axis": ...,
        "pressure_rotation": ...,
        "rotation": ...,
        "speed": ...,
        "predicted_target_speed": ...,
    },
    "recommended": {
        "pressure_axis": ...,
        "pressure_rotation": ...,
        "predicted_target_rotation": ...,
        "predicted_target_speed": ...,
        "optimizer_score": ...,
        "predicted_uplift_pct": ...,
        "delta_pressure_axis_pct": ...,
        "delta_pressure_rotation_pct": ...,
    },
    "surface": {
        "x": ...,
        "y": ...,
        "z": ...,
        "score": ...,
    },
}
```

Важное различие:

```text
recommendation["surface"] используется как fallback surface.
Основная visible surface сейчас берётся из notebook Plotly HTML.
recommendation["recommended"] всё равно остаётся источником recommended controls.
```

---

## 16. Status label

В нормальном advisory-режиме label показывает:

```text
Energy: hard_high_energy | depth=...m |
current p_ax=..., p_rot=..., speed=... m/s |
recommended p_ax=..., p_rot=... |
uplift=...% |
delta p_ax=...% |
delta p_rot=...%
```

Если буфер ещё маленький:

```text
Advisory: warming up buffer
```

Если advisory artifacts недоступны:

```text
Advisory: unavailable, fallback mode
```

---

## 17. Reset-логика

При новом цикле бурения:

```text
SideViewWidget
    ->
drilling_cycle_started
    ->
Performance3DWidget.on_drilling_cycle_started()
    ->
reset_live_mode()
```

Сбрасывается:

```text
live points
depths
surface items
marker items
cached advisory surface
advisory surface bounds
advisory surface scene data
AdvisoryEngine rolling buffer
```

Это нужно, чтобы разные циклы бурения не смешивали history и surface state.

---

## 18. Что происходит при смене energy type

На каждом telemetry-step приходит:

```text
rock_energy_type_final
```

Если energy type изменился:

```text
1. выбирается другая notebook surface;
2. surface items очищаются;
3. рисуется новая surface;
4. markers пересчитываются на новой surface;
5. status label показывает новый energy type.
```

Если energy type не изменился:

```text
surface остаётся той же;
обновляются только current/recommended markers и line.
```

---

## 19. Закрытый контур пока не реализован

Сейчас режим честно является replay/advisory:

```text
модель советует controls,
но не меняет физику симулятора и replay speed.
```

Не выполняется:

```text
recommended pressure_axis / pressure_rotation
    ->
изменение фактической depth/speed анимации
```

Это осознанно: для closed-loop нужно отдельно определить источник истины для новой скорости.

---

## 20. Важные ограничения

При изменениях нельзя ломать:

```text
SideViewWidget.drilling_sample
SideViewWidget.replay_sample
Performance3DWidget.append_drilling_point
Performance3DWidget.append_advisory_telemetry
drilling_cycle_started reset
fallback/synthetic mode
replay speed как фактическое значение
```

Также важно:

```text
1. не переобучать ML-модели внутри симулятора;
2. не менять порядок features без feature_config.json;
3. не заменять replay speed на predicted speed;
4. не использовать Plotly HTML как embedded browser UI;
5. если notebook surface недоступна, gracefully fallback на AdvisoryEngine surface.
```

---

## 21. Краткая логика всего проекта

```text
notebooks/
    ->
готовят energy labels, replay CSV, advisory artifacts, Plotly surfaces

SideViewWidget
    ->
визуально двигает бур и отправляет telemetry

AdvisoryEngine
    ->
считает recommended pressure_axis / pressure_rotation и uplift

notebook_surfaces.py
    ->
загружает pressure_axis x pressure_rotation -> speed surfaces из Plotly HTML

Performance3DWidget
    ->
рисует 2D rotation/speed,
рисует notebook surface,
проецирует current/recommended точки на эту surface,
показывает advisory status
```

Финальный смысл:

```text
OptimalDrilling показывает не только процесс бурения,
но и advisory-систему:
для текущего rock_energy_type_final отображается поверхность speed,
а модель показывает, куда она рекомендовала бы сместить pressure_axis / pressure_rotation
и какой uplift ожидается.
```
