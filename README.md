# OptimalDrilling

Проект для подготовки телеметрии бурения, расчета энергоемкости пород, обучения near5 рекомендательной модели и демонстрации результата в desktop-симуляторе.

Финальная логика проекта:

- предобработка исходных `datasets/dataset_1.csv` и `datasets/dataset_2.csv` в `datasets/united.csv`;
- расчет непрерывного proxy-показателя энергоемкости и 4 квантильных уровней: `soft_low_energy`, `medium_low_energy`, `medium_high_energy`, `hard_high_energy`;
- формирование признаков, лагов, скользящих статистик и изменений управляющих параметров;
- двухэтапная LightGBM-схема near5: прогноз ближайшей частоты вращения, затем прогноз ближайшей скорости проходки с прогнозной частотой как дополнительным признаком;
- offline-оценка рекомендаций на исторических данных. Это расчетная проверка, а не online A/B-тест;
- симулятор replay/advisory, где белая траектория оператора является визуальным слоем, а рекомендации считаются по финальным моделям.

## Структура

- `notebooks/preprocess.ipynb` - собирает и очищает исходные CSV, сохраняет `datasets/united.csv`.
- `notebooks/rock_energy_segment.ipynb` - рассчитывает энергоемкость, квантильные уровни и глобальную JSON-поверхность скорости для визуализации.
- `notebooks/train_drilling_advisory_lp.ipynb` - обучает near5 LightGBM-модели, строит candidate grid, считает score и offline-отчеты.
- `notebooks/model_artifacts/` - финальные модели, конфиги признаков/оптимизатора и machine-readable артефакты обучения.
- `notebooks/model_reports/` - финальные CSV-отчеты для анализа и дипломных таблиц.
- `simulator/` - desktop-приложение для replay/advisory-демонстрации.
- `simulator/app/ml_artifacts/` - копия финальных моделей и конфигов, которые загружает симулятор.
- `archive/` - устаревшие или служебные файлы, не участвующие в текущем воспроизведении пайплайна.

## Порядок воспроизведения

Установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустить ноутбуки сверху вниз в таком порядке:

```text
notebooks/preprocess.ipynb
notebooks/rock_energy_segment.ipynb
notebooks/train_drilling_advisory_lp.ipynb
```

Ключевые создаваемые артефакты:

- `datasets/united.csv`;
- `notebooks/telemetry_with_energy_quantiles.csv`;
- `notebooks/surface_data/global_speed_surface.json`;
- `notebooks/model_artifacts/rotation_model_near5.joblib`;
- `notebooks/model_artifacts/speed_model_near5.joblib`;
- `notebooks/model_artifacts/feature_config.json`;
- `notebooks/model_artifacts/optimizer_config.json`;
- `notebooks/model_reports/*.csv`.

Перед запуском симулятора убедитесь, что `simulator/app/ml_artifacts/model_artifacts/` содержит актуальные `joblib`-модели и JSON-конфиги.

## Симулятор

```bash
cd simulator
python3 run.py
```

Симулятор читает replay-данные из `notebooks/telemetry_with_energy_quantiles.csv`, загружает near5-модели из `simulator/app/ml_artifacts/` и использует `notebooks/surface_data/global_speed_surface.json` как визуальный фон 3D-поверхности. Эта поверхность не является отдельной рекомендательной моделью.
