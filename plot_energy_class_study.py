"""Create static research figures after study_energy_classes.py has completed."""
from pathlib import Path
import os

from config import DRILLING_ADVISORY_REPORT_DIR

OUT = DRILLING_ADVISORY_REPORT_DIR/"energy_class_study"
os.environ.setdefault("MPLCONFIGDIR", str(OUT/"cache"/"matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    prediction = pd.read_csv(OUT/"prediction_summary.csv")
    policy = pd.read_csv(OUT/"policy_summary.csv")
    density = pd.read_csv(OUT/"density_curves.csv")
    stability = pd.read_csv(OUT/"class_stability.csv")
    profiles = pd.read_csv(OUT/"profiles_q4.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(density.log_index, density['kde_1.0'], color='#245b9b', label='Плотность индекса')
    import json
    cuts = json.loads((OUT/'scheme_q4_development_only.json').read_text())['cuts_log_index']
    for i, cut in enumerate(cuts):
        ax.axvline(cut, color='#777777', linestyle='--', linewidth=.9, label='Квартили' if i==0 else None)
    ax.set(xlabel='Сглаженный log(1 + proxy E)', ylabel='Плотность', title='Распределение и четыре квартильных интервала')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    names = prediction.variant.tolist()
    labels = {'no_E':'Без E', 'continuous_E':'Непрерывный E'}
    display = [labels.get(n, f'K={n[1:]}') for n in names]
    y = prediction.RMSE_delta_vs_q4_pct.to_numpy()
    lo = prediction.delta_vs_q4_family_ci_low.to_numpy()
    hi = prediction.delta_vs_q4_family_ci_high.to_numpy()
    ax.errorbar(np.arange(len(names)), y, yerr=[np.maximum(0,y-lo), np.maximum(0,hi-y)], fmt='o', color='#245b9b', capsize=3)
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.set_xticks(np.arange(len(names)), display, rotation=40, ha='right')
    ax.set(ylabel='Изменение RMSE относительно K=4, %', title='Прогноз: ниже лучше\nИнтервалы с поправкой на сравнения')
    ax = axes[1, 0]
    order = ['continuous_E']+[f'q{k}' for k in range(2,9)]
    p = policy.set_index('policy').loc[order]
    y = p.independent_time_saved_pct.to_numpy()
    ax.errorbar(np.arange(len(order)), y, yerr=[y-p.time_ci025.to_numpy(), p.time_ci975.to_numpy()-y], fmt='o', color='#278369', capsize=3)
    ax.set_xticks(np.arange(len(order)), ['Только E']+[str(k) for k in range(2,9)], rotation=30, ha='right')
    ax.set(xlabel='Число классов', ylabel='Модельное сокращение времени, %', title='Независимый оценщик; 3 интервала на скважину')
    ax = axes[1, 1]
    ax.plot(stability.classes, 100*stability.bootstrap_median_reassigned_fraction, 'o-', color='#95613a')
    ax.set(xlabel='Число классов', ylabel='Сменили класс при bootstrap границ, %', title='Устойчивость границ при перевыборке скважин')
    ax.set_xticks(stability.classes)
    fig.savefig(OUT/'study_overview.png', dpi=180)
    fig.savefig(OUT/'study_overview.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    fields = [('median_speed_mh','Медианная скорость, м/ч'), ('median_pressure_axis','Медианное осевое давление\n(ед. телеметрии)'),
              ('median_pressure_rotation','Медианное давление вращения\n(ед. телеметрии)')]
    for ax, (column, label) in zip(axes, fields):
        ax.bar(profiles.rock_energy_type_final, profiles[column], color='#245b9b', width=.65)
        ax.set_ylabel(label)
        ax.set_xlabel('Относительный уровень индекса E')
        ax.set_ylim(0, profiles[column].max()*1.18)
        for i, value in enumerate(profiles[column]):
            ax.text(i, value, f'{value:.1f}', ha='center', va='bottom')
    fig.savefig(OUT/'four_class_profiles.png', dpi=180)
    plt.close(fig)
    print('Saved study_overview.png, study_overview.svg and four_class_profiles.png')


if __name__ == '__main__':
    run()
