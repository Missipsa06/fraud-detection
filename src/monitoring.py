"""
Génère un rapport de data drift entre les données d'entraînement et de validation.

Usage:
    python -m src.monitoring

Résultat : reports/drift_report.html
"""
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from .data import load_data, split_data
from .config import BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"


def run_monitoring():
    print("Chargement des données...")
    df = load_data()
    X_train, X_val, _, _ = split_data(df)

    print("Génération du rapport de drift...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_train, current_data=X_val)

    REPORTS_DIR.mkdir(exist_ok=True)
    output = REPORTS_DIR / "drift_report.html"
    report.save_html(str(output))
    print(f"Rapport sauvegardé : {output}")
    print("Ouvre le fichier dans ton navigateur pour visualiser le rapport.")


if __name__ == "__main__":
    run_monitoring()
