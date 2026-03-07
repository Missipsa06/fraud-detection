"""
Génère un rapport de data drift entre les données historiques (référence)
et les transactions récentes (simulation de production).

La séparation est faite par ordre chronologique (colonne Time) :
  - référence : les 80% premières transactions (les plus anciennes)
  - current   : les 20% dernières transactions (les plus récentes)

Méthode : test de Kolmogorov-Smirnov (KS) par feature.
Une feature est considérée en drift si p-value < 0.05.

Usage:
    python -m src.monitoring

Résultat : reports/drift_report.html
"""
from scipy.stats import ks_2samp

from .data import load_data
from .config import BASE_DIR

REPORTS_DIR = BASE_DIR / "reports"
REFERENCE_RATIO = 0.8
DRIFT_THRESHOLD = 0.05   # p-value max
MIN_KS_STAT    = 0.1    # effet minimum (évite les faux positifs sur grands datasets)


def run_monitoring():
    print("Chargement des données...")
    df = load_data().sort_values("Time").reset_index(drop=True)

    split_idx = int(len(df) * REFERENCE_RATIO)
    reference = df.iloc[:split_idx].drop(columns=["Class"])
    current   = df.iloc[split_idx:].drop(columns=["Class"])

    print(f"Référence : {len(reference):,} transactions  |  Current : {len(current):,} transactions\n")

    results = []
    for col in reference.columns:
        stat, p_value = ks_2samp(reference[col], current[col])
        drift = p_value < DRIFT_THRESHOLD and stat >= MIN_KS_STAT
        results.append({"feature": col, "ks_stat": stat, "p_value": p_value, "drift": drift})

    drifted = [r for r in results if r["drift"]]
    stable  = [r for r in results if not r["drift"]]

    print(f"{'Feature':<20} {'KS stat':>8} {'p-value':>10} {'Drift':>6}")
    print("-" * 48)
    for r in sorted(results, key=lambda x: x["p_value"]):
        flag = "YES" if r["drift"] else "no"
        print(f"{r['feature']:<20} {r['ks_stat']:>8.4f} {r['p_value']:>10.4f} {flag:>6}")

    print(f"\nFeatures en drift : {len(drifted)} / {len(results)}")

    REPORTS_DIR.mkdir(exist_ok=True)
    output = REPORTS_DIR / "drift_report.html"
    _save_html(results, output)
    print(f"Rapport sauvegardé : {output}")


def _save_html(results: list, output):
    rows = ""
    for r in sorted(results, key=lambda x: x["p_value"]):
        color = "#fdd" if r["drift"] else "#dfd"
        flag = "YES" if r["drift"] else "no"
        rows += (
            f'<tr style="background:{color}">'
            f'<td>{r["feature"]}</td>'
            f'<td>{r["ks_stat"]:.4f}</td>'
            f'<td>{r["p_value"]:.4f}</td>'
            f'<td><b>{flag}</b></td>'
            f'</tr>\n'
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Drift Report</title>
<style>
  body {{ font-family: sans-serif; padding: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: right; }}
  th {{ background: #eee; text-align: left; }}
  td:first-child {{ text-align: left; }}
</style>
</head><body>
<h2>Data Drift Report (KS test, p &lt; {DRIFT_THRESHOLD})</h2>
<table>
<tr><th>Feature</th><th>KS stat</th><th>p-value</th><th>Drift</th></tr>
{rows}
</table>
</body></html>"""
    output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    run_monitoring()
