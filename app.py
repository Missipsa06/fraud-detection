"""
Interface Streamlit pour le modèle de détection de fraude.

Prérequis : API FastAPI lancée sur http://localhost:8000
    uvicorn src.api:app --reload

Lancement :
    streamlit run app.py
"""
import io

import httpx
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_predict(payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def api_sample(fraud: bool) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/sample", params={"fraud": str(fraud).lower()}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


def api_health() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def show_result(result: dict):
    score = result["score"]
    fraud = result["fraud"]
    pct = score * 100

    if fraud:
        st.error(f"Fraude detectee  —  score {score:.4f}")
    else:
        st.success(f"Transaction legitime  —  score {score:.4f}")

    st.progress(score, text=f"{pct:.1f}%")


# ---------------------------------------------------------------------------
# Page : Prediction manuelle
# ---------------------------------------------------------------------------

def page_predict():
    st.header("Prediction manuelle")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Charger exemple legitime"):
            data = api_sample(fraud=False)
            if data:
                st.session_state["form_data"] = data
    with col2:
        if st.button("Charger exemple fraude"):
            data = api_sample(fraud=True)
            if data:
                st.session_state["form_data"] = data

    defaults = st.session_state.get("form_data", {})

    with st.form("predict_form"):
        st.subheader("Informations de base")
        c1, c2 = st.columns(2)
        time_val = c1.number_input("Time", value=float(defaults.get("Time", 0.0)), format="%.2f")
        amount_val = c2.number_input("Amount", value=float(defaults.get("Amount", 0.0)), format="%.4f")

        st.subheader("Composantes PCA")
        vals = {}
        cols = st.columns(4)
        for i in range(1, 29):
            key = f"V{i}"
            col_idx = (i - 1) % 4
            vals[key] = cols[col_idx].number_input(
                key,
                value=float(defaults.get(key, 0.0)),
                format="%.6f",
                label_visibility="visible",
            )

        submitted = st.form_submit_button("Analyser", use_container_width=True)

    if submitted:
        payload = {"Time": time_val, "Amount": amount_val, **vals}
        with st.spinner("Analyse en cours..."):
            result = api_predict(payload)
        if result:
            show_result(result)


# ---------------------------------------------------------------------------
# Page : Analyse par lot
# ---------------------------------------------------------------------------

def page_batch():
    st.header("Analyse par lot (CSV)")
    st.markdown(
        "Importe un CSV contenant les colonnes `Time`, `Amount`, `V1`…`V28`. "
        "Chaque ligne sera analysée par le modele."
    )

    uploaded = st.file_uploader("Choisir un fichier CSV", type="csv")
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        return

    st.write(f"{len(df)} transactions chargees. Apercu :")
    st.dataframe(df[FEATURES].head(5), use_container_width=True)

    if st.button("Lancer l'analyse", use_container_width=True):
        results = []
        progress = st.progress(0, text="Analyse en cours...")
        total = len(df)

        for i, row in df[FEATURES].iterrows():
            res = api_predict(row.to_dict())
            if res is None:
                st.error(f"Erreur a la ligne {i}. Arret.")
                break
            results.append(res)
            progress.progress((i + 1) / total, text=f"{i + 1} / {total}")

        progress.empty()

        if results:
            out = df[FEATURES].copy().reset_index(drop=True)
            out["score"] = [r["score"] for r in results]
            out["fraud"] = [r["fraud"] for r in results]

            n_fraud = out["fraud"].sum()
            st.metric("Fraudes detectees", f"{n_fraud} / {len(out)}")

            def color_fraud(val):
                return "background-color: #fdd" if val else "background-color: #dfd"

            st.dataframe(
                out[["Time", "Amount", "score", "fraud"]].style.applymap(
                    color_fraud, subset=["fraud"]
                ),
                use_container_width=True,
            )

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Telecharger les resultats (CSV)",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page : Monitoring
# ---------------------------------------------------------------------------

def page_monitoring():
    st.header("Monitoring — Data Drift")
    st.markdown(
        "Calcule le drift entre les 80% premieres transactions (reference) "
        "et les 20% les plus recentes (production simulee) via le **test KS**."
    )

    if st.button("Lancer le rapport de drift", use_container_width=True):
        with st.spinner("Analyse du drift..."):
            try:
                from scipy.stats import ks_2samp
                from src.data import load_data

                DRIFT_THRESHOLD = 0.05
                MIN_KS_STAT = 0.1

                df = load_data().sort_values("Time").reset_index(drop=True)
                split_idx = int(len(df) * 0.8)
                reference = df.iloc[:split_idx].drop(columns=["Class"])
                current = df.iloc[split_idx:].drop(columns=["Class"])

                rows = []
                for col in reference.columns:
                    stat, p_value = ks_2samp(reference[col], current[col])
                    drift = p_value < DRIFT_THRESHOLD and stat >= MIN_KS_STAT
                    rows.append(
                        {"Feature": col, "KS stat": round(stat, 4), "p-value": round(p_value, 4), "Drift": drift}
                    )

                report = pd.DataFrame(rows).sort_values("p-value")
                n_drift = report["Drift"].sum()
                st.metric("Features en drift", f"{n_drift} / {len(report)}")

                def color_drift(val):
                    return "background-color: #fdd" if val else "background-color: #dfd"

                st.dataframe(
                    report.style.applymap(color_drift, subset=["Drift"]),
                    use_container_width=True,
                    height=600,
                )

            except Exception as e:
                st.error(f"Erreur lors du monitoring : {e}")
                st.info("Assurez-vous que le dataset creditcard.csv est present.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection",
    page_icon=":shield:",
    layout="wide",
)

st.title("Fraud Detection Dashboard")

# Indicateur de statut API
healthy = api_health()
if healthy:
    st.sidebar.success("API connectee")
else:
    st.sidebar.error("API hors ligne — lance : uvicorn src.api:app --reload")

page = st.sidebar.radio(
    "Navigation",
    ["Prediction manuelle", "Analyse par lot", "Monitoring"],
)

if page == "Prediction manuelle":
    page_predict()
elif page == "Analyse par lot":
    page_batch()
else:
    page_monitoring()
