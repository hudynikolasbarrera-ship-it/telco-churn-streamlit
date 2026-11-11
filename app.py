"""
App de Streamlit para predecir el abandono de clientes (Telco Customer Churn).

Uso:
    streamlit run app.py

Carga model.joblib (generado por train.py) y NO vuelve a entrenar.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Etiquetas en español para la interfaz (el modelo recibe los valores originales)
FEATURE_LABELS = {
    "tenure": "Meses como cliente",
    "MonthlyCharges": "Cargo mensual (USD)",
    "Contract": "Tipo de contrato",
    "InternetService": "Servicio de internet",
    "PaymentMethod": "Método de pago",
    "OnlineSecurity": "Seguridad en línea",
    "TechSupport": "Soporte técnico",
    "PaperlessBilling": "Factura electrónica",
    "SeniorCitizen": "Adulto mayor (65+)",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene personas a cargo",
}

VALUE_LABELS = {
    "Yes": "Sí",
    "No": "No",
    "No internet service": "Sin servicio de internet",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "DSL": "DSL",
    "Fiber optic": "Fibra óptica",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)",
}

# Variables que dependen de tener internet
INTERNET_DEPENDENT = ["OnlineSecurity", "TechSupport"]


def label(value: str) -> str:
    return VALUE_LABELS.get(value, value)


@st.cache_resource
def load_artifact():
    """Carga el modelo una sola vez (no reentrena)."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Predicción de abandono", page_icon="📡", layout="centered")

artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
category_options = artifact["category_options"]
numeric_ranges = artifact["numeric_ranges"]
metrics = artifact.get("metrics", {})

# ----------------------------------------------------------------------
# Panel lateral: información del modelo
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.write(f"**Modelo:** {artifact.get('model_name', 'Clasificador')}")
    st.write("**Dataset:** Telco Customer Churn (IBM), 7.043 clientes.")
    st.write("**Objetivo:** predecir si un cliente abandonará el servicio.")
    if metrics:
        st.subheader("Desempeño en prueba")
        st.metric("Recall (abandona)", metrics.get("recall"))
        st.metric("F1 (abandona)", metrics.get("f1"))
        st.metric("ROC-AUC", metrics.get("roc_auc"))
    st.caption("Recall: proporción de clientes que abandonan y el modelo detecta.")

# ----------------------------------------------------------------------
# Formulario
# ----------------------------------------------------------------------
st.title("📡 Predicción de abandono de clientes")
st.write(
    "Ingrese las características de un cliente de telecomunicaciones para estimar "
    "la probabilidad de que **cancele el servicio**."
)

values = {}

st.subheader("Cuenta y facturación")
col1, col2 = st.columns(2)
with col1:
    r = numeric_ranges["tenure"]
    values["tenure"] = st.slider(
        FEATURE_LABELS["tenure"],
        min_value=int(r["min"]), max_value=int(r["max"]), value=int(r["median"]),
        help="Cantidad de meses que el cliente lleva en la empresa.",
    )
with col2:
    r = numeric_ranges["MonthlyCharges"]
    values["MonthlyCharges"] = st.number_input(
        FEATURE_LABELS["MonthlyCharges"],
        min_value=float(r["min"]), max_value=float(r["max"]),
        value=float(r["median"]), step=0.5,
        help="Valor que paga el cliente cada mes.",
    )

col1, col2 = st.columns(2)
with col1:
    values["Contract"] = st.selectbox(
        FEATURE_LABELS["Contract"], category_options["Contract"], format_func=label
    )
    values["PaperlessBilling"] = st.selectbox(
        FEATURE_LABELS["PaperlessBilling"], category_options["PaperlessBilling"], format_func=label
    )
with col2:
    values["PaymentMethod"] = st.selectbox(
        FEATURE_LABELS["PaymentMethod"], category_options["PaymentMethod"], format_func=label
    )

st.subheader("Servicios")
values["InternetService"] = st.selectbox(
    FEATURE_LABELS["InternetService"], category_options["InternetService"], format_func=label
)

has_internet = values["InternetService"] != "No"
col1, col2 = st.columns(2)
for col, feature in zip([col1, col2], INTERNET_DEPENDENT):
    with col:
        if has_internet:
            options = [o for o in category_options[feature] if o != "No internet service"]
        else:
            options = ["No internet service"]
        values[feature] = st.selectbox(
            FEATURE_LABELS[feature], options, format_func=label, disabled=not has_internet
        )

st.subheader("Perfil del cliente")
col1, col2, col3 = st.columns(3)
for col, feature in zip([col1, col2, col3], ["SeniorCitizen", "Partner", "Dependents"]):
    with col:
        values[feature] = st.selectbox(
            FEATURE_LABELS[feature], category_options[feature], format_func=label
        )

# ----------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------
st.divider()
if st.button("🔮 Predecir abandono", type="primary", use_container_width=True):
    X_new = pd.DataFrame([values], columns=features)

    pred = int(model.predict(X_new)[0])
    proba = float(model.predict_proba(X_new)[0][1])

    if proba < 0.30:
        nivel, mensaje = "🟢 Riesgo BAJO", "El cliente probablemente se quedará."
    elif proba < 0.60:
        nivel, mensaje = "🟡 Riesgo MEDIO", "Conviene hacer seguimiento al cliente."
    else:
        nivel, mensaje = "🔴 Riesgo ALTO", "Se recomienda una acción de retención (oferta, cambio de contrato, contacto)."

    st.subheader("Resultado")
    c1, c2 = st.columns(2)
    porcentaje = f"{proba * 100:.1f}%"
    c1.metric("Probabilidad de abandono", porcentaje)
    c2.metric("Predicción del modelo", artifact["target_labels"][pred])
    st.progress(proba)
