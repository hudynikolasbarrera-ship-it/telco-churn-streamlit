"""
Entrenamiento del modelo de abandono de clientes (Telco Customer Churn).

Uso:
    python train.py

Lee data/telco_churn.csv, limpia los datos, entrena un Pipeline
(preprocesamiento + Random Forest balanceado), lo evalúa y guarda
todo lo necesario para la app en model.joblib.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
RANDOM_STATE = 42

TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Carga el dataset crudo desde data/."""
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la misma limpieza validada en el notebook."""
    df = df.copy()
    # TotalCharges viene como texto; 11 clientes con tenure=0 lo tienen vacío
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # SeniorCitizen 0/1 -> No/Yes para que sea consistente con las demás categóricas
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Objetivo: 1 = abandona, 0 = se queda
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    return df


# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Preprocesamiento + clasificador en un solo objeto."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model(df: pd.DataFrame) -> dict:
    """Entrena, evalúa y construye el artefacto para la app."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluación
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": round(f1_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
    }

    print("=== Matriz de confusión (filas = real, columnas = predicho) ===")
    print(confusion_matrix(y_test, pred))
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_test, pred, target_names=["Se queda", "Abandona"]))
    print("=== Métricas clase 'Abandona' ===")
    for name, value in metrics.items():
        print(f"{name:>10}: {value}")

    # Información para construir la interfaz
    category_options = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    numeric_ranges = {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "median": float(X_train[col].median()),
        }
        for col in NUMERIC_FEATURES
    }

    return {
        "model": model,
        "model_name": "Random Forest balanceado",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "target_labels": {0: "Se queda", 1: "Abandona"},
        "task": "classification",
        "metrics": metrics,
    }


def main() -> None:
    df = clean_data(load_data())
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
