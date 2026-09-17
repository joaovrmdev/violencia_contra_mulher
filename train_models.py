import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Carregamento dos dados
df = pd.read_csv("dados_violencia.csv")

# 2. Definição das Features Base
NUM_COLS = ["idade_paciente", "numero_envolvidos"]
CAT_COLS = ["dia_semana_ocorrencia", "uf_ocorrencia", "raca_paciente", "escolaridade_paciente", "estado_civil_paciente", "autor_sexo"]
BIN_COLS = [
    "ocorreu_noite_madrugada", "reside_municipio_ocorrencia", "local_residencia",
    "gestante", "possui_deficiencia", "violencia_fisica", "violencia_psicologica",
    "violencia_sexual", "violencia_financeira", "violencia_negligencia", "violencia_tortura",
    "meio_forca_corporal", "meio_enforcamento", "meio_objeto_contundente",
    "meio_objeto_perfurante", "meio_arma_fogo", "meio_ameaca", "autor_alcoolizado"
]

TARGETS = {
    "reincidencia": "target_reincidencia",
    "agressor_intimo": "autor_parceiro_ou_ex",
    "delegacia_mulher": "encaminhamento_delegacia_mulher"
}

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_COLS),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="ignorado")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
            ("bin", SimpleImputer(strategy="constant", fill_value=0), BIN_COLS)
        ]
    )

for model_name, target_col in TARGETS.items():
    print(f"\n--- Treinando modelo: {model_name.upper()} (Target: {target_col}) ---")
    
    # Filtra linhas onde o target é não-nulo
    valid_data = df.dropna(subset=[target_col]).copy()
    y = valid_data[target_col].astype(int)
    X = valid_data[NUM_COLS + CAT_COLS + BIN_COLS]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", HistGradientBoostingClassifier(class_weight="balanced", random_state=42, max_iter=150))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Avaliação rápida
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    
    # Salva o pipeline serializado
    joblib.dump(pipeline, f"model_{model_name}.joblib")
    print(f"Modelo salvo como model_{model_name}.joblib")
