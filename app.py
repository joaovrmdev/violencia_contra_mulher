import os
import gdown
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

st.set_page_config(
    page_title="ML - Violência contra a Mulher (SINAN)",
    page_icon="🛡️",
    layout="wide"
)

# 1. Configurações de Download e Features
DRIVE_FILE_ID = "1QGpHtVrIfxhg7L32D5PFBSurVe392NLi"
LOCAL_CSV_PATH = "dados_violencia_drive.csv"

NUM_COLS = ["idade_paciente", "numero_envolvidos"]
CAT_COLS = [
    "dia_semana_ocorrencia", "uf_ocorrencia", "raca_paciente", 
    "escolaridade_paciente", "estado_civil_paciente", "autor_sexo"
]
BIN_COLS = [
    "ocorreu_noite_madrugada", "reside_municipio_ocorrencia", "local_residencia",
    "gestante", "possui_deficiencia", "violencia_fisica", "violencia_psicologica",
    "violencia_sexual", "violencia_financeira", "violencia_negligencia", "violencia_tortura",
    "meio_forca_corporal", "meio_enforcamento", "meio_objeto_contundente",
    "meio_objeto_perfurante", "meio_arma_fogo", "meio_ameaca", "autor_alcoolizado"
]

TARGET_MAP = {
    "Ciclo de Reincidência": "target_reincidencia",
    "Agressor Íntimo (Parceiro/Ex)": "autor_parceiro_ou_ex",
    "Encaminhamento DEAM": "encaminhamento_delegacia_mulher"
}

# 2. Carregamento com Cache do Google Drive
@st.cache_data(show_spinner="Baixando e carregando dados do Google Drive...")
def load_data_from_drive(file_id: str, local_path: str):
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False)
    
    df = pd.read_csv(local_path)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

# 3. Pipeline de Treinamento e Avaliação (80% Treino / 20% Teste)
@st.cache_resource(show_spinner="Treinando classificador e validando no conjunto de teste...")
def train_and_evaluate_model(df: pd.DataFrame, target_col: str):
    data = df.dropna(subset=[target_col]).copy()
    y = data[target_col].astype(int)
    
    features = [col for col in NUM_COLS + CAT_COLS + BIN_COLS if col in data.columns]
    X = data[features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="ignorado")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in NUM_COLS if c in features]),
            ("cat", cat_pipe, [c for c in CAT_COLS if c in features]),
            ("bin", SimpleImputer(strategy="constant", fill_value=0), [c for c in BIN_COLS if c in features])
        ]
    )
    
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=42,
            max_iter=150
        ))
    ])
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features
    }
    
    return clf, metrics

# --- BARRA LATERAL ---
st.sidebar.title("⚙️ Configurações do Modelo")
selected_label = st.sidebar.selectbox("Classificador:", list(TARGET_MAP.keys()))
target_column = TARGET_MAP[selected_label]

st.sidebar.info(
    "**1. Reincidência:** Prever se o evento faz parte de um ciclo repetido.\n"
    "**2. Agressor Íntimo:** Prever vínculo com parceiro/ex.\n"
    "**3. Encaminhamento DEAM:** Prever direcionamento à Delegacia da Mulher."
)

# --- FLUXO PRINCIPAL ---
st.title("🛡️ Avaliação e Inferência - Violência contra a Mulher")

try:
    df_raw = load_data_from_drive(DRIVE_FILE_ID, LOCAL_CSV_PATH)
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

model, metrics = train_and_evaluate_model(df_raw, target_column)

tab_eval, tab_sim = st.tabs(["📊 Desempenho no Conjunto de Teste (Validação)", "🔮 Simulador de Risco"])

with tab_eval:
    st.subheader(f"Métricas de Teste — {selected_label}")
    st.caption(f"Amostragem: {metrics['n_train']} no treino (80%) | {metrics['n_test']} no teste (20% não vistos)")
    
    m1, m2, m3, m4 = st.columns(4)
    rep = metrics["report"]
    
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("Precision (Classe 1)", f"{rep['1']['precision']:.3f}")
    m3.metric("Recall (Classe 1)", f"{rep['1']['recall']:.3f}")
    m4.metric("F1-Score (Classe 1)", f"{rep['1']['f1-score']:.3f}")
    
    col_cm, col_rep = st.columns([1, 2])
    
    with col_cm:
        st.markdown("**Matriz de Confusão (Teste)**")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            columns=["Pred 0 (Não)", "Pred 1 (Sim)"],
            index=["Real 0 (Não)", "Real 1 (Sim)"]
        )
        st.dataframe(cm_df, use_container_width=True)
        
    with col_rep:
        st.markdown("**Relatório de Classificação Detalhado**")
        df_rep = pd.DataFrame(rep).transpose()
        st.dataframe(df_rep.style.format(precision=3), use_container_width=True)

with tab_sim:
    st.subheader("Entrada de Variáveis para Inferência")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Perfil da Vítima**")
        idade = st.number_input("Idade", min_value=14, max_value=105, value=30)
        gestante = st.selectbox("Gestante?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        raca = st.selectbox("Raça/Cor", ["1", "2", "3", "4", "5", "ignorado"],
                            format_func=lambda x: {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(x, "Ignorado"))
        escolaridade = st.selectbox("Escolaridade", ["1", "2", "3", "4", "5", "6", "ignorado"],
                                    format_func=lambda x: {"1": "Fund. Incompleto", "2": "Fund. Completo", "3": "Médio Incompleto", "4": "Médio Completo", "5": "Superior Incompleto", "6": "Superior Completo"}.get(x, "Ignorado"))
        estado_civil = st.selectbox("Estado Civil", ["1", "2", "3", "4", "ignorado"],
                                    format_func=lambda x: {"1": "Solteira", "2": "Casada/União", "3": "Viúva", "4": "Separada"}.get(x, "Ignorado"))
        deficiencia = st.selectbox("Deficiência?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        
    with c2:
        st.markdown("**Circunstâncias do Evento**")
        uf = st.selectbox("UF", ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"])
        dia_semana = st.selectbox("Dia da Semana", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: "Domingo", 2: "Segunda", 3: "Terça", 4: "Quarta", 5: "Quinta", 6: "Sexta", 7: "Sábado"}[x])
        noite_madrugada = st.selectbox("Noturno (18h-06h)?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        reside_municipio = st.selectbox("Mesmo Município?", [1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
        local_casa = st.selectbox("Na Residência?", [1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
        num_envolvidos = st.number_input("Nº de Agressores", 1, 10, 1)
        autor_sexo = st.selectbox("Sexo do Agressor", ["M", "F", "Ambos", "ignorado"])
        autor_alcool = st.selectbox("Suspeita de Álcool?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        
    with c3:
        st.markdown("**Tipologia e Meios**")
        v_fisica = st.checkbox("Violência Física", value=True)
        v_psico = st.checkbox("Violência Psicológica", value=True)
        v_sexual = st.checkbox("Violência Sexual", value=False)
        v_financ = st.checkbox("Violência Financeira", value=False)
        v_neglig = st.checkbox("Negligência/Abandono", value=False)
        v_tortura = st.checkbox("Tortura", value=False)
        m_forca = st.checkbox("Força Corporal", value=True)
        m_ameaca = st.checkbox("Ameaça", value=True)
        m_perfurante = st.checkbox("Objeto Perfurante", value=False)
        m_contundente = st.checkbox("Objeto Contundente", value=False)
        m_enforcamento = st.checkbox("Enforcamento", value=False)
        m_arma = st.checkbox("Arma de Fogo", value=False)

    input_data = pd.DataFrame([{
        "idade_paciente": idade,
        "numero_envolvidos": num_envolvidos,
        "dia_semana_ocorrencia": str(dia_semana),
        "uf_ocorrencia": uf,
        "raca_paciente": raca,
        "escolaridade_paciente": escolaridade,
        "estado_civil_paciente": estado_civil,
        "autor_sexo": autor_sexo,
        "ocorreu_noite_madrugada": noite_madrugada,
        "reside_municipio_ocorrencia": reside_municipio,
        "local_residencia": local_casa,
        "gestante": gestante,
        "possui_deficiencia": deficiencia,
        "violencia_fisica": int(v_fisica),
        "violencia_psicologica": int(v_psico),
        "violencia_sexual": int(v_sexual),
        "violencia_financeira": int(v_financ),
        "violencia_negligencia": int(v_neglig),
        "violencia_tortura": int(v_tortura),
        "meio_forca_corporal": int(m_forca),
        "meio_enforcamento": int(m_enforcamento),
        "meio_objeto_contundente": int(m_contundente),
        "meio_objeto_perfurante": int(m_perfurante),
        "meio_arma_fogo": int(m_arma),
        "meio_ameaca": int(m_ameaca),
        "autor_alcoolizado": autor_alcool
    }])

    if st.button("Calcular Probabilidade de Risco", type="primary", use_container_width=True):
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            st.metric("Probabilidade Calculada", f"{prob * 100:.1f}%")
        with rc2:
            if prob >= 0.5:
                st.error(f"⚠️ Alerta: Alto risco detectado para **{selected_label}**.")
            else:
                st.success(f"✅ Risco calculado abaixo do limiar crítico para **{selected_label}**.")
