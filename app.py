# app.py
import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML - Violência contra a Mulher", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_assets():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    models = {
        label: joblib.load(f"model_{data['slug']}.joblib")
        for label, data in metrics.items()
    }
    return models, metrics

try:
    models, metrics_data = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar os artefatos serializados: {e}")
    st.stop()

st.sidebar.title("⚙️ Seleção de Modelo")
selected_label = st.sidebar.selectbox("Classificador:", list(metrics_data.keys()))
model = models[selected_label]
metrics = metrics_data[selected_label]

st.title("🛡️ Avaliação e Inferência - Violência contra a Mulher")

tab_eval, tab_sim = st.tabs(["📊 Desempenho no Teste (Validação)", "🔮 Simulador de Risco"])

with tab_eval:
    st.subheader(f"Métricas de Teste — {selected_label}")
    st.caption(f"Amostragem: {metrics['n_train']:,} treino (80%) | {metrics['n_test']:,} teste (20% não vistos)")
    
    m1, m2, m3, m4 = st.columns(4)
    rep = metrics["report"]
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("Precision (Classe 1)", f"{rep['1']['precision']:.3f}")
    m3.metric("Recall (Classe 1)", f"{rep['1']['recall']:.3f}")
    m4.metric("F1-Score (Classe 1)", f"{rep['1']['f1-score']:.3f}")
    
    c_cm, c_rep = st.columns([1, 2])
    with c_cm:
        st.markdown("**Matriz de Confusão (Teste)**")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], columns=["Pred 0", "Pred 1"], index=["Real 0", "Real 1"]), use_container_width=True)
    with c_rep:
        st.markdown("**Relatório Completo**")
        st.dataframe(pd.DataFrame(rep).transpose().style.format(precision=3), use_container_width=True)

with tab_sim:
    st.subheader("Entrada de Variáveis para Predição")
    c1, c2, c3 = st.columns(3)
    with c1:
        idade = st.number_input("Idade", 14, 105, 30)
        gestante = st.selectbox("Gestante?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        raca = st.selectbox("Raça/Cor", ["1", "2", "3", "4", "5", "ignorado"], format_func=lambda x: {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(x, "Ignorado"))
        escolaridade = st.selectbox("Escolaridade", ["1", "2", "3", "4", "5", "6", "ignorado"], format_func=lambda x: {"1": "Fund. Incompleto", "2": "Fund. Completo", "3": "Médio Incompleto", "4": "Médio Completo", "5": "Superior Incompleto", "6": "Superior Completo"}.get(x, "Ignorado"))
        estado_civil = st.selectbox("Estado Civil", ["1", "2", "3", "4", "ignorado"], format_func=lambda x: {"1": "Solteira", "2": "Casada/União", "3": "Viúva", "4": "Separada"}.get(x, "Ignorado"))
        deficiencia = st.selectbox("Deficiência?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    with c2:
        uf = st.selectbox("UF", ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"])
        dia_semana = st.selectbox("Dia da Semana", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: "Dom", 2: "Seg", 3: "Ter", 4: "Qua", 5: "Qui", 6: "Sex", 7: "Sáb"}[x])
        noite = st.selectbox("Noturno (18h-06h)?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        mesmo_mun = st.selectbox("Mesmo Município?", [1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
        casa = st.selectbox("Na Residência?", [1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
        num_env = st.number_input("Nº Agressores", 1, 10, 1)
        autor_sexo = st.selectbox("Sexo do Agressor", ["M", "F", "Ambos", "ignorado"])
        alcool = st.selectbox("Uso de Álcool?", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    with c3:
        v_fisica = st.checkbox("Violência Física", True)
        v_psico = st.checkbox("Violência Psicológica", True)
        v_sexual = st.checkbox("Violência Sexual", False)
        v_financ = st.checkbox("Violência Financeira", False)
        v_neglig = st.checkbox("Negligência", False)
        v_tortura = st.checkbox("Tortura", False)
        m_forca = st.checkbox("Força Corporal", True)
        m_ameaca = st.checkbox("Ameaça", True)
        m_perfurante = st.checkbox("Objeto Perfurante", False)
        m_contundente = st.checkbox("Objeto Contundente", False)
        m_enforcamento = st.checkbox("Enforcamento", False)
        m_arma = st.checkbox("Arma de Fogo", False)

    input_df = pd.DataFrame([{
        "idade_paciente": idade, "numero_envolvidos": num_env, "dia_semana_ocorrencia": str(dia_semana),
        "uf_ocorrencia": uf, "raca_paciente": raca, "escolaridade_paciente": escolaridade,
        "estado_civil_paciente": estado_civil, "autor_sexo": autor_sexo, "ocorreu_noite_madrugada": noite,
        "reside_municipio_ocorrencia": mesmo_mun, "local_residencia": casa, "gestante": gestante,
        "possui_deficiencia": deficiencia, "violencia_fisica": int(v_fisica), "violencia_psicologica": int(v_psico),
        "violencia_sexual": int(v_sexual), "violencia_financeira": int(v_financ), "violencia_negligencia": int(v_neglig),
        "violencia_tortura": int(v_tortura), "meio_forca_corporal": int(m_forca), "meio_enforcamento": int(m_enforcamento),
        "meio_objeto_contundente": int(m_contundente), "meio_objeto_perfurante": int(m_perfurante),
        "meio_arma_fogo": int(m_arma), "meio_ameaca": int(m_ameaca), "autor_alcoolizado": alcool
    }])

    if st.button("Executar Inferência", type="primary", use_container_width=True):
        prob = model.predict_proba(input_df)[0][1]
        st.markdown("---")
        rc1, rc2 = st.columns([1, 3])
        with rc1:
            st.metric("Probabilidade de Risco", f"{prob * 100:.1f}%")
        with rc2:
            if prob >= 0.5:
                st.error(f"⚠️ Alerta: Padrão compatível com **{selected_label}**.")
            else:
                st.success(f"✅ Classificação abaixo do limiar crítico para **{selected_label}**.")
