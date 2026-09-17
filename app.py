import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predição de Risco - Violência contra a Mulher",
    page_icon="🛡️",
    layout="wide"
)

# Cache para carregar modelos de forma eficiente no Streamlit Cloud
@st.cache_resource
def load_models():
    models = {
        "Ciclo de Reincidência": joblib.load("model_reincidencia.joblib"),
        "Agressor Íntimo (Cônjuge/Ex/Namorado)": joblib.load("model_agressor_intimo.joblib"),
        "Encaminhamento para DEAM": joblib.load("model_delegacia_mulher.joblib")
    }
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"Erro ao carregar os arquivos de modelo (.joblib). Certifique-se de executar `train_models.py` primeiro. Detalhes: {e}")
    st.stop()

st.title("🛡️ Sistema de Avaliação de Risco e Tipificação de Violência")
st.markdown("Ferramenta de predição orientada a dados do SINAN/SUS para apoiar a triagem e medidas protetivas.")

# Painel Lateral: Seleção do Modelo
st.sidebar.header("🎯 Objetivo da Predição")
selected_target = st.sidebar.selectbox(
    "Escolha o que deseja classificar:",
    options=list(models.keys())
)

st.sidebar.info(
    "**1. Reincidência:** Avalia se o fato faz parte de violência prévia repetida.\n"
    "**2. Agressor Íntimo:** Estima a probabilidade de autoria de parceiro/ex.\n"
    "**3. Encaminhamento DEAM:** Projeta adesão/fluxo para delegacia especializada."
)

st.subheader("📋 Dados da Ocorrência e Perfil")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Perfil da Vítima**")
    idade = st.number_input("Idade", min_value=14, max_value=110, value=28)
    gestante = st.selectbox("Gestante?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    raca = st.selectbox("Raça/Cor", options=["1", "2", "3", "4", "5", "ignorado"], 
                        format_func=lambda x: {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(x, "Ignorado"))
    escolaridade = st.selectbox("Escolaridade (Grau)", options=["1", "2", "3", "4", "5", "6", "ignorado"],
                                format_func=lambda x: {"1": "Fund. Incompleto", "2": "Fund. Completo", "3": "Médio Incompleto", "4": "Médio Completo", "5": "Superior Incompleto", "6": "Superior Completo"}.get(x, "Ignorado"))
    estado_civil = st.selectbox("Estado Civil", options=["1", "2", "3", "4", "ignorado"],
                                format_func=lambda x: {"1": "Solteira", "2": "Casada/União", "3": "Viúva", "4": "Separada"}.get(x, "Ignorado"))
    deficiencia = st.selectbox("Possui Deficiência?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

with col2:
    st.markdown("**Circunstâncias do Evento**")
    uf = st.selectbox("UF da Ocorrência", options=["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"])
    dia_semana = st.selectbox("Dia da Semana", options=[1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: "Domingo", 2: "Segunda", 3: "Terça", 4: "Quarta", 5: "Quinta", 6: "Sexta", 7: "Sábado"}[x])
    noite_madrugada = st.selectbox("Período Noturno (18h-06h)?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
    reside_municipio = st.selectbox("Ocorreu no município de residência?", options=[1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
    local_casa = st.selectbox("Ocorreu na residência?", options=[1, 0], format_func=lambda x: "Sim" if x == 1 else "Não")
    num_envolvidos = st.number_input("Número de Agressores", min_value=1, max_value=10, value=1)
    autor_sexo = st.selectbox("Sexo do Agressor", options=["M", "F", "Ambos", "ignorado"])
    autor_alcool = st.selectbox("Suspeita de uso de álcool?", options=[0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

with col3:
    st.markdown("**Tipologia e Meios Empregados**")
    v_fisica = st.checkbox("Violência Física", value=True)
    v_psico = st.checkbox("Violência Psicológica", value=True)
    v_sexual = st.checkbox("Violência Sexual", value=False)
    v_financ = st.checkbox("Violência Financeira", value=False)
    v_neglig = st.checkbox("Negligência / Abandono", value=False)
    v_tortura = st.checkbox("Tortura", value=False)
    st.markdown("---")
    m_forca = st.checkbox("Força Corporal / Espancamento", value=True)
    m_ameaca = st.checkbox("Ameaça Verbal", value=True)
    m_perfurante = st.checkbox("Objeto Perfurante / Faca", value=False)
    m_contundente = st.checkbox("Objeto Contundente", value=False)
    m_enforcamento = st.checkbox("Enforcamento / Sufocamento", value=False)
    m_arma = st.checkbox("Arma de Fogo", value=False)

# Montagem do DataFrame para Inferência
input_dict = {
    "idade_paciente": [idade],
    "numero_envolvidos": [num_envolvidos],
    "dia_semana_ocorrencia": [str(dia_semana)],
    "uf_ocorrencia": [uf],
    "raca_paciente": [raca],
    "escolaridade_paciente": [escolaridade],
    "estado_civil_paciente": [estado_civil],
    "autor_sexo": [autor_sexo],
    "ocorreu_noite_madrugada": [noite_madrugada],
    "reside_municipio_ocorrencia": [reside_municipio],
    "local_residencia": [local_casa],
    "gestante": [gestante],
    "possui_deficiencia": [deficiencia],
    "violencia_fisica": [int(v_fisica)],
    "violencia_psicologica": [int(v_psico)],
    "violencia_sexual": [int(v_sexual)],
    "violencia_financeira": [int(v_financ)],
    "violencia_negligencia": [int(v_neglig)],
    "violencia_tortura": [int(v_tortura)],
    "meio_forca_corporal": [int(m_forca)],
    "meio_enforcamento": [int(m_enforcamento)],
    "meio_objeto_contundente": [int(m_contundente)],
    "meio_objeto_perfurante": [int(m_perfurante)],
    "meio_arma_fogo": [int(m_arma)],
    "meio_ameaca": [int(m_ameaca)],
    "autor_alcoolizado": [autor_alcool]
}

input_df = pd.DataFrame(input_dict)

st.write("")
if st.button("Executar Análise de Risco", type="primary", use_container_width=True):
    model = models[selected_target]
    proba = model.predict_proba(input_df)[0][1]
    pred = int(proba >= 0.5)

    st.markdown("---")
    st.subheader("📊 Resultado da Classificação")
    
    r_col1, r_col2 = st.columns([1, 2])
    
    with r_col1:
        st.metric(
            label="Probabilidade Estimada",
            value=f"{proba * 100:.1f}%",
            delta="Alto Risco" if proba >= 0.5 else "Baixo Risco",
            delta_color="inverse" if proba >= 0.5 else "normal"
        )
        
    with r_col2:
        if proba >= 0.5:
            st.error(f"⚠️ **Alerta:** Padrão compatível com **{selected_target.upper()}** identificado pelo modelo.")
            st.write("Recomenda-se acionamento prioritário dos protocolos de acolhimento e rede de apoio.")
        else:
            st.success(f"✅ **Baixa Probabilidade:** Padrão estatístico não indica correlação primária com **{selected_target}** sob os limiares usuais.")
            st.write("Ainda assim, siga a rotina padrão de acolhimento e escuta ativa.")
