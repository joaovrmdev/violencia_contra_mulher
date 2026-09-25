import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML - Violência contra a Mulher", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_assets():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    with open("test_samples.json", "r") as f:
        samples = json.load(f)
    
    models = {}
    for label, data in metrics.items():
        slug = data["slug"]
        models[f"{label} - Regressão Logística"] = joblib.load(f"model_{slug}_lr.joblib")
        models[f"{label} - Árvores de Decisão (GBDT)"] = joblib.load(f"model_{slug}_tree.joblib")
        
    return models, metrics, samples

try:
    models, metrics_data, samples_data = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar os artefatos: {e}")
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Seleção de Caso e Algoritmo")
selected_theme = st.sidebar.selectbox("Tema:", list(metrics_data.keys()))
theme_metrics = metrics_data[selected_theme]

selected_algo = st.sidebar.radio(
    "Algoritmo:",
    ["Árvores de Decisão (GBDT)", "Regressão Logística"]
)

model_key = f"{selected_theme} - {selected_algo}"
active_model = models[model_key]

st.title("🛡️ Avaliação e Inferência - Violência contra a Mulher")

tab_eval, tab_sim = st.tabs(["📊 Comparativo & Validação", "🔮 Simulador & Casos de Teste"])

# ==================== ABA 1: VALIDAÇÃO E CORRELAÇÕES ====================
with tab_eval:
    st.subheader(f"Desempenho de Validação — {selected_theme}")
    st.caption(f"Amostragem: {theme_metrics['n_train']:,} treino (80%) | {theme_metrics['n_test']:,} teste (20% não vistos)")
    
    comp_col1, comp_col2 = st.columns(2)
    for col, algo_name in zip([comp_col1, comp_col2], ["Árvores de Decisão (GBDT)", "Regressão Logística"]):
        with col:
            st.markdown(f"### {algo_name}")
            m_data = theme_metrics[algo_name]
            rep = m_data["report"]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", f"{m_data['roc_auc']:.3f}")
            c2.metric("Recall (Classe 1)", f"{rep['1']['recall']:.3f}")
            c3.metric("F1-Score", f"{rep['1']['f1-score']:.3f}")
            
            st.markdown("**Matriz de Confusão**")
            st.dataframe(
                pd.DataFrame(m_data["confusion_matrix"], columns=["Pred 0", "Pred 1"], index=["Real 0", "Real 1"]),
                use_container_width=True
            )

    st.markdown("---")
    st.subheader(f"📈 Correlação das Variáveis com: {selected_theme}")
    st.caption("Correlação linear direta (Pearson / Ponto-Bisserial) com a variável alvo.")
    
    corr_df = pd.DataFrame(
        list(theme_metrics["correlations"].items()),
        columns=["Variável", "Correlação"]
    ).sort_values(by="Correlação", ascending=True)
    
    st.bar_chart(corr_df.set_index("Variável"), horizontal=True, color="#ff4b4b")

# ==================== ABA 2: SIMULADOR E CASOS REAIS ====================
with tab_sim:
    st.subheader("Entrada de Dados para Teste e Predição")
    
    # Seletor de Casos Reais vs Modo Manual
    samples_list = samples_data.get(selected_theme, [])
    sample_options = ["Modo Manual (Preenchimento Livre)"] + [
        f"Caso Real #{i+1} (Desfecho Real: {'Sim (1)' if s['target_real'] == 1 else 'Não (0)'} | Idade: {int(s['idade_paciente'])} | {s['uf_ocorrencia']})"
        for i, s in enumerate(samples_list)
    ]
    
    selected_sample_idx = st.selectbox(
        "Selecione uma amostra do conjunto de teste (não vista pelo modelo) ou monte um caso livre:",
        options=range(len(sample_options)),
        format_func=lambda x: sample_options[x]
    )
    
    is_sample = selected_sample_idx > 0
    sample_val = samples_list[selected_sample_idx - 1] if is_sample else None
    
    if is_sample:
        st.info(f"📌 **Carregando dados reais do teste.** Desfecho registrado no SINAN: **{'POSITIVO (1)' if sample_val['target_real'] == 1 else 'NEGATIVO (0)'}**")

    # Mapeamento dos valores padrão conforme a seleção
    def get_val(key, default):
        if is_sample and key in sample_val:
            return sample_val[key]
        return default

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Perfil da Vítima**")
        idade = st.number_input("Idade", 14, 105, int(get_val("idade_paciente", 30)))
        gestante = st.selectbox("Gestante?", [0, 1], index=int(get_val("gestante", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        
        raca_opts = ["1", "2", "3", "4", "5", "ignorado"]
        cur_raca = str(get_val("raca_paciente", "1"))
        raca_idx = raca_opts.index(cur_raca) if cur_raca in raca_opts else 0
        raca = st.selectbox("Raça/Cor", raca_opts, index=raca_idx, format_func=lambda x: {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(x, "Ignorado"))
        
        esc_opts = ["1", "2", "3", "4", "5", "6", "ignorado"]
        cur_esc = str(get_val("escolaridade_paciente", "4"))
        esc_idx = esc_opts.index(cur_esc) if cur_esc in esc_opts else 0
        escolaridade = st.selectbox("Escolaridade", esc_opts, index=esc_idx, format_func=lambda x: {"1": "Fund. Incompleto", "2": "Fund. Completo", "3": "Médio Incompleto", "4": "Médio Completo", "5": "Superior Incompleto", "6": "Superior Completo"}.get(x, "Ignorado"))
        
        civ_opts = ["1", "2", "3", "4", "ignorado"]
        cur_civ = str(get_val("estado_civil_paciente", "1"))
        civ_idx = civ_opts.index(cur_civ) if cur_civ in civ_opts else 0
        estado_civil = st.selectbox("Estado Civil", civ_opts, index=civ_idx, format_func=lambda x: {"1": "Solteira", "2": "Casada/União", "3": "Viúva", "4": "Separada"}.get(x, "Ignorado"))
        
        deficiencia = st.selectbox("Possui Deficiência?", [0, 1], index=int(get_val("possui_deficiencia", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        
    with c2:
        st.markdown("**Circunstâncias**")
        uf_opts = ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"]
        cur_uf = str(get_val("uf_ocorrencia", "SP"))
        uf_idx = uf_opts.index(cur_uf) if cur_uf in uf_opts else 0
        uf = st.selectbox("UF", uf_opts, index=uf_idx)
        
        dia_semana = st.selectbox("Dia da Semana", [1, 2, 3, 4, 5, 6, 7], index=int(get_val("dia_semana_ocorrencia", 1))-1, format_func=lambda x: {1: "Dom", 2: "Seg", 3: "Ter", 4: "Qua", 5: "Qui", 6: "Sex", 7: "Sáb"}[x])
        noite = st.selectbox("Noturno (18h-06h)?", [0, 1], index=int(get_val("ocorreu_noite_madrugada", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        mesmo_mun = st.selectbox("Mesmo Município?", [1, 0], index=0 if get_val("reside_municipio_ocorrencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não")
        casa = st.selectbox("Na Residência?", [1, 0], index=0 if get_val("local_residencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não")
        num_env = st.number_input("Nº Agressores", 1, 10, int(get_val("numero_envolvidos", 1)))
        
        sex_opts = ["M", "F", "Ambos", "ignorado"]
        cur_sex = str(get_val("autor_sexo", "M"))
        sex_idx = sex_opts.index(cur_sex) if cur_sex in sex_opts else 0
        autor_sexo = st.selectbox("Sexo do Agressor", sex_opts, index=sex_idx)
        
        alcool = st.selectbox("Uso de Álcool?", [0, 1], index=int(get_val("autor_alcoolizado", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        
    with c3:
        st.markdown("**Meios e Armas**")
        v_fisica = st.checkbox("Violência Física", bool(get_val("violencia_fisica", True)))
        v_psico = st.checkbox("Violência Psicológica", bool(get_val("violencia_psicologica", True)))
        v_sexual = st.checkbox("Violência Sexual", bool(get_val("violencia_sexual", False)))
        v_financ = st.checkbox("Violência Financeira", bool(get_val("violencia_financeira", False)))
        v_neglig = st.checkbox("Negligência", bool(get_val("violencia_negligencia", False)))
        v_tortura = st.checkbox("Tortura", bool(get_val("violencia_tortura", False)))
        m_forca = st.checkbox("Força Corporal", bool(get_val("meio_forca_corporal", True)))
        m_ameaca = st.checkbox("Ameaça", bool(get_val("meio_ameaca", True)))
        m_perfurante = st.checkbox("Objeto Perfurante", bool(get_val("meio_objeto_perfurante", False)))
        m_contundente = st.checkbox("Objeto Contundente", bool(get_val("meio_objeto_contundente", False)))
        m_enforcamento = st.checkbox("Enforcamento / Sufocamento", bool(get_val("meio_enforcamento", False)))
        m_arma = st.checkbox("Arma de Fogo", bool(get_val("meio_arma_fogo", False)))

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
        prob = active_model.predict_proba(input_df)[0][1]
        pred_label = 1 if prob >= 0.5 else 0
        
        st.markdown("---")
        rc1, rc2, rc3 = st.columns([1, 2, 1])
        with rc1:
            st.metric(f"Probabilidade ({selected_algo})", f"{prob * 100:.1f}%")
        with rc2:
            if pred_label == 1:
                st.error(f"⚠️ Alerta: Padrão estatístico compatível com **{selected_theme}**.")
            else:
                st.success(f"✅ Classificação abaixo do limiar crítico para **{selected_theme}**.")
                
        with rc3:
            if is_sample:
                real_val = sample_val["target_real"]
                acertou = pred_label == real_val
                st.metric(
                    label="Desfecho Real do Caso",
                    value="Positivo (1)" if real_val == 1 else "Negativo (0)",
                    delta="Acerto do Modelo" if acertou else "Erro de Classificação",
                    delta_color="normal" if acertou else "inverse"
                )

        if m_arma or m_enforcamento:
            st.warning(
                "🚨 **Aviso de Risco Crítico de Letalidade (Viés de Sobrevivência):** "
                "O uso de arma de fogo ou enforcamento/asfixia é o maior preditor isolado de feminicídio. "
                "Caso o score de reincidência se mostre baixo, isso decorre do viés do SINAN, "
                "onde casos fatais no local não geram repetição ambulatorial."
            )
