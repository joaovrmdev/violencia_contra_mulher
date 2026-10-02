import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="ML - Violencia contra a Mulher (SINAN)",
    layout="wide"
)

st.markdown("""
<style>
    /* Estilização global e dos Cards de Métricas Nativos */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #f1f3f5;
        padding: 20px 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.04);
    }
    
    /* Grid Responsivo e Fluido para a Matriz de Confusão */
    .cm-grid {
        display: grid;
        /* auto-fit e minmax garantem que o grid quebre a linha sozinho se a tela encolher */
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 15px;
        margin-bottom: 35px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Cartões Flutuantes Sóbrios */
    .cm-card {
        background-color: #ffffff;
        border: 1px solid #f8f9fa;
        border-radius: 12px;
        padding: 24px;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap; /* Permite que o conteúdo interno quebre se espremido */
        gap: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .cm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
    
    /* Bloco de texto flexível */
    .cm-info { 
        display: flex; 
        flex-direction: column; 
        flex: 1 1 160px; /* Cresce, mas respeita um mínimo antes de forçar o wrap */
    }
    
    .cm-title {
        font-size: 0.85em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .cm-desc {
        font-size: 0.8em;
        color: #868e96;
        line-height: 1.4;
    }
    
    /* Valores unificados em Grafite Escuro para seriedade */
    .cm-value {
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #212529; 
        text-align: right;
    }
    
    /* Paleta Executiva: Cores apenas nos títulos, com tons fechados e dessaturados */
    .card-tn .cm-title { color: #6c757d; } /* Cinza Ardósia */
    .card-fp .cm-title { color: #a56a6a; } /* Terracota Fechado */
    .card-fn .cm-title { color: #a68a56; } /* Ouro Velho */
    .card-tp .cm-title { color: #5f8270; } /* Verde Musgo */

    /* Media Query para extrema restrição de espaço (Smartphones ou divisão de tela) */
    @media (max-width: 360px) {
        .cm-card {
            flex-direction: column;
            align-items: flex-start;
        }
        .cm-value {
            text-align: left;
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# 1. Carregamento dos Artefatos
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
    st.error(f"Erro ao carregar arquivos gerados pelo script de treino: {e}")
    st.stop()

# 2. Barra Lateral: Parâmetros
st.sidebar.title("Parâmetros")
selected_theme = st.sidebar.selectbox("Cenário de Análise:", list(metrics_data.keys()))
theme_metrics = metrics_data[selected_theme]

selected_algo = st.sidebar.radio(
    "Família de Algoritmo:",
    ["Árvores de Decisão (GBDT)", "Regressão Logística"]
)

model_key = f"{selected_theme} - {selected_algo}"
active_model = models[model_key]

st.title("Avaliação e Inferência - Violência contra a Mulher")
st.markdown("Plataforma de modelagem supervisionada de triagem e risco (Dados SINAN/DATASUS).")

tab_eval, tab_corr, tab_sim = st.tabs([
    "Desempenho dos Modelos",
    "Matriz de Correlação",
    "Simulador e Casos de Teste"
])

# ==================== ABA 1: COMPARAÇÃO DOS MODELOS ====================
with tab_eval:
    st.subheader(f"Validação no Conjunto de Teste - {selected_theme}")
    st.caption(f"Amostragem: {theme_metrics['n_train']:,} registros no treino (80%) | {theme_metrics['n_test']:,} registros no teste (20% não vistos)")
    
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
            
            st.markdown("#### Matriz de Confusão")
            
            tn, fp = m_data["confusion_matrix"][0]
            fn, tp = m_data["confusion_matrix"][1]
            
           # Matriz de Confusão Redesenhada em Grid Card 2x2 (HTML compactado)
            html_cm = f"""
            <div class="cm-grid">
                <div class="cm-card card-tn">
                    <div class="cm-info"><span class="cm-title">Verdadeiro Negativo</span><span class="cm-desc">Modelo previu negativo e acertou.</span></div>
                    <div class="cm-value">{tn:,}</div>
                </div>
                <div class="cm-card card-fp">
                    <div class="cm-info"><span class="cm-title">Falso Positivo</span><span class="cm-desc">Erro: Modelo acusou risco inexistente (Alarme Falso).</span></div>
                    <div class="cm-value">{fp:,}</div>
                </div>
                <div class="cm-card card-fn">
                    <div class="cm-info"><span class="cm-title">Falso Negativo</span><span class="cm-desc">Erro Crítico: Modelo falhou em detectar o risco real.</span></div>
                    <div class="cm-value">{fn:,}</div>
                </div>
                <div class="cm-card card-tp">
                    <div class="cm-info"><span class="cm-title">Verdadeiro Positivo</span><span class="cm-desc">Modelo detectou o risco existente e acertou.</span></div>
                    <div class="cm-value">{tp:,}</div>
                </div>
            </div>
            """
            st.markdown(html_cm, unsafe_allow_html=True)

            
            with st.expander("Ver Relatório Completo de Classificação"):
                df_rep = pd.DataFrame(rep).transpose()
                st.dataframe(df_rep.style.format(precision=3), use_container_width=True)

# ==================== ABA 2: MATRIZ DE CORRELAÇÃO ====================
with tab_corr:
    st.subheader(f"Matriz de Correlação Multivariada - {selected_theme}")
    st.caption("Correlação linear de Pearson (triângulo inferior). Omissão da metade simétrica para foco analítico.")
    
    corr_info = theme_metrics.get("full_correlation_matrix")
    if corr_info:
        df_corr = pd.DataFrame(
            corr_info["data"],
            columns=corr_info["columns"],
            index=corr_info["index"]
        )

        mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)
        df_corr_masked = df_corr.mask(mask)

        fig = px.imshow(
            df_corr_masked,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1]
        )
        
        fig.update_layout(
            height=750,
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), side="bottom"),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=40, r=40, t=30, b=120),
            coloraxis_colorbar=dict(title="Correlação", thickness=15, len=0.75)
        )
        
        fig.for_each_annotation(lambda a: a.update(text="") if a.text == "nan" else ())
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Matriz de correlação não encontrada no arquivo metrics.json.")

# ==================== ABA 3: SIMULADOR E CASOS REAIS ====================
with tab_sim:
    st.subheader("Simulação de Risco e Inserção de Casos Reais")
    
    samples_list = samples_data.get(selected_theme, [])
    sample_options = ["Modo Manual (Preenchimento Livre)"] + [
        f"Caso Real #{i+1} (Desfecho Real: {'Sim (1)' if s['target_real'] == 1 else 'Não (0)'} | Idade: {int(s['idade_paciente'])} anos | {s['uf_ocorrencia']})"
        for i, s in enumerate(samples_list)
    ]
    
    selected_sample_idx = st.selectbox(
        "Carregue uma ocorrência real do conjunto de teste ou mantenha em preenchimento livre:",
        options=range(len(sample_options)),
        format_func=lambda x: sample_options[x]
    )
    
    is_sample = selected_sample_idx > 0
    sample_val = samples_list[selected_sample_idx - 1] if is_sample else None
    
    if is_sample:
        desfecho_txt = "POSITIVO (1)" if sample_val["target_real"] == 1 else "NEGATIVO (0)"
        st.info(f"Registro Real Carregado. Desfecho documentado originalmente na ficha do SINAN: {desfecho_txt}")

    def get_val(key, default):
        if is_sample and key in sample_val:
            return sample_val[key]
        return default

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Perfil da Vítima**")
        idade = st.number_input("Idade", 14, 105, int(get_val("idade_paciente", 28)))
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
        st.markdown("**Circunstâncias da Ocorrência**")
        uf_opts = ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"]
        cur_uf = str(get_val("uf_ocorrencia", "SP"))
        uf_idx = uf_opts.index(cur_uf) if cur_uf in uf_opts else 0
        uf = st.selectbox("UF", uf_opts, index=uf_idx)
        
        dia_semana = st.selectbox("Dia da Semana", [1, 2, 3, 4, 5, 6, 7], index=int(get_val("dia_semana_ocorrencia", 1)) - 1, format_func=lambda x: {1: "Dom", 2: "Seg", 3: "Ter", 4: "Qua", 5: "Qui", 6: "Sex", 7: "Sáb"}[x])
        noite = st.selectbox("Período Noturno (18h-06h)?", [0, 1], index=int(get_val("ocorreu_noite_madrugada", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        mesmo_mun = st.selectbox("Reside no mesmo município?", [1, 0], index=0 if get_val("reside_municipio_ocorrencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não")
        casa = st.selectbox("Ocorreu na Residência?", [1, 0], index=0 if get_val("local_residencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não")
        num_env = st.number_input("Número de Agressores", 1, 10, int(get_val("numero_envolvidos", 1)))
        
        sex_opts = ["M", "F", "Ambos", "ignorado"]
        cur_sex = str(get_val("autor_sexo", "M"))
        sex_idx = sex_opts.index(cur_sex) if cur_sex in sex_opts else 0
        autor_sexo = st.selectbox("Sexo do Agressor", sex_opts, index=sex_idx)
        
        alcool = st.selectbox("Suspeita de Uso de Álcool?", [0, 1], index=int(get_val("autor_alcoolizado", 0)), format_func=lambda x: "Sim" if x == 1 else "Não")
        
    with c3:
        st.markdown("**Tipologia e Meios Empregados**")
        v_fisica = st.checkbox("Violência Física", bool(get_val("violencia_fisica", True)))
        v_psico = st.checkbox("Violência Psicológica", bool(get_val("violencia_psicologica", True)))
        v_sexual = st.checkbox("Violência Sexual", bool(get_val("violencia_sexual", False)))
        v_financ = st.checkbox("Violência Financeira", bool(get_val("violencia_financeira", False)))
        v_neglig = st.checkbox("Negligência / Abandono", bool(get_val("violencia_negligencia", False)))
        v_tortura = st.checkbox("Tortura", bool(get_val("violencia_tortura", False)))
        m_forca = st.checkbox("Força Corporal / Espancamento", bool(get_val("meio_forca_corporal", True)))
        m_ameaca = st.checkbox("Ameaça Verbal", bool(get_val("meio_ameaca", True)))
        m_perfurante = st.checkbox("Objeto Perfurante / Faca", bool(get_val("meio_objeto_perfurante", False)))
        m_contundente = st.checkbox("Objeto Contundente", bool(get_val("meio_objeto_contundente", False)))
        m_enforcamento = st.checkbox("Enforcamento / Estrangulamento", bool(get_val("meio_enforcamento", False)))
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
            st.metric(
                label=f"Probabilidade ({selected_algo})",
                value=f"{prob * 100:.1f}%",
                delta="Alto Risco" if prob >= 0.5 else "Baixo Risco",
                delta_color="inverse" if prob >= 0.5 else "normal"
            )
        with rc2:
            if pred_label == 1:
                st.error(f"Alerta: Padrão estatístico compatível com {selected_theme}.")
            else:
                st.success(f"Classificação abaixo do limiar crítico para {selected_theme}.")
                
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
                "Atenção - Risco Crítico de Letalidade: "
                "O emprego de arma de fogo ou asfixia é o maior preditor isolado de feminicídio. "
                "Em bases hospitalares como o SINAN, a letalidade imediata tende a truncar o histórico ambulatorial, "
                "gerando uma probabilidade de reincidência contraintuitivamente menor devido ao viés de sobrevivência."
            )
