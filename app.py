# app.py
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

# Injeção de CSS para Dashboard Profissional e Grid de Confusão
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
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .cm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    }
    
    .cm-info { 
        display: flex; 
        flex-direction: column; 
        flex: 1 1 160px;
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
    
    .cm-value {
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #212529; 
        text-align: right;
    }
    
    /* Paleta Executiva: Cores apenas nos títulos */
    .card-tn .cm-title { color: #6c757d; }
    .card-fp .cm-title { color: #a56a6a; }
    .card-fn .cm-title { color: #a68a56; }
    .card-tp .cm-title { color: #5f8270; }

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
    st.info("Certifique-se de que os arquivos 'metrics.json', 'test_samples.json' e os modelos '.joblib' estão no diretório.")
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
    st.subheader(f"Associações entre Fatores - {selected_theme}")
    st.markdown(
        "Os valores medem o grau de ligação entre os fatores registrados na ocorrência: quanto mais próximo de 1 (e mais azul), maior a frequência com que esses eventos acontecem juntos. "
        "Valores próximos de 0 indicam ausência de relação direta. A diagonal de auto-associação (1,00) foi retirada para destacar contrastes reais."
    )
    
    corr_info = theme_metrics.get("full_correlation_matrix")
    if corr_info:
        LABEL_MAP_Y = {
            "idade_paciente": "Idade da Vítima",
            "numero_envolvidos": "Nº de Agressores Envolvidos",
            "ocorreu_noite_madrugada": "Ocorrido à Noite/Madrugada (18h às 6h)",
            "reside_municipio_ocorrencia": "Mora no Município do Fato",
            "local_residencia": "Ocorrência dentro da Residência",
            "gestante": "Vítima Gestante",
            "possui_deficiencia": "Vítima com Deficiência",
            "violencia_fisica": "Agressão / Lesão Física",
            "violencia_psicologica": "Violência Psicológica / Humilhação",
            "violencia_sexual": "Violência Sexual",
            "violencia_financeira": "Violência Patrimonial / Financeira",
            "violencia_negligencia": "Negligência / Abandono",
            "violencia_tortura": "Tortura",
            "meio_forca_corporal": "Uso de Força Corporal / Socos",
            "meio_enforcamento": "Tentativa de Asfixia / Enforcamento",
            "meio_objeto_contundente": "Objeto Contundente (Pedaço de pau, barra)",
            "meio_objeto_perfurante": "Objeto Cortante / Faca",
            "meio_arma_fogo": "Uso de Arma de Fogo",
            "meio_ameaca": "Ameaça Verbal",
            "autor_alcoolizado": "Agressor sob Efeito de Álcool",
            "target_reincidencia": "Reincidência (Alvo do Modelo)",
            "autor_parceiro_ou_ex": "Agressor é Parceiro/Ex (Alvo)",
            "encaminhamento_delegacia_mulher": "Encaminhada à Delegacia da Mulher (DEAM)"
        }

        LABEL_MAP_X = {
            "idade_paciente": "Idade<br>Vítima",
            "numero_envolvidos": "Nº<br>Agressores",
            "ocorreu_noite_madrugada": "Noite /<br>Madrugada",
            "reside_municipio_ocorrencia": "Mora no<br>Município",
            "local_residencia": "Dentro de<br>Casa",
            "gestante": "Vítima<br>Gestante",
            "possui_deficiencia": "Pessoa c/<br>Deficiência",
            "violencia_fisica": "Agressão<br>Física",
            "violencia_psicologica": "Violência<br>Psicológica",
            "violencia_sexual": "Violência<br>Sexual",
            "violencia_financeira": "Violência<br>Financeira",
            "violencia_negligencia": "Negligência<br>Abandono",
            "violencia_tortura": "Tortura",
            "meio_forca_corporal": "Força<br>Corporal",
            "meio_enforcamento": "Asfixia /<br>Enforc.",
            "meio_objeto_contundente": "Objeto<br>Contund.",
            "meio_objeto_perfurante": "Faca /<br>Cortante",
            "meio_arma_fogo": "Arma de<br>Fogo",
            "meio_ameaca": "Ameaça<br>Verbal",
            "autor_alcoolizado": "Agressor c/<br>Álcool",
            "target_reincidencia": "Reincidência<br>(Alvo)",
            "autor_parceiro_ou_ex": "Parceiro/Ex<br>(Alvo)",
            "encaminhamento_delegacia_mulher": "Delegacia<br>Mulher (DEAM)"
        }

        df_raw = pd.DataFrame(
            corr_info["data"],
            columns=corr_info["columns"],
            index=corr_info["index"]
        )

        target_slug = theme_metrics["slug"]
        if target_slug in df_raw.columns:
            target_series = df_raw[target_slug].drop(index=target_slug, errors="ignore")
            max_pos_col = target_series.idxmax()
            max_pos_val = target_series.max()
            max_neg_col = target_series.idxmin()
            max_neg_val = target_series.min()

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Maior Ligação Direta", f"+{max_pos_val:.2f}", LABEL_MAP_Y.get(max_pos_col, max_pos_col))
            with k2:
                st.metric("Menor Ligação / Incompatível", f"{max_neg_val:.2f}", LABEL_MAP_Y.get(max_neg_col, max_neg_col))
            with k3:
                st.metric("Faixa Efetiva de Variação", f"{max_neg_val:.2f} até +{max_pos_val:.2f}", "Escala calibrada")

        st.markdown("---")

        arr = df_raw.to_numpy(dtype=float, copy=True)
        mask_upper_and_diag = np.triu(np.ones_like(arr, dtype=bool), k=0)
        arr[mask_upper_and_diag] = np.nan

        cols_x = [LABEL_MAP_X.get(col, col) for col in df_raw.columns]
        idx_y = [LABEL_MAP_Y.get(idx, idx) for idx in df_raw.index]
        df_plot = pd.DataFrame(arr, columns=cols_x, index=idx_y)

        gemini_scale = [
            [0.0, "#f1f3f4"],
            [0.35, "#ffffff"],
            [0.55, "#d2e3fc"],
            [0.80, "#4285f4"],
            [1.0, "#1a73e8"]
        ]

        fig = px.imshow(
            df_plot,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=gemini_scale,
            range_color=[-0.35, 0.65]
        )

        fig.update_coloraxes(showscale=False)
        fig.update_layout(
            height=760,
            xaxis=dict(tickangle=0, tickfont=dict(size=9.5, color="#3c4043"), side="bottom"),
            yaxis=dict(tickfont=dict(size=10.5, color="#3c4043")),
            margin=dict(l=40, r=20, t=10, b=80)
        )
        fig.for_each_annotation(lambda a: a.update(text="") if a.text == "nan" else ())

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Entenda os Termos e Variáveis da Base de Dados (SINAN)"):
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("""
                **Termos Institucionais e Locais**
                * Delegacia da Mulher (DEAM): Encaminhamento oficial para a Delegacia Especializada de Atendimento à Mulher da Polícia Civil.
                * Noite/Madrugada (18h-6h): Episódio praticado durante o horário noturno.
                * Dentro de Casa: Ocorrido no domicílio habitual da vítima ou do agressor.
                * Mora no Município: Vítima reside na mesma cidade em que a agressão foi registrada.
                
                **Perfil e Vulnerabilidades**
                * Vítima Gestante: Confirmação na ficha médica de gravidez no momento do atendimento.
                * Pessoa com Deficiência: Presença de limitação física, auditiva, visual ou motora.
                """)
            with g2:
                st.markdown("""
                **Tipos de Agressão e Instrumentos**
                * Agressão Física: Agressão direta geradora de lesões, hematomas ou escoriações.
                * Força Corporal: Agressão cometida por espancamento, socos ou chutes.
                * Asfixia / Enforcamento: Tentativa de estrangulamento ou sufocamento (alto risco de vida).
                * Objeto Contundente: Golpe com pau, barra de ferro ou pedra.
                * Objeto Cortante: Faca, estilete ou tesoura.
                * Agressor com Álcool: Odor etílico ou confirmação de consumo alcoólico pelo autor.
                """)

        with st.expander("Visualizar Matriz Completa em Tabela"):
            st.dataframe(
                df_raw.rename(columns=LABEL_MAP_Y, index=LABEL_MAP_Y).style.format(precision=2),
                use_container_width=True
            )
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
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_sample_idx = st.selectbox(
            "Carregue uma ocorrência real do conjunto de teste ou mantenha em preenchimento livre:",
            options=range(len(sample_options)),
            format_func=lambda x: sample_options[x]
        )
    
    is_sample = selected_sample_idx > 0
    sample_val = samples_list[selected_sample_idx - 1] if is_sample else None

    # Trava de edição ativa por padrão quando um caso real está selecionado
    bloquear_edicao = False
    if is_sample:
        with col_sel2:
            st.write("")
            st.write("")
            bloquear_edicao = st.checkbox("Bloquear campos", value=True, help="Impede modificações acidentais no caso real.")

    def get_val(key, default):
        if is_sample and key in sample_val:
            return sample_val[key]
        return default

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Perfil da Vítima**")
        idade = st.number_input("Idade", 14, 105, int(get_val("idade_paciente", 28)), disabled=bloquear_edicao)
        gestante = st.selectbox("Gestante?", [0, 1], index=int(get_val("gestante", 0)), format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        
        raca_opts = ["1", "2", "3", "4", "5", "ignorado"]
        cur_raca = str(get_val("raca_paciente", "1"))
        raca_idx = raca_opts.index(cur_raca) if cur_raca in raca_opts else 0
        raca = st.selectbox("Raça/Cor", raca_opts, index=raca_idx, format_func=lambda x: {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(x, "Ignorado"), disabled=bloquear_edicao)
        
        esc_opts = ["1", "2", "3", "4", "5", "6", "ignorado"]
        cur_esc = str(get_val("escolaridade_paciente", "4"))
        esc_idx = esc_opts.index(cur_esc) if cur_esc in esc_opts else 0
        escolaridade = st.selectbox("Escolaridade", esc_opts, index=esc_idx, format_func=lambda x: {"1": "Fund. Incompleto", "2": "Fund. Completo", "3": "Médio Incompleto", "4": "Médio Completo", "5": "Superior Incompleto", "6": "Superior Completo"}.get(x, "Ignorado"), disabled=bloquear_edicao)
        
        civ_opts = ["1", "2", "3", "4", "ignorado"]
        cur_civ = str(get_val("estado_civil_paciente", "1"))
        civ_idx = civ_opts.index(cur_civ) if cur_civ in civ_opts else 0
        estado_civil = st.selectbox("Estado Civil", civ_opts, index=civ_idx, format_func=lambda x: {"1": "Solteira", "2": "Casada/União", "3": "Viúva", "4": "Separada"}.get(x, "Ignorado"), disabled=bloquear_edicao)
        
        deficiencia = st.selectbox("Possui Deficiência?", [0, 1], index=int(get_val("possui_deficiencia", 0)), format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        
    with c2:
        st.markdown("**Circunstâncias da Ocorrência**")
        uf_opts = ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "PA", "SC", "GO", "MA", "PB", "ES", "AM", "RN", "AL", "PI", "MT", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"]
        cur_uf = str(get_val("uf_ocorrencia", "SP"))
        uf_idx = uf_opts.index(cur_uf) if cur_uf in uf_opts else 0
        uf = st.selectbox("UF", uf_opts, index=uf_idx, disabled=bloquear_edicao)
        
        dia_semana = st.selectbox("Dia da Semana", [1, 2, 3, 4, 5, 6, 7], index=int(get_val("dia_semana_ocorrencia", 1)) - 1, format_func=lambda x: {1: "Dom", 2: "Seg", 3: "Ter", 4: "Qua", 5: "Qui", 6: "Sex", 7: "Sáb"}[x], disabled=bloquear_edicao)
        noite = st.selectbox("Período Noturno (18h-06h)?", [0, 1], index=int(get_val("ocorreu_noite_madrugada", 0)), format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        mesmo_mun = st.selectbox("Reside no mesmo município?", [1, 0], index=0 if get_val("reside_municipio_ocorrencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        casa = st.selectbox("Ocorreu na Residência?", [1, 0], index=0 if get_val("local_residencia", 1) == 1 else 1, format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        num_env = st.number_input("Número de Agressores", 1, 10, int(get_val("numero_envolvidos", 1)), disabled=bloquear_edicao)
        
        sex_opts = ["M", "F", "Ambos", "ignorado"]
        cur_sex = str(get_val("autor_sexo", "M"))
        sex_idx = sex_opts.index(cur_sex) if cur_sex in sex_opts else 0
        autor_sexo = st.selectbox("Sexo do Agressor", sex_opts, index=sex_idx, disabled=bloquear_edicao)
        
        alcool = st.selectbox("Suspeita de Uso de Álcool?", [0, 1], index=int(get_val("autor_alcoolizado", 0)), format_func=lambda x: "Sim" if x == 1 else "Não", disabled=bloquear_edicao)
        
    with c3:
        st.markdown("**Tipologia e Meios Empregados**")
        v_fisica = st.checkbox("Violência Física", bool(get_val("violencia_fisica", True)), disabled=bloquear_edicao)
        v_psico = st.checkbox("Violência Psicológica", bool(get_val("violencia_psicologica", True)), disabled=bloquear_edicao)
        v_sexual = st.checkbox("Violência Sexual", bool(get_val("violencia_sexual", False)), disabled=bloquear_edicao)
        v_financ = st.checkbox("Violência Financeira", bool(get_val("violencia_financeira", False)), disabled=bloquear_edicao)
        v_neglig = st.checkbox("Negligência / Abandono", bool(get_val("violencia_negligencia", False)), disabled=bloquear_edicao)
        v_tortura = st.checkbox("Tortura", bool(get_val("violencia_tortura", False)), disabled=bloquear_edicao)
        m_forca = st.checkbox("Força Corporal / Espancamento", bool(get_val("meio_forca_corporal", True)), disabled=bloquear_edicao)
        m_ameaca = st.checkbox("Ameaça Verbal", bool(get_val("meio_ameaca", True)), disabled=bloquear_edicao)
        m_perfurante = st.checkbox("Objeto Perfurante / Faca", bool(get_val("meio_objeto_perfurante", False)), disabled=bloquear_edicao)
        m_contundente = st.checkbox("Objeto Contundente", bool(get_val("meio_objeto_contundente", False)), disabled=bloquear_edicao)
        m_enforcamento = st.checkbox("Enforcamento / Estrangulamento", bool(get_val("meio_enforcamento", False)), disabled=bloquear_edicao)
        m_arma = st.checkbox("Arma de Fogo", bool(get_val("meio_arma_fogo", False)), disabled=bloquear_edicao)

    # Detecção de divergências em relação aos valores originais do caso real
    campos_alterados = []
    if is_sample:
        conferencias = [
            (int(idade) != int(sample_val.get("idade_paciente", 0)), "Idade"),
            (int(gestante) != int(sample_val.get("gestante", 0)), "Gestante"),
            (str(raca) != str(sample_val.get("raca_paciente", "")), "Raça"),
            (str(escolaridade) != str(sample_val.get("escolaridade_paciente", "")), "Escolaridade"),
            (str(estado_civil) != str(sample_val.get("estado_civil_paciente", "")), "Estado Civil"),
            (int(deficiencia) != int(sample_val.get("possui_deficiencia", 0)), "Deficiência"),
            (str(uf) != str(sample_val.get("uf_ocorrencia", "")), "UF"),
            (int(dia_semana) != int(sample_val.get("dia_semana_ocorrencia", 0)), "Dia da Semana"),
            (int(noite) != int(sample_val.get("ocorreu_noite_madrugada", 0)), "Período Noturno"),
            (int(mesmo_mun) != int(sample_val.get("reside_municipio_ocorrencia", 0)), "Mesmo Município"),
            (int(casa) != int(sample_val.get("local_residencia", 0)), "Na Residência"),
            (int(num_env) != int(sample_val.get("numero_envolvidos", 1)), "Nº Agressores"),
            (str(autor_sexo) != str(sample_val.get("autor_sexo", "")), "Sexo do Autor"),
            (int(alcool) != int(sample_val.get("autor_alcoolizado", 0)), "Álcool"),
            (int(v_fisica) != int(sample_val.get("violencia_fisica", 0)), "Violência Física"),
            (int(v_psico) != int(sample_val.get("violencia_psicologica", 0)), "Violência Psicológica"),
            (int(v_sexual) != int(sample_val.get("violencia_sexual", 0)), "Violência Sexual"),
            (int(v_financ) != int(sample_val.get("violencia_financeira", 0)), "Violência Financeira"),
            (int(v_neglig) != int(sample_val.get("violencia_negligencia", 0)), "Negligência"),
            (int(v_tortura) != int(sample_val.get("violencia_tortura", 0)), "Tortura"),
            (int(m_forca) != int(sample_val.get("meio_forca_corporal", 0)), "Força Corporal"),
            (int(m_ameaca) != int(sample_val.get("meio_ameaca", 0)), "Ameaça"),
            (int(m_perfurante) != int(sample_val.get("meio_objeto_perfurante", 0)), "Objeto Perfurante"),
            (int(m_contundente) != int(sample_val.get("meio_objeto_contundente", 0)), "Objeto Contundente"),
            (int(m_enforcamento) != int(sample_val.get("meio_enforcamento", 0)), "Enforcamento"),
            (int(m_arma) != int(sample_val.get("meio_arma_fogo", 0)), "Arma de Fogo")
        ]
        campos_alterados = [label for alterado, label in conferencias if alterado]

    # Feedback dinâmico do estado do formulário
    if is_sample:
        desfecho_txt = "POSITIVO (1)" if sample_val["target_real"] == 1 else "NEGATIVO (0)"
        if len(campos_alterados) == 0:
            st.info(f"Registro Real Carregado (Fidedigno). Desfecho documentado originalmente na ficha do SINAN: {desfecho_txt}")
        else:
            st.warning(
                f"Modo Customizado Ativo: Você alterou os parâmetros originais deste caso ({', '.join(campos_alterados)}). "
                f"A verificação contra o desfecho histórico de teste foi desativada para manter a integridade métrica."
            )

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
            if is_sample and len(campos_alterados) == 0:
                real_val = sample_val["target_real"]
                acertou = pred_label == real_val
                st.metric(
                    label="Desfecho Real do Caso",
                    value="Positivo (1)" if real_val == 1 else "Negativo (0)",
                    delta="Acerto do Modelo" if acertou else "Erro de Classificação",
                    delta_color="normal" if acertou else "inverse"
                )
            elif is_sample and len(campos_alterados) > 0:
                st.metric(
                    label="Validação Histórica",
                    value="Suspensa",
                    delta="Dados Modificados",
                    delta_color="off"
                )
            else:
                st.metric(
                    label="Tipo de Entrada",
                    value="Simulação Livre",
                    delta="Sem histórico",
                    delta_color="off"
                )

        if m_arma or m_enforcamento:
            st.warning(
                "Atenção - Risco Crítico de Letalidade: "
                "O emprego de arma de fogo ou asfixia é o maior preditor isolado de feminicídio. "
                "Em bases hospitalares como o SINAN, a letalidade imediata tende a truncar o histórico ambulatorial, "
                "gerando uma probabilidade de reincidência contraintuitivamente menor devido ao viés de sobrevivência."
            )
