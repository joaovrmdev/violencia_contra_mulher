# app.py — v3
"""Avaliação, interpretação e simulação dos modelos de violência contra a mulher (SINAN).

Arquivos esperados no mesmo diretório (gerados pelo notebook treino_modelos_v3.ipynb):
  metrics.json · test_samples.json · model_<alvo>_lr.joblib · model_<alvo>_tree.joblib · features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import features as F

st.set_page_config(
    page_title="Violência contra a Mulher · Modelos SINAN",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Feminism_symbol.svg/512px-Feminism_symbol.svg.png",
    layout="wide",
)
BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Tokens visuais (paleta validada para daltonismo: azul/laranja nas séries)
# ---------------------------------------------------------------------------
C = {
    "lr": "#2a78d6", "gbdt": "#eb6834", "v2": "#898781",
    "pos": "#6250d6", "neg": "#a9a79f",            # desfecho sim / não
    "up": "#e34948", "down": "#2a78d6", "mid": "#f0efec",   # divergente (aumenta / reduz)
    "muted": "#898781", "hair": "rgba(137,135,129,0.35)",
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ALGOS = ["lr", "gbdt"]
ALGO_NAME = {"lr": "Regressão logística", "gbdt": "Gradient boosting"}
ALGO_TAG = {"lr": "LR", "gbdt": "GBDT"}
SCEN_INFO = {
    "target_reincidencia": ("Reincidência", "A violência já havia acontecido outras vezes?"),
    "autor_parceiro_ou_ex": ("Agressor íntimo", "O agressor é parceiro ou ex-parceiro?"),
    "encaminhamento_delegacia_mulher": ("Encaminhamento DEAM", "O caso foi encaminhado à Delegacia da Mulher?"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

# Cards espelhados: brancos translúcidos, com reflexo diagonal e sombra suave; afundam no hover.
GLASS = """
  position:relative;
  background: linear-gradient(125deg, rgba(255,255,255,1) 0%, rgba(255,255,255,.86) 40%, rgba(255,255,255,.58) 100%);
  border:1px solid rgba(24,24,48,.07);
  border-radius:14px;
  box-shadow:
    0 12px 26px -16px rgba(30,30,60,.28),
    0 2px 6px -2px rgba(30,30,60,.06),
    inset 0 1px 0 #fff;
  transition: transform .2s cubic-bezier(.2,.7,.3,1), box-shadow .2s ease, background .2s ease;
"""
GLASS_PRESSED = """
  transform: translateY(2px) scale(.988);
  background: linear-gradient(125deg, rgba(255,255,255,.78) 0%, rgba(255,255,255,.62) 40%, rgba(255,255,255,.42) 100%);
  box-shadow:
    0 2px 6px -4px rgba(30,30,60,.25),
    inset 0 2px 9px rgba(30,30,60,.09);
"""

st.markdown(f"""
<style>
  .tile, .cm .cell, .pred, .note {{ {GLASS} }}
  .tile:hover, .cm .cell:hover, .pred:hover {{ {GLASS_PRESSED} }}
  @media (prefers-reduced-motion: reduce) {{
    .tile, .cm .cell, .pred {{ transition: none; }}
    .tile:hover, .cm .cell:hover, .pred:hover {{ transform: none; }}
  }}
  .block-container {{ padding-top: 3.4rem; max-width: 1320px; }}
  .ct {{ font-weight: 650; font-size: .98rem; margin: .5rem 0 0; }}
  .ct-sub {{ font-size: .8rem; opacity: .7; margin: 0 0 -.3rem; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .hero {{ display:flex; flex-direction:column; gap:.35rem; margin-bottom:.6rem; }}
  .hero h1 {{ font-size: 1.85rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.2; }}
  .hero p {{ margin: 0; opacity: .78; font-size: .98rem; max-width: 900px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.35rem; }}
  .chip {{ font-size:.78rem; padding:.18rem .6rem; border-radius:999px; border:1px solid {C['hair']};
           opacity:.9; white-space:nowrap; }}
  .chip b {{ font-weight:600; }}
  .tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin: .5rem 0 1.1rem; }}
  .tile {{ padding:15px 18px; }}
  .tile .lab {{ font-size:.8rem; opacity:.72; }}
  .tile .val {{ font-size:1.65rem; font-weight:650; line-height:1.25; margin-top:2px; }}
  .tile .sub {{ font-size:.78rem; opacity:.68; margin-top:2px; }}
  .tile .delta-up {{ color:#0b8a0b; font-weight:600; }}
  .tile .delta-down {{ color:#c23a3a; font-weight:600; }}
  .key {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:baseline; }}
  .cm {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:10px; }}
  .cm .cell {{ padding:14px 16px; }}
  .cm .cell .t {{ font-size:.82rem; font-weight:600; opacity:.8; display:flex; align-items:center; gap:7px; }}
  .cm .cell .t::before {{ content:""; width:8px; height:8px; border-radius:50%; flex:none;
                          box-shadow: 0 0 0 3px rgba(255,255,255,.7); }}
  .cm .ok .t::before {{ background:#1baf7a; }}
  .cm .err .t::before {{ background:#e34948; }}
  .cm .cell .v {{ font-size:1.5rem; font-weight:650; margin-top:2px; }}
  .cm .cell .d {{ font-size:.76rem; opacity:.68; }}
  .note {{ border-radius:14px; padding:11px 16px 11px 34px; font-size:.9rem; margin:.4rem 0 .8rem; }}
  .note::before {{ content:""; position:absolute; left:15px; top:calc(11px + .8em - 4px); width:8px; height:8px; border-radius:50%;
                   background:{C['pos']}; box-shadow: 0 0 0 3px rgba(255,255,255,.7); }}
  .note.warn::before {{ background:#eda100; }}
  .note.gray::before {{ background:{C['muted']}; }}
  .note b {{ font-weight:650; }}
  .pred {{ padding:16px 18px; }}
  .pred.sel {{ border:1.5px solid var(--accent); }}
  .pred .top {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
  .pred .name {{ font-size:.85rem; font-weight:600; }}
  .pred .p {{ font-size:2rem; font-weight:700; line-height:1.1; }}
  .pred .cls {{ font-size:.82rem; opacity:.8; }}
  .bullet {{ position:relative; height:12px; border-radius:6px; background:rgba(127,127,127,.15); margin:12px 0 6px; }}
  .bullet .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:6px; }}
  .bullet .thr {{ position:absolute; top:-5px; bottom:-5px; width:3px; background:#2b2a27; border-radius:2px; }}
  .bullet .prev {{ position:absolute; top:-3px; bottom:-3px; width:2px; background:{C['muted']}; }}
  .legend-mini {{ font-size:.74rem; opacity:.7; display:flex; gap:14px; flex-wrap:wrap; }}
  .feat {{ display:inline-block; font-size:.78rem; padding:.15rem .55rem; margin:.12rem .12rem; border-radius:999px;
           border:1px solid {C['hair']}; }}
  .feat.on {{ background:rgba(98,80,214,.08); border-color:rgba(98,80,214,.35); }}
  .footer {{ font-size:.8rem; opacity:.7; margin-top:2rem; border-top:1px solid {C['hair']}; padding-top:.8rem; }}
  div[data-testid="stTabs"] button p {{ font-size: .95rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def br(x, d=0):
    """Número no padrão brasileiro (1.234,5)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, d=1):
    return "–" if x is None else br(100 * x, d) + "%"


def signed(x, d=3):
    return ("+" if x >= 0 else "−") + br(abs(x), d)


def natural_key(code: str):
    if code == F.MISSING:
        return (2, 0, "")
    try:
        return (0, float(code), "")
    except ValueError:
        return (1, 0, code)


def plot(fig, key=None):
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    if meta.get("title"):
        st.markdown(f'<div class="ct">{meta["title"]}</div>', unsafe_allow_html=True)
    try:
        st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config={"displaylogo": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", key=key, config={"displaylogo": False})


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def layout(fig, height=360, title=None, legend=True, **kw):
    # O título vai para um <div> acima do gráfico (evita colisão com a legenda)
    fig.update_layout(
        height=height, font=dict(family=FONT, size=12.5), meta={"title": title} if title else None,
        margin=dict(l=10, r=16, t=12, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", title=None,
                    font=dict(size=12)) if legend else None,
        showlegend=legend, hoverlabel=dict(font=dict(family=FONT)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    return fig


def tiles_html(items):
    out = []
    for it in items:
        delta = it.get("delta")
        dh = ""
        if delta is not None:
            cls = "delta-up" if delta[1] else "delta-down"
            dh = f' <span class="{cls}">{delta[0]}</span>'
        out.append(f'<div class="tile"><div class="lab">{it["label"]}</div><div class="val">{it["value"]}</div>'
                   f'<div class="sub">{it.get("sub", "")}{dh}</div></div>')
    st.markdown(f'<div class="tiles">{"".join(out)}</div>', unsafe_allow_html=True)


def note(html, kind=""):
    st.markdown(f'<div class="note {kind}">{html}</div>', unsafe_allow_html=True)


def flabel(c):
    return F.label(c)


# ---------------------------------------------------------------------------
# Artefatos
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def merc(lat):
    """Projeção de Mercator (em graus), para o mapa não ficar achatado."""
    return float(np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))))


def uf_outline(geom):
    """Contorno de um estado (Polygon/MultiPolygon do GeoJSON) em listas x/y separadas por None."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    xs, ys = [], []
    for poly in polys:
        ring = poly[0]
        xs += [p[0] for p in ring] + [None]
        ys += [merc(p[1]) for p in ring] + [None]
    return xs, ys


@st.cache_resource
def load_model(fname):
    return joblib.load(BASE / fname)


try:
    METRICS = load_json("metrics.json")
    SAMPLES = load_json("test_samples.json")
except Exception as e:  # noqa: BLE001
    st.error(f"Não consegui ler os arquivos gerados pelo treino: {e}")
    st.info("Rode o notebook `treino_modelos_v3.ipynb` e coloque o conteúdo do `modelos_v3.zip` na pasta do app.")
    st.stop()
if "scenarios" not in METRICS:
    st.error("Este `metrics.json` é da versão anterior (v2). Gere os artefatos novos com o notebook `treino_modelos_v3.ipynb`.")
    st.stop()

META = METRICS["meta"]
SCEN = METRICS["scenarios"]
LABELS = list(SCEN)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Cenário")
    scen_label = st.radio(
        "Cenário", LABELS, label_visibility="collapsed",
        format_func=lambda l: SCEN_INFO[SCEN[l]["slug"]][0],
        captions=[SCEN_INFO[SCEN[l]["slug"]][1] for l in LABELS],
    )
    st.markdown("#### Modelo em foco")
    algo = None
    if hasattr(st, "segmented_control"):
        algo = st.segmented_control("Modelo", ALGOS, default="gbdt", format_func=ALGO_NAME.get,
                                    label_visibility="collapsed", key="algo")
    if algo is None:
        algo = st.radio("Modelo", ALGOS, index=1, format_func=ALGO_NAME.get, label_visibility="collapsed",
                        horizontal=True, key="algo_radio") if not hasattr(st, "segmented_control") else "gbdt"
    st.caption("O modelo em foco define o limiar, a importância e a explicação local. Os gráficos comparativos "
               "mostram os dois modelos.")
    st.divider()
    st.caption(f"Treino v3 · {META.get('generated_at', '')[:10]} · scikit-learn {META.get('sklearn_version')}")
    if META.get("modo_rapido"):
        st.warning("Artefatos gerados em MODO_RAPIDO (amostras pequenas). Rode o notebook completo.")

S = SCEN[scen_label]
SLUG = S["slug"]
M = S["models"]
MA = M[algo]
SHORT, QUESTION = SCEN_INFO[SLUG]
SHORT_LC = SHORT[0].lower() + SHORT[1:]  # "encaminhamento DEAM", sem baixar a sigla

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
yrs = META.get("years") or [None, None]
st.markdown(f"""
<div class="hero">
  <h1>Violência contra a mulher · modelos preditivos</h1>
  <p>Modelos supervisionados treinados com <b>{br(META['n_rows_csv'])}</b> notificações de violência interpessoal
  contra mulheres de 14 anos ou mais no SINAN ({yrs[0]}–{yrs[1]}). Ferramenta de estudo — não substitui avaliação
  profissional nem deve orientar decisões sobre casos individuais.</p>
  <div class="chips">
    <span class="chip">Cenário: <b>{SHORT}</b></span>
    <span class="chip">{QUESTION}</span>
    <span class="chip">{br(S['n_total'])} casos com desfecho informado · {S['years'][0]}–{S['years'][1]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_over, tab_perf, tab_feat, tab_assoc, tab_fair, tab_sim, tab_meth = st.tabs([
    "Visão geral", "Desempenho", "Variáveis", "Associações", "Equidade", "Simulador", "Metodologia"])

# =========================================================================== #
# 1. VISÃO GERAL
# =========================================================================== #
with tab_over:
    best = max(ALGOS, key=lambda a: M[a]["test"]["roc_auc"])
    b_auc = M[best]["test"]["roc_auc"]
    v2 = M[best].get("baseline_v2", {})
    d = b_auc - v2["roc_auc"] if v2 else None
    tiles_html([
        {"label": "Casos no cenário", "value": br(S["n_total"]),
         "sub": f"treino {br(S['n_train'])} · teste {br(S['n_test'])}"},
        {"label": "Prevalência do desfecho", "value": pct(S["prevalence_test"]), "sub": "no conjunto de teste"},
        {"label": f"Melhor AUC no teste · {ALGO_TAG[best]}", "value": br(b_auc, 3),
         "sub": "vs v2 nos mesmos dados:", "delta": (signed(d), d >= 0) if d is not None else None},
        {"label": "Variáveis usadas", "value": f"{len(M['lr']['selected_features'])} · {len(M['gbdt']['selected_features'])}",
         "sub": f"LR · GBDT — de {len(S['candidates'])} candidatas (v2: 26)"},
    ])

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        # Dumbbell v2 -> v3 para os 6 modelos
        fig = go.Figure()
        ycat = [[], []]
        for lab in LABELS:
            for a in ALGOS:
                ycat[0].append(SCEN_INFO[SCEN[lab]["slug"]][0])
                ycat[1].append(ALGO_TAG[a])
        v2v = [SCEN[l]["models"][a].get("baseline_v2", {}).get("roc_auc") for l in LABELS for a in ALGOS]
        v3v = [SCEN[l]["models"][a]["test"]["roc_auc"] for l in LABELS for a in ALGOS]
        for i in range(len(v3v)):
            if v2v[i] is not None:
                fig.add_trace(go.Scatter(x=[v2v[i], v3v[i]], y=[[ycat[0][i]] * 2, [ycat[1][i]] * 2], mode="lines",
                                         line=dict(color=C["hair"], width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=v2v, y=ycat, mode="markers", name="v2 (26 variáveis, mesmos dados)",
                                 marker=dict(size=11, color=C["v2"], line=dict(width=2, color="white")),
                                 hovertemplate="v2 · AUC %{x:.3f}<extra></extra>"))
        for a in ALGOS:
            idx = [i for i, t in enumerate(ycat[1]) if t == ALGO_TAG[a]]
            fig.add_trace(go.Scatter(
                x=[v3v[i] for i in idx], y=[[ycat[0][i] for i in idx], [ycat[1][i] for i in idx]],
                mode="markers+text", name=f"v3 · {ALGO_NAME[a]}", text=[br(v3v[i], 3) for i in idx],
                textposition="middle right", textfont=dict(size=11.5),
                marker=dict(size=12, color=C[a], line=dict(width=2, color="white")),
                hovertemplate="v3 · AUC %{x:.3f}<extra></extra>"))
        xmin = min(v for v in v2v + v3v if v is not None)
        layout(fig, 360, "AUC no teste: v2 × v3 (mesma divisão dos dados)")
        fig.update_xaxes(range=[xmin - 0.02, max(v3v) + 0.045], tickformat=".2f", showgrid=True)
        fig.update_yaxes(autorange="reversed")
        plot(fig, "dumbbell")
        st.caption("A v2 foi retreinada nos dados novos com a configuração original (26 variáveis, "
                   "class_weight='balanced'), para a comparação ser justa.")
    with c2:
        rby = S.get("rate_by_year", {})
        if rby:
            yy = sorted(int(k) for k in rby)
            rows_year = META.get("rows_per_year", {})
            ns = [rows_year.get(str(y), rows_year.get(y)) for y in yy]
            low = [n is not None and n < 15000 for n in ns]
            fig = go.Figure(go.Scatter(
                x=yy, y=[rby[str(y)] if str(y) in rby else rby[y] for y in yy], mode="lines+markers",
                line=dict(color=C["pos"], width=2), name="taxa",
                marker=dict(size=[9 if l else 7 for l in low], color=["white" if l else C["pos"] for l in low],
                            line=dict(color=C["pos"], width=2)),
                customdata=[br(n) if n else "–" for n in ns],
                hovertemplate="%{x}: %{y:.1%}<br>%{customdata} notificações no ano<extra></extra>"))
            layout(fig, 360, f"Taxa do desfecho por ano — {SHORT_LC}", legend=False)
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(dtick=2)
            plot(fig, "rate_year")
            if any(low):
                st.caption("Marcadores vazados: anos com menos de 15 mil notificações na base (2013–2014 estão "
                           "incompletos no reprocessamento atual da Base dos Dados).")

    # Leitura automática
    k_lr, k_gb = len(M["lr"]["selected_features"]), len(M["gbdt"]["selected_features"])
    top_feat = [flabel(r["feature"]) for r in M[best]["importance"][:3]]
    temporal = M[best].get("temporal")
    bullets = [
        f"O melhor modelo é o **{ALGO_NAME[best]}**, com AUC de **{br(b_auc, 3)}** no teste"
        + (f" (**{signed(d)}** em relação à v2 nos mesmos dados)." if d is not None else "."),
        f"A seleção reduziu as variáveis para **{k_lr}** (LR) e **{k_gb}** (GBDT), de {len(S['candidates'])} candidatas.",
        f"O que mais pesa no {ALGO_TAG[best]}: {', '.join(top_feat)}.",
    ]
    if temporal:
        bullets.append(f"Na validação temporal (treino {temporal['train_years']}, teste {temporal['test_years']}) a AUC "
                       f"foi **{br(temporal['auc'], 3)}**, contra {br(b_auc, 3)} na divisão aleatória.")
    st.markdown("**Leitura rápida**\n\n" + "\n".join(f"- {b}" for b in bullets))

# =========================================================================== #
# 2. DESEMPENHO
# =========================================================================== #
with tab_perf:
    hist = MA["curves"]["score_hist"]
    pos_h, neg_h = np.array(hist["pos"], dtype=float), np.array(hist["neg"], dtype=float)
    thr_default = float(np.round(MA["threshold"], 2))
    key_thr = f"thr_{SLUG}_{algo}"

    def _reset_thr():
        st.session_state[key_thr] = thr_default

    st.markdown(f"##### Limiar de decisão · {ALGO_NAME[algo]}")
    cA, cB = st.columns([3, 1])
    with cA:
        thr = st.slider("Probabilidade a partir da qual o caso é classificado como positivo", 0.01, 0.99,
                        value=thr_default, step=0.01, key=key_thr)
    with cB:
        st.write("")
        st.button("Voltar ao limiar recomendado", on_click=_reset_thr, key=f"reset_{key_thr}")
    st.caption(f"Limiar recomendado = {br(MA['threshold'], 3)} (máximo do índice de Youden na validação interna). "
               "Métricas abaixo recalculadas no conjunto de teste em passos de 0,01.")

    i = int(round(thr * 100))
    TP, FN, FP, TN = pos_h[i:].sum(), pos_h[:i].sum(), neg_h[i:].sum(), neg_h[:i].sum()
    N = TP + FN + FP + TN
    rec = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    prec = TP / max(TP + FP, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    tiles_html([
        {"label": "Sensibilidade (recall)", "value": pct(rec), "sub": "positivos reais detectados"},
        {"label": "Especificidade", "value": pct(spec), "sub": "negativos reais descartados"},
        {"label": "Precisão (VPP)", "value": pct(prec), "sub": "alertas que eram positivos"},
        {"label": "F1", "value": br(f1, 3), "sub": "média harmônica precisão × recall"},
        {"label": "Casos sinalizados", "value": pct((TP + FP) / N), "sub": f"prevalência real {pct(S['prevalence_test'])}"},
    ])

    c1, c2 = st.columns([1, 1.35], gap="large")
    with c1:
        st.markdown(f"""
<div class="cm">
  <div class="cell ok"><div class="t">Verdadeiro negativo</div><div class="v">{br(TN)}</div>
    <div class="d">{pct(TN / N)} do teste · previu não, era não</div></div>
  <div class="cell err"><div class="t">Falso positivo</div><div class="v">{br(FP)}</div>
    <div class="d">{pct(FP / N)} · alarme sem desfecho</div></div>
  <div class="cell err"><div class="t">Falso negativo</div><div class="v">{br(FN)}</div>
    <div class="d">{pct(FN / N)} · desfecho não detectado</div></div>
  <div class="cell ok"><div class="t">Verdadeiro positivo</div><div class="v">{br(TP)}</div>
    <div class="d">{pct(TP / N)} · previu sim, era sim</div></div>
</div>""", unsafe_allow_html=True)
        st.caption("Linhas = desfecho real (não / sim); colunas = previsão (não / sim).")
    with c2:
        mids = (np.array(hist["bin_edges"][:-1]) + np.array(hist["bin_edges"][1:])) / 2
        fig = go.Figure()
        for arr, name, col in [(neg_h, "Desfecho real = não", C["neg"]), (pos_h, "Desfecho real = sim", C["pos"])]:
            dens = arr / max(arr.sum(), 1)
            fig.add_trace(go.Scatter(x=mids, y=dens, mode="lines", line=dict(color=col, width=2, shape="hvh"),
                                     fill="tozeroy", fillcolor=col + "1f", name=name,
                                     hovertemplate="prob. %{x:.2f}: %{y:.1%} dos casos<extra>" + name + "</extra>"))
        fig.add_vline(x=thr, line=dict(color=C["muted"], width=1.5))
        fig.add_annotation(x=thr, y=1, yref="paper", text=f"limiar {br(thr, 2)}", showarrow=False,
                           xanchor="left", xshift=4, font=dict(size=11, color=C["muted"]))
        layout(fig, 300, "Distribuição das probabilidades previstas no teste")
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="probabilidade prevista")
        fig.update_yaxes(tickformat=".0%", title="% dos casos da classe")
        plot(fig, "score_hist")

    st.markdown("##### Curvas dos dois modelos")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["hair"], width=1),
                                 hoverinfo="skip", showlegend=False))
        for a in ALGOS:
            r = M[a]["curves"]["roc"]
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", line=dict(color=C[a], width=2),
                                     name=f"{ALGO_TAG[a]} · AUC {br(M[a]['test']['roc_auc'], 3)}",
                                     hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[1 - spec], y=[rec], mode="markers", name="ponto de operação",
                                 marker=dict(size=11, color=C[algo], symbol="circle", line=dict(width=2, color="white")),
                                 hovertemplate="limiar atual<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
        layout(fig, 330, "ROC")
        fig.update_xaxes(title="taxa de falso positivo", range=[0, 1])
        fig.update_yaxes(title="sensibilidade", range=[0, 1.02])
        plot(fig, "roc")
    with c2:
        fig = go.Figure()
        prev = S["prevalence_test"]
        fig.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines", line=dict(color=C["hair"], width=1),
                                 hoverinfo="skip", showlegend=False))
        for a in ALGOS:
            r = M[a]["curves"]["pr"]
            fig.add_trace(go.Scatter(x=r["recall"], y=r["precision"], mode="lines", line=dict(color=C[a], width=2),
                                     name=f"{ALGO_TAG[a]} · AP {br(M[a]['test']['average_precision'], 3)}",
                                     hovertemplate="recall %{x:.2f} · precisão %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", name="ponto de operação",
                                 marker=dict(size=11, color=C[algo], line=dict(width=2, color="white")),
                                 hovertemplate="limiar atual<extra></extra>", showlegend=False))
        layout(fig, 330, "Precisão × recall")
        fig.update_xaxes(title="recall", range=[0, 1])
        fig.update_yaxes(title="precisão", range=[0, 1.02])
        plot(fig, "pr")
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C["hair"], width=1),
                                 hoverinfo="skip", showlegend=False))
        for a in ALGOS:
            cal = M[a]["curves"]["calibration"]
            fig.add_trace(go.Scatter(x=cal["mean_predicted"], y=cal["observed"], mode="lines+markers",
                                     line=dict(color=C[a], width=2), marker=dict(size=7),
                                     name=f"{ALGO_TAG[a]} · Brier {br(M[a]['test']['brier'], 3)}",
                                     hovertemplate="previsto %{x:.1%} · observado %{y:.1%}<extra></extra>"))
        layout(fig, 330, "Calibração (decis)")
        fig.update_xaxes(title="probabilidade média prevista", range=[0, 1], tickformat=".0%")
        fig.update_yaxes(title="taxa observada", range=[0, 1], tickformat=".0%")
        plot(fig, "calib")
    st.caption("Calibração: pontos sobre a diagonal indicam que uma probabilidade de 30% corresponde, de fato, a ~30% "
               "de casos positivos. Os modelos v3 são treinados sem pesos de classe, por isso a probabilidade é "
               "interpretável — na v2 (class_weight='balanced') ela era inflada.")

    st.markdown("##### Comparação detalhada")
    rows = []
    for a in ALGOS:
        t = M[a]["test"]
        r1 = t["report"]["1"]
        rows.append({"Modelo": f"v3 · {ALGO_NAME[a]}", "Nº variáveis": len(M[a]["selected_features"]),
                     "AUC": t["roc_auc"], "IC 95% AUC": f"{br(t['roc_auc_ci95'][0], 3)}–{br(t['roc_auc_ci95'][1], 3)}",
                     "AP": t["average_precision"], "Brier": t["brier"], "Limiar": t["threshold"],
                     "Recall": r1["recall"], "Precisão": r1["precision"], "F1": r1["f1-score"],
                     "AUC temporal": M[a].get("temporal", {}).get("auc")})
    for a in ALGOS:
        b = M[a].get("baseline_v2")
        if b:
            rows.append({"Modelo": f"v2 · {ALGO_NAME[a]} (mesmos dados)", "Nº variáveis": b["n_features"],
                         "AUC": b["roc_auc"], "IC 95% AUC": "", "AP": b.get("average_precision"),
                         "Brier": b.get("brier"), "Limiar": 0.5, "Recall": b["recall"], "Precisão": b["precision"],
                         "F1": b["f1"], "AUC temporal": None})
    dfc = pd.DataFrame(rows).set_index("Modelo")
    for c_ in ["AUC", "AP", "Brier", "Limiar", "Recall", "Precisão", "F1", "AUC temporal"]:
        dfc[c_] = pd.to_numeric(dfc[c_], errors="coerce")
    table(dfc.style.format({c: "{:.3f}" for c in ["AUC", "AP", "Brier", "Limiar", "Recall", "Precisão", "F1",
                                                   "AUC temporal"]}, na_rep="–"))
    tm = [M[a].get("temporal") for a in ALGOS]
    if all(tm):
        t0 = tm[0]
        st.caption(f"AUC temporal: treino com casos de {t0['train_years']} e teste em {t0['test_years']} "
                   f"({br(t0['n_test'])} casos). Taxa real no período de teste {pct(t0['prevalence_test'])}; "
                   f"média prevista {pct(tm[0]['mean_pred_test'])} (LR) e {pct(tm[1]['mean_pred_test'])} (GBDT).")

# =========================================================================== #
# 3. VARIÁVEIS
# =========================================================================== #
with tab_feat:
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        fig = go.Figure()
        for a in ALGOS:
            steps = M[a]["selection"]["steps"]
            ks = [s["k"] for s in steps]
            aucs = [s["auc"] for s in steps]
            removed = [", ".join(flabel(f) for f in s.get("removed", [])) or "–" for s in steps]
            fig.add_trace(go.Scatter(x=ks, y=aucs, mode="lines", line=dict(color=C[a], width=2), name=ALGO_NAME[a],
                                     customdata=removed,
                                     hovertemplate="%{x} variáveis · AUC %{y:.4f}<br>próxima a sair: %{customdata}"
                                                   "<extra></extra>"))
            ck, ca = M[a]["selection"]["chosen_k"], M[a]["selection"]["chosen_auc"]
            fig.add_trace(go.Scatter(x=[ck], y=[ca], mode="markers+text", text=[f"{ck}"], textposition="top center",
                                     marker=dict(size=11, color=C[a], line=dict(width=2, color="white")),
                                     showlegend=False, hovertemplate=f"escolhido: {ck} variáveis<extra></extra>"))
        cut = MA["selection"]["cutoff"]
        fig.add_hline(y=cut, line=dict(color=C[algo], width=1, dash="dot"))
        top_auc = max(s_["auc"] for a in ALGOS for s_ in M[a]["selection"]["steps"])
        layout(fig, 380, "Curva de seleção: AUC na validação × nº de variáveis")
        fig.update_xaxes(autorange="reversed", title="nº de variáveis no modelo")
        fig.update_yaxes(title="AUC (validação interna)", tickformat=".3f", range=[top_auc - 0.05, top_auc + 0.006])
        plot(fig, "rfe")
        st.caption("Da esquerda para a direita, a cada passo sai a variável menos importante (por permutação). "
                   "Fica o menor conjunto cuja AUC não é estatisticamente pior que a melhor (regra de 1 erro-padrão; "
                   f"linha pontilhada = corte do {ALGO_TAG[algo]}). "
                   "Eixo vertical recortado perto do topo; passe o mouse para ver a AUC dos modelos com poucas variáveis.")
    with c2:
        imp = pd.DataFrame(MA["importance"]).sort_values("mean")
        fig = go.Figure(go.Bar(
            x=imp["mean"], y=[flabel(f) for f in imp["feature"]], orientation="h",
            marker=dict(color=C[algo], cornerradius=4), width=0.62,
            error_x=dict(type="data", array=imp["std"], color=C["muted"], thickness=1, width=0),
            text=[signed(v, 3) for v in imp["mean"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: queda de AUC %{x:.4f}<extra></extra>"))
        layout(fig, max(300, 26 * len(imp) + 80), f"Importância por permutação · {ALGO_NAME[algo]}", legend=False)
        fig.update_xaxes(title="queda na AUC ao embaralhar a variável", tickformat=".3f")
        fig.update_layout(margin=dict(r=48))
        plot(fig, "imp")

    # Matriz de seleção: quais variáveis cada um dos 6 modelos usa
    st.markdown("##### Quais variáveis cada modelo usa")
    feats_order = [c for g, cols in F.GROUPS.items() for c in cols if c in F.ALL_CANDIDATES]
    feats_order += [c for c in F.ALL_CANDIDATES if c not in feats_order]
    xcat = [[], []]
    for lab in LABELS:
        for a in ALGOS:
            xcat[0].append(SCEN_INFO[SCEN[lab]["slug"]][0])
            xcat[1].append(ALGO_TAG[a])
    fig = go.Figure()
    # Trace invisível que fixa a ordem das linhas (grupos) e colunas (cenário × modelo)
    sx, sy = [[], []], [[], []]
    for f in feats_order:
        for x0, x1 in zip(*xcat):
            sx[0].append(x0); sx[1].append(x1)
            sy[0].append(F.GROUP_OF.get(f, "Outras")); sy[1].append(F.short_label(f))
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    for status, sym, size, color_fn, name in [
        ("sel", "circle", 12, lambda a: C[a], "usada"),
        ("not", "circle-open", 8, lambda a: C["muted"], "candidata não usada"),
        ("exc", "x-thin-open", 8, lambda a: C["muted"], "excluída (vazamento / alvo)"),
    ]:
        for a in ALGOS:
            xs, ys, cd = [[], []], [[], []], []
            for j, lab in enumerate(LABELS):
                sc = SCEN[lab]
                mm = sc["models"][a]
                impd = {r["feature"]: r["mean"] for r in mm["importance"]}
                for f in feats_order:
                    if f in mm["selected_features"]:
                        stt = "sel"
                    elif f in sc["excluded"] or f not in sc["candidates"]:
                        stt = "exc"
                    else:
                        stt = "not"
                    if stt != status:
                        continue
                    xs[0].append(SCEN_INFO[sc["slug"]][0]); xs[1].append(ALGO_TAG[a])
                    ys[0].append(F.GROUP_OF.get(f, "Outras")); ys[1].append(F.short_label(f))
                    cd.append(f"{flabel(f)}<br>{SCEN_INFO[sc['slug']][0]} · {ALGO_NAME[a]}<br>"
                              + (f"importância {impd.get(f, 0):.4f}" if stt == "sel" else name))
            if not cd:
                continue
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=name, legendgroup=status,
                                     showlegend=(a == "lr" and status != "sel") or (status == "sel"),
                                     marker=dict(symbol=sym, size=size, color=color_fn(a),
                                                 line=dict(width=1.5 if status != "sel" else 2,
                                                           color=color_fn(a) if status != "sel" else "white")),
                                     customdata=cd, hovertemplate="%{customdata}<extra></extra>"))
    for tr in fig.data:
        if tr.legendgroup == "sel":
            tr.name = f"usada · {'LR' if tr.marker.color == C['lr'] else 'GBDT'}"
    layout(fig, 24 * len(feats_order) + 120)
    fig.update_yaxes(autorange="reversed", showgrid=True, tickfont=dict(size=11.5))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_layout(legend=dict(y=-0.02, yanchor="top"))
    plot(fig, "selmatrix")

    if algo == "lr" and "coefficients" in MA:
        st.markdown("##### Coeficientes da regressão logística (razão de chances)")
        terms = pd.DataFrame(MA["coefficients"]["terms"])
        terms["absc"] = terms["coef"].abs()
        top = terms.sort_values("absc", ascending=False).head(22).sort_values("coef")
        lab_terms = [flabel(r.column) + (f" = {F.code_label(r.column, r.category)}" if r.category else "")
                     + (" (+1 DP)" if r.block == "num" else "") for r in top.itertuples()]
        fig = go.Figure()
        fig.add_vline(x=1, line=dict(color=C["muted"], width=1))
        fig.add_trace(go.Scatter(
            x=top["odds_ratio"], y=lab_terms, mode="markers",
            marker=dict(size=10, color=[C["up"] if v > 1 else C["down"] for v in top["odds_ratio"]],
                        line=dict(width=2, color="white")),
            hovertemplate="%{y}<br>razão de chances %{x:.2f}<extra></extra>", showlegend=False))
        for r, yl in zip(top.itertuples(), lab_terms):
            fig.add_shape(type="line", x0=1, x1=r.odds_ratio, y0=yl, y1=yl, line=dict(color=C["hair"], width=2), layer="below")
        layout(fig, 26 * len(top) + 90, legend=False)
        fig.update_xaxes(type="log", title="razão de chances (escala log) — vermelho aumenta, azul reduz a chance")
        plot(fig, "or")
        st.caption("Os 22 termos de maior magnitude. Categóricas usam one-hot completo com regularização L2: leia como "
                   "direção e força relativa dentro da variável, não como efeito causal.")
    elif algo == "gbdt":
        note("Modelos de árvores não têm coeficientes. Para ver <b>como</b> cada variável empurra a previsão de um caso, "
             "use a explicação local no <b>Simulador</b>.", "gray")

    with st.expander("Lista das variáveis escolhidas e ordem de eliminação"):
        for a in ALGOS:
            mm = M[a]
            chips = "".join(f'<span class="feat on">{flabel(f)}</span>' for f in mm["selected_features"])
            st.markdown(f"**{ALGO_NAME[a]}** — {len(mm['selected_features'])} variáveis<br>{chips}",
                        unsafe_allow_html=True)
            order = [f for s_ in mm["selection"]["steps"] for f in s_.get("removed", [])]
            st.caption("Eliminadas (primeira → última): " + ", ".join(flabel(f) for f in order))
        excl = {k: v for k, v in S["excluded"].items() if v != "é o alvo"}
        if excl:
            st.caption("Excluídas deste cenário: " + "; ".join(f"{flabel(k)} ({v})" for k, v in excl.items()))

# =========================================================================== #
# 4. ASSOCIAÇÕES (descritivo, dados de treino)
# =========================================================================== #
# Estados pequenos demais para o rótulo caber dentro: rótulo fora, com linha até o estado (lon, lat).
UF_LINE_FROM = {"PE": (-35.7, -8.25)}  # ponto de partida da linha quando o centro do estado fica longe da costa
UF_CALLOUT = {"RN": (-33.9, -5.0), "PB": (-33.6, -6.7), "PE": (-33.6, -8.4), "AL": (-34.0, -10.1),
              "SE": (-35.4, -11.9), "ES": (-38.4, -20.3), "RJ": (-40.6, -23.9), "DF": (-49.9, -14.0)}
ORDINAL_COLS = {"faixa_etaria", "escolaridade_paciente", "mes_ocorrencia", "dia_semana_ocorrencia", "faixa_horaria",
                "numero_envolvidos"}
FAIXA_ORDER = {"madrugada": 0, "manha": 1, "tarde": 2, "noite": 3}


def ord_key(var, code):
    """Ordem natural das categorias ordinais (ignorado sempre por último)."""
    if code == F.MISSING:
        return (2, 0.0)
    if var == "faixa_horaria":
        return (0, float(FAIXA_ORDER.get(code, 9)))
    if var == "faixa_etaria":
        return (0, float(code.split("-")[0].rstrip("+")))
    k = natural_key(code)
    return (k[0], k[1])

with tab_assoc:
    st.caption("Estatísticas descritivas do conjunto de treino — associação não implica causalidade, e notificações "
               "refletem também quem chega ao serviço de saúde e como a ficha é preenchida.")
    assoc = pd.DataFrame(S["target_association"])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        a1 = assoc[assoc.measure == "pearson"].copy()
        a1 = a1.reindex(a1["value"].abs().sort_values(ascending=False).index).head(20).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=a1["value"], y=[flabel(f) for f in a1["feature"]], orientation="h", width=0.62,
            marker=dict(color=[C["up"] if v > 0 else C["down"] for v in a1["value"]], cornerradius=4),
            hovertemplate="%{y}: r = %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=C["muted"], width=1))
        layout(fig, 26 * len(a1) + 90, "Correlação com o desfecho · binárias e numéricas", legend=False)
        fig.update_xaxes(title="correlação (r) · vermelho = mais comum com o desfecho")
        plot(fig, "assoc_bin")
    with c2:
        a2 = assoc[assoc.measure == "cramers_v"].sort_values("value")
        fig = go.Figure(go.Bar(x=a2["value"], y=[flabel(f) for f in a2["feature"]], orientation="h", width=0.62,
                               marker=dict(color=C["pos"], cornerradius=4),
                               hovertemplate="%{y}: V = %{x:.3f}<extra></extra>"))
        layout(fig, 26 * len(a2) + 90, "Associação com o desfecho · categóricas", legend=False)
        fig.update_xaxes(title="V de Cramér (0 = nenhuma, 1 = total)")
        plot(fig, "assoc_cat")

    st.markdown("##### Taxa do desfecho por categoria")
    rate_vars = [c for c in S["category_rates"]]
    default_var = "uf_ocorrencia" if "uf_ocorrencia" in rate_vars else rate_vars[0]
    var = st.selectbox("Variável", rate_vars, index=rate_vars.index(default_var), format_func=flabel, key="rate_var")
    rr = pd.DataFrame(S["category_rates"][var])
    prev = S["prevalence_train"]
    if var == "uf_ocorrencia":
        c1, c2 = st.columns([1.35, 1], gap="large")
        with c1:
            geo = load_json("br_uf.geojson")
            props = {f["id"]: f["properties"] for f in geo["features"]}
            ufr = rr[rr.code.isin(list(props))].set_index("code")
            lo, hi = float(ufr["rate"].min()), float(ufr["rate"].max())
            scale = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
            fig = go.Figure()
            for f in geo["features"]:
                uf = f["id"]
                if uf not in ufr.index:
                    continue
                r = ufr.loc[uf]
                t = (r["rate"] - lo) / max(hi - lo, 1e-9)
                xs, ys = uf_outline(f["geometry"])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", fill="toself", hoveron="fills", hoverinfo="text",
                    fillcolor=sample_colorscale(scale, [t])[0], line=dict(color="#ffffff", width=1.1),
                    text=f"<b>{props[uf]['nome']}</b><br>{pct(r['rate'])} · {br(r['n'])} casos", showlegend=False))
            order = [f["id"] for f in geo["features"] if f["id"] in ufr.index]
            inside = [u for u in order if u not in UF_CALLOUT]
            fig.add_trace(go.Scatter(
                x=[props[u]["lon"] for u in inside], y=[merc(props[u]["lat"]) for u in inside], text=inside,
                mode="text", hoverinfo="skip", showlegend=False,
                textfont=dict(size=10.5, family=FONT,
                              color=["#ffffff" if (ufr.loc[u, "rate"] - lo) / max(hi - lo, 1e-9) > .55 else "#1d1d1b"
                                     for u in inside])))
            out = [u for u in order if u in UF_CALLOUT]
            lx, ly = [], []
            for u in out:  # linha fina do estado até o rótulo de fora
                x0, y0 = UF_LINE_FROM.get(u, (props[u]["lon"], props[u]["lat"]))
                lx += [x0, UF_CALLOUT[u][0], None]
                ly += [merc(y0), merc(UF_CALLOUT[u][1]), None]
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color="#8a8984", width=0.8)))
            fig.add_trace(go.Scatter(
                x=[UF_CALLOUT[u][0] for u in out], y=[merc(UF_CALLOUT[u][1]) for u in out], text=out,
                mode="text", hoverinfo="skip", showlegend=False,
                textposition=["middle left" if UF_CALLOUT[u][0] < props[u]["lon"] else "middle right" for u in out],
                textfont=dict(size=10.5, family=FONT, color="#1d1d1b")))
            fig.add_trace(go.Scatter(  # só para desenhar a barra de cores
                x=[props[order[0]]["lon"]] * 2, y=[merc(props[order[0]]["lat"])] * 2, mode="markers", hoverinfo="skip",
                showlegend=False, marker=dict(size=0.1, opacity=0, color=[lo, hi], cmin=lo, cmax=hi, colorscale=scale,
                                              showscale=True, colorbar=dict(
                                                  title=dict(text="taxa", side="top"), tickformat=".0%", thickness=10,
                                                  len=0.34, x=1.0, xanchor="right", y=0.02, yanchor="bottom",
                                                  outlinewidth=0))))
            layout(fig, 620, f"Mapa por UF — taxa de {SHORT_LC}", legend=False)
            fig.update_xaxes(visible=False, fixedrange=True, range=[-74.5, -30.5], constrain="domain")
            fig.update_yaxes(visible=False, fixedrange=True, scaleanchor="x", scaleratio=1,
                             range=[merc(-34.3), merc(5.8)], constrain="domain")
            fig.update_layout(margin=dict(l=0, r=0, t=4, b=0), hoverdistance=1)
            plot(fig, "ufmap")
            st.caption("Malha estadual do IBGE (simplificada). Passe o mouse para ver o nome do estado, a taxa e o número de casos.")
        with c2:
            rrs = rr[rr.code != F.MISSING].sort_values("rate")
            bar_cols = sample_colorscale(scale, [(v - lo) / max(hi - lo, 1e-9) for v in rrs["rate"]])
            fig = go.Figure(go.Bar(x=rrs["rate"], y=rrs["code"], orientation="h", width=0.62,
                                   marker=dict(color=bar_cols, cornerradius=4),
                                   customdata=[br(n) for n in rrs["n"]],
                                   hovertemplate="%{y}: %{x:.1%} · %{customdata} casos<extra></extra>"))
            fig.add_vline(x=prev, line=dict(color=C["muted"], width=1))
            fig.add_annotation(x=prev, y=1, yref="paper", text=f"média {pct(prev)}", showarrow=False, xanchor="center",
                               yanchor="bottom", font=dict(size=11, color=C["muted"]))
            layout(fig, 640, legend=False)
            fig.update_layout(margin=dict(t=28))
            fig.update_xaxes(tickformat=".0%")
            plot(fig, "uf_bars")
    else:
        if var in ORDINAL_COLS:
            rr = rr.iloc[sorted(range(len(rr)), key=lambda i: ord_key(var, rr["code"].iloc[i]))]
        else:
            rr = rr.sort_values("rate", ascending=False)
        labs = [F.code_label(var, c) if F.kind_of(var) != "bin" else ("Sim" if c == "1" else "Não") for c in rr["code"]]
        fig = go.Figure(go.Bar(x=labs, y=rr["rate"], width=0.6, marker=dict(color=C["pos"], cornerradius=4),
                               customdata=[br(n) for n in rr["n"]],
                               text=[pct(v, 0) for v in rr["rate"]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.1%} · %{customdata} casos<extra></extra>"))
        fig.add_hline(y=prev, line=dict(color=C["muted"], width=1))
        fig.add_annotation(x=1, xref="paper", y=prev, text=f"média {pct(prev)}", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=11, color=C["muted"]))
        layout(fig, 380, f"{flabel(var)} — taxa de {SHORT_LC}", legend=False)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        plot(fig, "rate_bars")

    with st.expander("Matriz de correlação completa (binárias e numéricas)"):
        cm = S["full_correlation_matrix"]
        dfc = pd.DataFrame(cm["data"], columns=cm["columns"], index=cm["index"])
        arr = dfc.to_numpy(dtype=float, copy=True)  # pandas 3 (copy-on-write) devolve view só-leitura
        arr[np.triu(np.ones_like(arr, dtype=bool), k=0)] = np.nan
        names = [F.short_label(c) for c in dfc.columns]
        lim = float(np.nanmax(np.abs(arr)))
        fig = go.Figure(go.Heatmap(
            z=arr, x=names, y=names, zmid=0, zmin=-lim, zmax=lim,
            colorscale=[[0, "#184f95"], [0.25, "#6da7ec"], [0.5, C["mid"]], [0.75, "#ee8c8b"], [1, "#b72d2d"]],
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>", xgap=1, ygap=1,
            colorbar=dict(thickness=10, len=0.5, title=dict(text="r", side="top"))))
        layout(fig, 760, legend=False)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10.5))
        fig.update_xaxes(tickangle=-50, tickfont=dict(size=10.5))
        plot(fig, "corr")
        st.caption("Escala divergente centrada em zero: vermelho = ocorrem juntos com mais frequência; azul = raramente "
                   "juntos. Apenas o triângulo inferior, sem a diagonal.")

# =========================================================================== #
# 5. EQUIDADE
# =========================================================================== #
GROUP_NAMES = {"raca_paciente": "Raça/cor", "faixa_etaria": "Faixa etária", "regiao_ocorrencia": "Região"}
with tab_fair:
    st.markdown(f"##### Desempenho por subgrupo · {ALGO_NAME[algo]}")
    st.caption("Mesmo limiar para todos os grupos. Diferenças grandes de sensibilidade ou de falso positivo indicam que o "
               "modelo erra mais para alguns grupos — algo a reportar em qualquer uso do modelo.")
    gsel = None
    if hasattr(st, "segmented_control"):
        gsel = st.segmented_control("Grupo", list(GROUP_NAMES), format_func=GROUP_NAMES.get, default="raca_paciente",
                                    key="gsel", label_visibility="collapsed")
    gsel = gsel or "raca_paciente"
    sg = pd.DataFrame(MA["subgroups"].get(gsel, []))
    if sg.empty:
        st.info("Sem subgrupos com tamanho suficiente.")
    else:
        if gsel == "faixa_etaria":
            sg = sg.iloc[sorted(range(len(sg)), key=lambda i: ord_key("faixa_etaria", sg["group"].iloc[i]))]
        sg["lab"] = [F.code_label(gsel, g) for g in sg["group"]]
        t = MA["test"]
        overall = {"auc": t["roc_auc"], "recall": t["report"]["1"]["recall"],
                   "fpr": 1 - t["report"]["0"]["recall"], "precision": t["report"]["1"]["precision"]}
        panels = [("auc", "AUC"), ("recall", "Sensibilidade"), ("fpr", "Taxa de falso positivo"), ("calib", "Calibração")]
        fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.035,
                            subplot_titles=[p[1] for p in panels])
        for j, (k, _) in enumerate(panels, start=1):
            if k == "calib":
                fig.add_trace(go.Scatter(x=sg["prevalence"], y=sg["lab"], mode="markers", name="taxa real",
                                         marker=dict(size=10, color=C["pos"], line=dict(width=2, color="white")),
                                         customdata=[br(n) for n in sg["n"]],
                                         hovertemplate="%{y}: real %{x:.1%} · %{customdata} casos<extra></extra>"),
                              row=1, col=j)
                fig.add_trace(go.Scatter(x=sg["mean_pred"], y=sg["lab"], mode="markers", name="média prevista",
                                         marker=dict(size=10, color=C[algo], symbol="diamond",
                                                     line=dict(width=2, color="white")),
                                         hovertemplate="%{y}: previsto %{x:.1%}<extra></extra>"), row=1, col=j)
                fig.update_xaxes(tickformat=".0%", row=1, col=j)
            else:
                fig.add_trace(go.Scatter(x=sg[k], y=sg["lab"], mode="markers", showlegend=False,
                                         marker=dict(size=11, color=C[algo], line=dict(width=2, color="white")),
                                         customdata=[br(n) for n in sg["n"]],
                                         hovertemplate="%{y}: %{x:.3f} · %{customdata} casos<extra></extra>"),
                              row=1, col=j)
                fig.add_vline(x=overall[k], line=dict(color=C["muted"], width=1), row=1, col=j)
                fig.update_xaxes(tickformat=".2f" if k == "auc" else ".0%", row=1, col=j)
        layout(fig, 70 + 44 * len(sg))
        fig.update_layout(legend=dict(y=-0.12, yanchor="top", orientation="h"), margin=dict(t=40))
        fig.update_yaxes(autorange="reversed")
        plot(fig, "fair")
        st.caption("Linha cinza = valor no teste inteiro. No painel de calibração, círculo = taxa real e losango = "
                   "probabilidade média prevista; quanto mais próximos, melhor calibrado o modelo para o grupo.")
        if gsel in MA["selected_features"]:
            note(f"<b>{GROUP_NAMES[gsel]}</b> é uma das variáveis usadas por este modelo. Para um uso sensível, vale "
                 "comparar com um modelo treinado sem ela (no notebook, adicione-a às exclusões).", "warn")
        tbl = sg[["lab", "n", "prevalence", "mean_pred", "auc", "recall", "fpr", "precision"]].rename(columns={
            "lab": "Grupo", "n": "Casos (teste)", "prevalence": "Taxa real", "mean_pred": "Média prevista",
            "auc": "AUC", "recall": "Sensibilidade", "fpr": "Falso positivo", "precision": "Precisão"}).set_index("Grupo")
        with st.expander("Tabela"):
            table(tbl.style.format({"Casos (teste)": lambda v: br(v), "Taxa real": "{:.1%}", "Média prevista": "{:.1%}",
                                    "AUC": "{:.3f}", "Sensibilidade": "{:.1%}", "Falso positivo": "{:.1%}",
                                    "Precisão": "{:.1%}"}))

# =========================================================================== #
# 6. SIMULADOR
# =========================================================================== #
with tab_sim:
    samples = SAMPLES.get(scen_label, [])
    ref = S["reference_profile"]
    used_by = {c: [a for a in ALGOS if c in M[a]["raw_inputs"]] for c in F.RAW_COLS}
    hidden = set(F.VINCULO_COLS) if SLUG == "autor_parceiro_ou_ex" else set()
    hidden.add(SLUG)

    def case_label(i):
        if i == 0:
            return "Preenchimento livre (perfil mais comum)"
        s_ = samples[i - 1]
        uf = s_.get("uf_ocorrencia") or "–"
        return (f"Caso real #{i} · {s_.get('ano', '')} · {uf} · {int(s_['idade_paciente'] or 0)} anos · "
                f"desfecho registrado: {'sim' if s_['target_real'] == 1 else 'não'}")

    case_idx = st.selectbox("Carregar um caso do conjunto de teste (não visto no treino) ou preencher livremente",
                            range(len(samples) + 1), format_func=case_label, key=f"case_{SLUG}")
    case = samples[case_idx - 1] if case_idx > 0 else None
    base = {c: (case.get(c) if case else ref.get(c)) for c in F.RAW_COLS}
    if not case:
        base["hora_ocorrencia_h"] = None

    def wkey(c):
        return f"w_{SLUG}_{case_idx}_{c}"

    def help_for(c):
        u = used_by.get(c, [])
        return "Usado por: " + (" e ".join(ALGO_TAG[a] for a in u) if u else "nenhum modelo deste cenário")

    def widget(c, container):
        v = base.get(c)
        k = F.kind_of(c)
        lab = flabel(c) + (" ●" if used_by.get(c) else "")
        if c == "idade_paciente":
            return container.number_input(lab, 14, 110, int(v if v is not None else 30), key=wkey(c), help=help_for(c))
        if c == "numero_envolvidos":
            opts = [1.0, 2.0, None]
            cur = v if v in (1.0, 2.0) else None
            return container.selectbox(lab, opts, index=opts.index(cur), key=wkey(c), help=help_for(c),
                                       format_func=lambda x: {1.0: "Um", 2.0: "Dois ou mais"}.get(x, "Ignorado"))
        if c == "hora_ocorrencia_h":
            opts = [None] + [float(h) for h in range(24)]
            cur = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            return container.selectbox(lab, opts, index=opts.index(cur), key=wkey(c), help=help_for(c),
                                       format_func=lambda x: "Não informada" if x is None else f"{int(x):02d}h")
        if k == "cat":
            opts = sorted(S["categories"].get(c, [F.MISSING]), key=natural_key)
            cur = F.normalize_code(v)
            if cur not in opts:
                opts.append(cur)
            return container.selectbox(lab, opts, index=opts.index(cur), key=wkey(c), help=help_for(c),
                                       format_func=lambda x, c=c: F.code_label(c, x))
        return int(container.checkbox(lab, value=bool(v), key=wkey(c), help=help_for(c)))

    used_cols = [c for c in F.RAW_COLS if used_by.get(c) and c not in hidden]
    other_cols = [c for c in F.RAW_COLS if not used_by.get(c) and c not in hidden]
    st.caption("● = campo usado por pelo menos um dos modelos deste cenário (passe o mouse no ícone de ajuda para ver "
               "qual). Os demais ficam no fim e não alteram as previsões.")
    vals = {}
    groups_form = [("Vítima", ["idade_paciente", "raca_paciente", "escolaridade_paciente", "estado_civil_paciente",
                               "gestante", "possui_deficiencia"]),
                   ("Quando e onde", ["hora_ocorrencia_h", "dia_semana_ocorrencia", "mes_ocorrencia", "uf_ocorrencia",
                                      "local_ocorrencia_cod", "reside_municipio_ocorrencia", "ocorrencia_antiga"]),
                   ("Agressor", ["autor_sexo", "numero_envolvidos", "autor_alcoolizado", *F.VINCULO_COLS]),
                   ("Violência e meios", F.VIOLENCIA_COLS + F.MEIO_COLS)]
    cols = st.columns(4, gap="medium")
    for (gname, gcols), col in zip(groups_form, cols):
        col.markdown(f"**{gname}**")
        for c in gcols:
            if c in used_cols:
                vals[c] = widget(c, col)
    with st.expander("Campos não usados pelos modelos deste cenário"):
        oc = st.columns(4)
        for i, c in enumerate(other_cols):
            vals[c] = widget(c, oc[i % 4])
    for c in F.RAW_COLS:
        if c not in vals:
            vals[c] = base.get(c)

    # Estado do caso
    def same(a_, b_):
        na_, nb_ = F.normalize_code(a_), F.normalize_code(b_)
        return na_ == nb_

    changed = [flabel(c) for c in F.RAW_COLS if case and c not in hidden and not same(vals[c], case.get(c))]
    if case and not changed:
        note(f"<b>Caso real carregado.</b> Registro do conjunto de teste, não usado no treino. Desfecho registrado no "
             f"SINAN: <b>{'sim' if case['target_real'] == 1 else 'não'}</b>.")
    elif case:
        note(f"<b>Caso modificado — agora é uma simulação.</b> Você alterou: {', '.join(changed)}. A comparação com o "
             "desfecho real foi desativada.", "warn")
    else:
        note("<b>Preenchimento livre.</b> Os campos começam no perfil mais comum do treino; ajuste para simular "
             "hipóteses.", "gray")

    row = pd.DataFrame([vals])[F.RAW_COLS]
    probs = {a: float(load_model(M[a]["file"]).predict_proba(row)[0, 1]) for a in ALGOS}

    st.markdown("##### Resultado")
    pc = st.columns(2, gap="medium")
    for a, col in zip(ALGOS, pc):
        p, t_ = probs[a], M[a]["threshold"]
        above = p >= t_
        verdict = ""
        if case and not changed:
            ok = int(above) == case["target_real"]
            verdict = f" · {'acertou' if ok else 'errou'} o desfecho real"
        col.markdown(f"""
<div class="pred {'sel' if a == algo else ''}" style="--accent:{C[a]}">
  <div class="top"><span class="name"><span class="key" style="background:{C[a]}"></span>{ALGO_NAME[a]}</span>
    <span class="cls">{'acima' if above else 'abaixo'} do limiar ({pct(t_, 0)}){verdict}</span></div>
  <div class="p">{pct(p)}</div>
  <div class="bullet" style="color:{C[a]}">
    <div class="fill" style="width:{100 * p:.1f}%; background:{C[a]}"></div>
    <div class="thr" style="left:calc({100 * t_:.1f}% - 1px)"></div>
    <div class="prev" style="left:calc({100 * S['prevalence_test']:.1f}% - 1px)"></div>
  </div>
  <div class="legend-mini"><span>▮ probabilidade prevista</span><span>| limiar</span>
    <span style="color:{C['muted']}">| prevalência no teste ({pct(S['prevalence_test'], 0)})</span></div>
</div>""", unsafe_allow_html=True)

    if vals.get("meio_arma_fogo") or vals.get("meio_enforcamento"):
        note("<b>Atenção — marcadores de risco de letalidade.</b> Arma de fogo e asfixia/enforcamento estão entre os "
             "principais sinais de risco de feminicídio. Em bases de atendimento como o SINAN, casos fatais muitas vezes "
             "não chegam a gerar histórico, o que pode deixar a probabilidade de reincidência contraintuitivamente baixa "
             "(viés de sobrevivência).", "warn")

    sim_a = st.container()
    with sim_a:
        # Contribuição local: p(caso) − p(caso com o campo no valor mais comum)
        inputs = [c for c in M[algo]["raw_inputs"] if c not in hidden]
        refv = dict(ref)
        refv["hora_ocorrencia_h"] = None
        alt_rows, alt_cols = [], []
        for c in inputs:
            if not same(vals[c], refv.get(c)):
                r_ = dict(vals)
                r_[c] = refv.get(c)
                alt_rows.append(r_)
                alt_cols.append(c)
        st.markdown(f"**O que mais pesou neste caso** · {ALGO_NAME[algo]}")
        if alt_rows:
            pa = load_model(M[algo]["file"]).predict_proba(pd.DataFrame(alt_rows)[F.RAW_COLS])[:, 1]
            contrib = pd.DataFrame({"col": alt_cols, "delta": probs[algo] - pa})
            contrib = contrib[contrib["delta"].abs() >= 0.0005]
            contrib = contrib.reindex(contrib["delta"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]

            def _desc(c):
                v = vals[c]
                if F.kind_of(c) == "bin":
                    return f"{flabel(c)}: {'sim' if v else 'não'}"
                if c == "hora_ocorrencia_h":
                    return f"Hora: {'não informada' if v is None else f'{int(v):02d}h'}"
                if c == "idade_paciente":
                    return f"Idade: {int(v)} anos"
                if c == "numero_envolvidos":
                    return f"Nº de agressores: {F.code_label(c, v)}"
                return f"{flabel(c)}: {F.code_label(c, v)}"

            lo, hi = min(contrib["delta"].min(), 0), max(contrib["delta"].max(), 0)
            pad = 0.18 * max(hi - lo, 0.01)
            fig = go.Figure(go.Bar(
                x=contrib["delta"], y=[_desc(c) for c in contrib["col"]], orientation="h", width=0.62,
                marker=dict(color=[C["up"] if v > 0 else C["down"] for v in contrib["delta"]], cornerradius=4),
                text=[("+" if v > 0 else "−") + br(abs(100 * v), 1) + " p.p." for v in contrib["delta"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:+.1%} em relação ao valor mais comum<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=C["muted"], width=1))
            layout(fig, 34 * len(contrib) + 70, legend=False)
            fig.update_xaxes(tickformat="+.0%", title="efeito na probabilidade", range=[lo - pad, hi + pad])
            fig.update_layout(margin=dict(r=24))
            plot(fig, "contrib")
            st.caption("Vermelho aumenta e azul reduz a probabilidade. Cada barra mostra quanto ela mudaria se apenas "
                       "aquele campo voltasse ao valor mais comum no treino; campos que já estão no valor mais comum "
                       "não aparecem.")
        else:
            st.info("Todos os campos usados pelo modelo estão no valor mais comum do treino.")
    with st.container():
        # "E se...?": varia um campo e mantém os demais fixos
        wi_opts = [c for c in F.RAW_COLS if used_by.get(c) and c not in hidden]
        if wi_opts:
            wi_default = "idade_paciente" if "idade_paciente" in wi_opts else wi_opts[0]
            wi = st.selectbox("**E se…?** Varie um campo (os demais ficam como estão)", wi_opts,
                              index=wi_opts.index(wi_default), format_func=flabel, key=f"whatif_{SLUG}")
            kind = F.kind_of(wi)
            if wi == "idade_paciente":
                xs = [float(x) for x in range(14, 81)]
            elif wi == "hora_ocorrencia_h":
                xs = [None] + [float(h) for h in range(24)]
            elif wi == "numero_envolvidos":
                xs = [1.0, 2.0, None]
            elif kind == "cat":
                xs = sorted(S["categories"].get(wi, []), key=lambda c_: ord_key(wi, c_))
            else:
                xs = [0, 1]
            grid = pd.DataFrame([dict(vals, **{wi: x}) for x in xs])[F.RAW_COLS]
            preds = {a: load_model(M[a]["file"]).predict_proba(grid)[:, 1] for a in ALGOS}
            fig = go.Figure()
            if wi == "idade_paciente":
                for a in ALGOS:
                    fig.add_trace(go.Scatter(x=xs, y=preds[a], mode="lines", line=dict(color=C[a], width=2),
                                             name=ALGO_NAME[a],
                                             hovertemplate="%{x} anos: %{y:.1%}<extra>" + ALGO_TAG[a] + "</extra>"))
                    fig.add_trace(go.Scatter(x=[vals[wi]], y=[probs[a]], mode="markers", showlegend=False,
                                             marker=dict(size=10, color=C[a], line=dict(width=2, color="white")),
                                             hoverinfo="skip"))
                fig.update_xaxes(title="idade da vítima")
            else:
                if wi == "hora_ocorrencia_h":
                    labs = ["n/i" if x is None else f"{int(x)}h" for x in xs]
                elif kind == "bin":
                    labs = ["Não", "Sim"]
                else:
                    labs = [F.code_label(wi, x) for x in xs]
                cur = F.normalize_code(vals[wi])
                for a in ALGOS:
                    fig.add_trace(go.Bar(
                        x=labs, y=preds[a], name=ALGO_NAME[a], marker=dict(color=C[a], cornerradius=4,
                        opacity=[1.0 if F.normalize_code(x) == cur else 0.45 for x in xs]),
                        hovertemplate="%{x}: %{y:.1%}<extra>" + ALGO_TAG[a] + "</extra>"))
                fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
                fig.update_xaxes(tickangle=-35 if len(xs) > 8 else 0)
            layout(fig, 340)
            fig.update_yaxes(tickformat=".0%", title="probabilidade prevista", rangemode="tozero")
            plot(fig, f"whatif_{wi}")
            unused = [ALGO_TAG[a] for a in ALGOS if wi not in M[a]["raw_inputs"]]
            st.caption("Barras opacas = valor atual. "
                       + (f"O {unused[0]} não usa este campo — por isso não varia." if unused else ""))

# =========================================================================== #
# 7. METODOLOGIA
# =========================================================================== #
with tab_meth:
    cfg = META.get("config", {})
    st.markdown(f"""
#### Dados
- **Fonte:** `basedosdados.br_ms_sinan.microdados_violencia` (SINAN/DATASUS via Base dos Dados), consulta `query_violencia_v3.sql`.
- **Recorte:** vítimas do sexo feminino, 14 anos ou mais, violência **não autoprovocada**; {br(META['n_rows_csv'])} notificações de {yrs[0]} a {yrs[1]}.
- **Códigos:** a base usa os códigos originais do SINAN (1 = sim, 2 = não, 9 = ignorado). "Ignorado" é mantido como categoria própria — não é contado como "não".
- **Lacunas conhecidas:** 2013 e 2014 estão incompletos no reprocessamento atual da Base dos Dados; o campo de encaminhamento à DEAM só existe a partir de 2015 (mudança da ficha), então esse cenário usa 2015 em diante.

#### Variáveis
- **{len(F.ALL_CANDIDATES)} candidatas:** {len(F.RAW_COLS)} campos da ficha + {len(F.DERIVED_COLS)} derivadas ({', '.join(flabel(c).lower() for c in F.DERIVED_COLS)}).
- **Excluídas por vazamento:** os outros encaminhamentos (decisões do mesmo atendimento) em todos os cenários; o vínculo do agressor no cenário "Agressor íntimo", porque é o próprio alvo.
- **Normalização** dentro do pipeline (`features.prepare_features`): o app envia a linha crua e o modelo trata códigos como `4`, `"4"` ou `"4.0"` da mesma forma — isso corrige o bug da v2, em que raça, escolaridade e situação conjugal eram ignoradas no simulador.

#### Seleção de variáveis
- Para **cada algoritmo em cada cenário**: eliminação recursiva guiada por **importância de permutação** (queda de AUC) numa validação interna ({br(cfg.get('sel_fit', 0))} linhas de ajuste, {br(cfg.get('sel_val', 0))} de validação).
- Escolha do **menor subconjunto** com AUC ≥ melhor AUC − 1 erro-padrão (Hanley & McNeil) — a regra de 1 EP prefere o modelo mais simples quando a diferença não é distinguível do ruído.

#### Modelos e avaliação
- **Regressão logística** (mediana + padronização, one-hot) e **HistGradientBoosting** (categóricas nativas, early stopping).
- Sem pesos de classe: probabilidades calibradas. **Limiar** escolhido na validação pelo índice de Youden; ajustável na aba Desempenho.
- Divisão **80/20 estratificada** (`random_state=42`); 15% do treino separados para validação. O teste não participa de nenhuma escolha.
- **Validação temporal:** treino até {cfg.get('temporal_corte', 2021)} e teste nos anos seguintes.
- **Baseline:** a configuração v2 retreinada nos mesmos dados e na mesma divisão.

#### Limitações e uso responsável
- Notificação ≠ ocorrência: o SINAN registra os casos que chegam aos serviços de saúde, com subnotificação desigual entre regiões, grupos e períodos.
- Os modelos aprendem padrões de **registro** (incluindo práticas de preenchimento e de encaminhamento), não causas.
- Probabilidades se referem a perfis estatísticos; **não** são avaliação de risco de uma pessoa e não devem orientar decisões sobre casos individuais.
- Atributos sensíveis (raça/cor, idade, região) podem ser usados pelos modelos — veja a aba **Equidade** antes de qualquer uso aplicado.
""")
    with st.expander("Configuração do treino"):
        st.json(cfg)

st.markdown('<div class="footer">Precisa de ajuda ou conhece alguém em situação de violência? <b>Ligue 180</b> — '
            'Central de Atendimento à Mulher (gratuito, 24h). Em emergência, <b>190</b>.</div>', unsafe_allow_html=True)
