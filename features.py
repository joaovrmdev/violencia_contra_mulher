# features.py
"""
Esquema de variáveis compartilhado entre o treino (notebook treino_modelos_v3.ipynb) e o app (app.py).

Concentra num só lugar:
  * as colunas brutas do CSV exportado do BigQuery (query_violencia_v3.sql);
  * a normalização dos códigos categóricos (4 / "4" / "4.0" -> "4"; nulo -> "ignorado");
  * as variáveis derivadas (hora -> noite/faixa horária, contagens, faixa etária, região...);
  * os rótulos usados na interface.

IMPORTANTE: os pipelines salvos em .joblib começam com
`FunctionTransformer(features.prepare_features)`. Por isso este arquivo precisa estar
no mesmo diretório do app.py. O app envia a linha "crua" e o próprio pipeline cuida
de tipagem, normalização e variáveis derivadas — treino e inferência nunca divergem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "ignorado"

# ---------------------------------------------------------------------------
# 1. Colunas brutas (entradas do formulário do app)
# ---------------------------------------------------------------------------
RAW_NUM_COLS = ["idade_paciente", "numero_envolvidos", "hora_ocorrencia_h"]
RAW_CAT_COLS = [
    "dia_semana_ocorrencia", "mes_ocorrencia", "uf_ocorrencia", "local_ocorrencia_cod",
    "reside_municipio_ocorrencia", "raca_paciente", "escolaridade_paciente",
    "estado_civil_paciente", "autor_sexo",
]
VIOLENCIA_COLS = [
    "violencia_fisica", "violencia_psicologica", "violencia_sexual",
    "violencia_financeira", "violencia_negligencia", "violencia_tortura",
]
MEIO_COLS = [
    "meio_forca_corporal", "meio_enforcamento", "meio_objeto_contundente",
    "meio_objeto_perfurante", "meio_arma_fogo", "meio_ameaca",
]
# Vínculo do agressor. Fica fora do cenário "Agressor Íntimo" (o vínculo é o próprio alvo).
VINCULO_COLS = ["autor_parceiro_ou_ex", "autor_familiar", "autor_conhecido", "autor_desconhecido"]
RAW_BIN_COLS = [
    "gestante", "possui_deficiencia", *VIOLENCIA_COLS, *MEIO_COLS,
    "autor_alcoolizado", *VINCULO_COLS, "ocorrencia_antiga",
]
RAW_COLS = RAW_NUM_COLS + RAW_CAT_COLS + RAW_BIN_COLS

TARGETS = {
    "Ciclo de Reincidência": "target_reincidencia",
    "Agressor Íntimo (Parceiro/Ex)": "autor_parceiro_ou_ex",
    "Encaminhamento DEAM": "encaminhamento_delegacia_mulher",
}
# Colunas do CSV que nunca entram como variável (vazamento ou identificação)
NEVER_FEATURES = [
    "encaminhamento_assistencia_social", "encaminhamento_atendimento_mulher",
    "encaminhamento_delegacia_mulher", "target_reincidencia",
    "ano", "data_ocorrencia", "id_municipio_ocorrencia", "nome_municipio_ocorrencia",
    "hora_ocorrencia_raw",
]

# ---------------------------------------------------------------------------
# 2. Variáveis derivadas (calculadas a partir das brutas, sem novos campos)
# ---------------------------------------------------------------------------
UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}
FAIXAS_ETARIAS = [(0, 17, "14-17"), (18, 24, "18-24"), (25, 34, "25-34"),
                  (35, 44, "35-44"), (45, 59, "45-59"), (60, 200, "60+")]
FAIXAS_HORARIAS = [(0, 5, "madrugada"), (6, 11, "manha"), (12, 17, "tarde"), (18, 23, "noite")]
LETAL_COLS = ["meio_arma_fogo", "meio_objeto_perfurante", "meio_enforcamento"]

DERIVED = {
    "ocorreu_noite_madrugada": {"kind": "cat", "sources": ["hora_ocorrencia_h"]},
    "faixa_horaria": {"kind": "cat", "sources": ["hora_ocorrencia_h"]},
    "local_residencia": {"kind": "cat", "sources": ["local_ocorrencia_cod"]},
    "n_tipos_violencia": {"kind": "num", "sources": VIOLENCIA_COLS},
    "n_meios_agressao": {"kind": "num", "sources": MEIO_COLS},
    "meio_potencialmente_letal": {"kind": "bin", "sources": LETAL_COLS},
    "faixa_etaria": {"kind": "cat", "sources": ["idade_paciente"]},
    "fim_de_semana": {"kind": "bin", "sources": ["dia_semana_ocorrencia"]},
    "regiao_ocorrencia": {"kind": "cat", "sources": ["uf_ocorrencia"]},
}
DERIVED_COLS = list(DERIVED)
ALL_CANDIDATES = RAW_COLS + DERIVED_COLS


def kind_of(col: str) -> str:
    """'num', 'cat' ou 'bin'."""
    if col in RAW_NUM_COLS:
        return "num"
    if col in RAW_CAT_COLS:
        return "cat"
    if col in RAW_BIN_COLS:
        return "bin"
    if col in DERIVED:
        return DERIVED[col]["kind"]
    raise KeyError(col)


def sources_of(col: str) -> list[str]:
    """Campos brutos de que uma variável depende (ela mesma, se for bruta)."""
    return list(DERIVED[col]["sources"]) if col in DERIVED else [col]


def raw_inputs_for(features: list[str]) -> list[str]:
    """Campos brutos (do formulário) necessários para um conjunto de variáveis."""
    used = {src for f in features for src in sources_of(f)}
    return [c for c in RAW_COLS if c in used]


# ---------------------------------------------------------------------------
# 3. Normalização e preparo
# ---------------------------------------------------------------------------
def normalize_code(value, missing: str = MISSING) -> str:
    """4 / '4' / '4.0' / ' 4 ' -> '4';  NaN / '' / None -> 'ignorado'."""
    if value is None:
        return missing
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return missing
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "<na>", missing}:
        return missing
    try:
        f = float(s.replace(",", "."))
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def _normalize_series(ser: pd.Series, col: str) -> pd.Series:
    mapping = {}
    for u in pd.unique(ser):
        code = normalize_code(u)
        if code != MISSING:
            if col == "uf_ocorrencia":
                code = code.upper()
            elif col == "autor_sexo":
                code = {"M": "M", "1": "M", "F": "F", "2": "F", "AMBOS": "Ambos", "3": "Ambos"}.get(code.upper(), MISSING)
            elif col == "reside_municipio_ocorrencia":
                code = code if code in ("0", "1") else MISSING
        mapping[u] = code

    def get(v):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return MISSING
        return mapping.get(v, MISSING)

    return pd.Series([get(v) for v in ser.to_numpy(dtype=object)], index=ser.index, dtype=object)


def _bins(values: pd.Series, bins) -> pd.Series:
    out = np.full(len(values), MISSING, dtype=object)
    v = values.to_numpy(dtype=float)
    for lo, hi, lab in bins:
        out[(v >= lo) & (v <= hi)] = lab
    return pd.Series(out, index=values.index, dtype=object)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem + normalização + variáveis derivadas. Idempotente.

    Recebe um DataFrame com (pelo menos parte das) colunas brutas e devolve um DataFrame
    com todas as ALL_CANDIDATES. Colunas ausentes viram faltantes (numéricas -> NaN,
    categóricas -> 'ignorado', binárias -> 0).
    """
    df = pd.DataFrame(df)
    out = pd.DataFrame(index=df.index)

    for c in RAW_NUM_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
    out.loc[(out["idade_paciente"] < 0) | (out["idade_paciente"] > 110), "idade_paciente"] = np.nan
    out.loc[~out["numero_envolvidos"].isin([1, 2]), "numero_envolvidos"] = np.nan   # 9 = ignorado
    out.loc[(out["hora_ocorrencia_h"] < 0) | (out["hora_ocorrencia_h"] > 23), "hora_ocorrencia_h"] = np.nan

    for c in RAW_CAT_COLS:
        out[c] = _normalize_series(df[c], c) if c in df else MISSING

    for c in RAW_BIN_COLS:
        # Qualquer valor diferente de 1 (0, 2 = "não" no SINAN, 9 = ignorado, NaN) vira 0
        out[c] = (pd.to_numeric(df[c], errors="coerce") == 1).astype(int) if c in df else 0

    # --- derivadas ---
    hora = out["hora_ocorrencia_h"]
    if "hora_ocorrencia_h" in df or "ocorreu_noite_madrugada" not in df:
        noite = np.where(hora.isna(), MISSING, np.where((hora >= 18) | (hora < 6), "1", "0"))
        out["ocorreu_noite_madrugada"] = pd.Series(noite, index=out.index, dtype=object)
    else:  # compatibilidade com CSVs antigos, que só tinham a coluna binária
        out["ocorreu_noite_madrugada"] = _normalize_series(df["ocorreu_noite_madrugada"], "reside_municipio_ocorrencia")
    out["faixa_horaria"] = _bins(hora, FAIXAS_HORARIAS)

    if "local_ocorrencia_cod" in df or "local_residencia" not in df:
        loc = out["local_ocorrencia_cod"]
        out["local_residencia"] = pd.Series(np.where(loc == MISSING, MISSING, np.where(loc == "1", "1", "0")),
                                            index=out.index, dtype=object)
    else:
        out["local_residencia"] = _normalize_series(df["local_residencia"], "reside_municipio_ocorrencia")

    out["n_tipos_violencia"] = out[VIOLENCIA_COLS].sum(axis=1).astype(float)
    out["n_meios_agressao"] = out[MEIO_COLS].sum(axis=1).astype(float)
    out["meio_potencialmente_letal"] = (out[LETAL_COLS].sum(axis=1) > 0).astype(int)
    out["faixa_etaria"] = _bins(out["idade_paciente"], FAIXAS_ETARIAS)
    out["fim_de_semana"] = out["dia_semana_ocorrencia"].isin(["1", "7"]).astype(int)
    out["regiao_ocorrencia"] = out["uf_ocorrencia"].map(UF_REGIAO).fillna(MISSING).astype(object)

    return out[ALL_CANDIDATES]


# ---------------------------------------------------------------------------
# 4. Rótulos para a interface
# ---------------------------------------------------------------------------
LABELS = {
    "idade_paciente": "Idade da vítima",
    "numero_envolvidos": "Nº de agressores",
    "hora_ocorrencia_h": "Hora da ocorrência",
    "dia_semana_ocorrencia": "Dia da semana",
    "mes_ocorrencia": "Mês da ocorrência",
    "uf_ocorrencia": "UF da ocorrência",
    "local_ocorrencia_cod": "Local da ocorrência",
    "reside_municipio_ocorrencia": "Mora no município do fato",
    "raca_paciente": "Raça/cor da vítima",
    "escolaridade_paciente": "Escolaridade da vítima",
    "estado_civil_paciente": "Situação conjugal da vítima",
    "autor_sexo": "Sexo do agressor",
    "gestante": "Vítima gestante",
    "possui_deficiencia": "Vítima com deficiência/transtorno",
    "violencia_fisica": "Violência física",
    "violencia_psicologica": "Violência psicológica",
    "violencia_sexual": "Violência sexual",
    "violencia_financeira": "Violência financeira/patrimonial",
    "violencia_negligencia": "Negligência/abandono",
    "violencia_tortura": "Tortura",
    "meio_forca_corporal": "Força corporal/espancamento",
    "meio_enforcamento": "Enforcamento/asfixia",
    "meio_objeto_contundente": "Objeto contundente",
    "meio_objeto_perfurante": "Objeto perfurocortante",
    "meio_arma_fogo": "Arma de fogo",
    "meio_ameaca": "Ameaça",
    "autor_alcoolizado": "Agressor sob efeito de álcool",
    "autor_parceiro_ou_ex": "Agressor é parceiro/ex",
    "autor_familiar": "Agressor é familiar",
    "autor_conhecido": "Agressor é conhecido",
    "autor_desconhecido": "Agressor é desconhecido",
    "ocorrencia_antiga": "Fato com mais de 1 ano",
    "ocorreu_noite_madrugada": "Noite/madrugada (18h–6h)",
    "faixa_horaria": "Faixa horária",
    "local_residencia": "Ocorreu na residência",
    "n_tipos_violencia": "Nº de tipos de violência",
    "n_meios_agressao": "Nº de meios de agressão",
    "meio_potencialmente_letal": "Meio potencialmente letal",
    "faixa_etaria": "Faixa etária",
    "fim_de_semana": "Fim de semana",
    "regiao_ocorrencia": "Região",
    "target_reincidencia": "Reincidência (alvo)",
    "encaminhamento_delegacia_mulher": "Encaminhamento à DEAM (alvo)",
}

SHORT_LABELS = {
    "idade_paciente": "Idade", "numero_envolvidos": "Nº agressores", "hora_ocorrencia_h": "Hora",
    "dia_semana_ocorrencia": "Dia semana", "mes_ocorrencia": "Mês", "uf_ocorrencia": "UF",
    "local_ocorrencia_cod": "Local", "reside_municipio_ocorrencia": "Mora no mun.",
    "raca_paciente": "Raça/cor", "escolaridade_paciente": "Escolaridade",
    "estado_civil_paciente": "Sit. conjugal", "autor_sexo": "Sexo agressor", "gestante": "Gestante",
    "possui_deficiencia": "Deficiência", "violencia_fisica": "V. física", "violencia_psicologica": "V. psicológica",
    "violencia_sexual": "V. sexual", "violencia_financeira": "V. financeira", "violencia_negligencia": "Negligência",
    "violencia_tortura": "Tortura", "meio_forca_corporal": "Força corporal", "meio_enforcamento": "Asfixia",
    "meio_objeto_contundente": "Obj. contundente", "meio_objeto_perfurante": "Obj. cortante",
    "meio_arma_fogo": "Arma de fogo", "meio_ameaca": "Ameaça", "autor_alcoolizado": "Álcool (agressor)",
    "autor_parceiro_ou_ex": "Parceiro/ex", "autor_familiar": "Familiar", "autor_conhecido": "Conhecido",
    "autor_desconhecido": "Desconhecido", "ocorrencia_antiga": "Fato antigo",
    "ocorreu_noite_madrugada": "Noite/madr.", "faixa_horaria": "Faixa horária", "local_residencia": "Residência",
    "n_tipos_violencia": "Nº tipos viol.", "n_meios_agressao": "Nº meios", "meio_potencialmente_letal": "Meio letal",
    "faixa_etaria": "Faixa etária", "fim_de_semana": "Fim de semana", "regiao_ocorrencia": "Região",
    "target_reincidencia": "Reincidência", "autor_parceiro_ou_ex_alvo": "Parceiro/ex",
    "encaminhamento_delegacia_mulher": "DEAM",
}

GROUPS = {
    "Vítima": ["idade_paciente", "faixa_etaria", "raca_paciente", "escolaridade_paciente",
               "estado_civil_paciente", "gestante", "possui_deficiencia"],
    "Quando e onde": ["hora_ocorrencia_h", "ocorreu_noite_madrugada", "faixa_horaria", "dia_semana_ocorrencia",
                      "fim_de_semana", "mes_ocorrencia", "ocorrencia_antiga", "uf_ocorrencia", "regiao_ocorrencia",
                      "local_ocorrencia_cod", "local_residencia", "reside_municipio_ocorrencia"],
    "Agressor": ["autor_sexo", "numero_envolvidos", "autor_alcoolizado", *VINCULO_COLS],
    "Tipos de violência": VIOLENCIA_COLS + ["n_tipos_violencia"],
    "Meios de agressão": MEIO_COLS + ["n_meios_agressao", "meio_potencialmente_letal"],
}
GROUP_OF = {col: g for g, cols in GROUPS.items() for col in cols}

_SIM_NAO = {"1": "Sim", "0": "Não"}
CODE_LABELS = {
    "dia_semana_ocorrencia": {"1": "Domingo", "2": "Segunda", "3": "Terça", "4": "Quarta",
                              "5": "Quinta", "6": "Sexta", "7": "Sábado"},
    "mes_ocorrencia": {"1": "Jan", "2": "Fev", "3": "Mar", "4": "Abr", "5": "Mai", "6": "Jun",
                       "7": "Jul", "8": "Ago", "9": "Set", "10": "Out", "11": "Nov", "12": "Dez"},
    "local_ocorrencia_cod": {"1": "Residência", "2": "Habitação coletiva", "3": "Escola",
                             "4": "Local de prática esportiva", "5": "Bar ou similar", "6": "Via pública",
                             "7": "Comércio/serviços", "8": "Indústria/construção", "9": "Outro"},
    "reside_municipio_ocorrencia": _SIM_NAO,
    "ocorreu_noite_madrugada": _SIM_NAO,
    "local_residencia": _SIM_NAO,
    "faixa_horaria": {"madrugada": "Madrugada (0–5h)", "manha": "Manhã (6–11h)",
                      "tarde": "Tarde (12–17h)", "noite": "Noite (18–23h)"},
    "raca_paciente": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"},
    # Códigos originais do SINAN (CS_ESCOL_N)
    "escolaridade_paciente": {"0": "Analfabeta", "1": "1ª–4ª série incompleta", "2": "4ª série completa",
                              "3": "5ª–8ª série incompleta", "4": "Fundamental completo",
                              "5": "Médio incompleto", "6": "Médio completo",
                              "7": "Superior incompleto", "8": "Superior completo"},
    "estado_civil_paciente": {"1": "Solteira", "2": "Casada/união consensual", "3": "Viúva", "4": "Separada"},
    "autor_sexo": {"M": "Masculino", "F": "Feminino", "Ambos": "Ambos os sexos"},
    "numero_envolvidos": {"1": "Um", "2": "Dois ou mais"},
}


def label(col: str) -> str:
    return LABELS.get(col, col)


def short_label(col: str) -> str:
    return SHORT_LABELS.get(col, LABELS.get(col, col))


def code_label(col: str, code) -> str:
    code = normalize_code(code)
    if code == MISSING:
        return "Ignorado"
    return CODE_LABELS.get(col, {}).get(code, str(code))
