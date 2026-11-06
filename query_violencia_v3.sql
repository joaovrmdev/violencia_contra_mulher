-- query_violencia_v3.sql  (revisão 2 — códigos atuais da Base dos Dados, nov/2026)
--
-- A tabela basedosdados.br_ms_sinan.microdados_violencia foi reprocessada e voltou aos
-- códigos originais do SINAN: 1 = sim, 2 = não, 9 = ignorado; sexo F/M/I.
-- Mudanças em relação à consulta original estão marcadas com [v3].
-- Os nomes de coluna de saída são os mesmos de antes (+ 5 colunas novas no fim).

WITH
diretorio_municipios AS (
    SELECT DISTINCT
        SAFE_CAST(id_municipio AS STRING) AS id_municipio_7,
        SAFE_CAST(id_municipio_6 AS STRING) AS id_municipio_6,
        nome AS nome_municipio,
        sigla_uf
    FROM `basedosdados.br_bd_diretorios_brasil.municipio`
),
base AS (
    SELECT
        dados.*,
        -- [v3] hora_ocorrencia não converte mais para TIME (100% "formato_invalido").
        --      Extrai a hora de 'HH:MM', 'H:MM', 'HH:MM:SS', 'HHMM', 'HMM', 'HH', '20h30'
        --      e de valores com data na frente ('...T20:30:00').
        REGEXP_REPLACE(
            IFNULL(SPLIT(TRIM(SAFE_CAST(dados.hora_ocorrencia AS STRING)), 'T')[SAFE_OFFSET(1)],
                   TRIM(SAFE_CAST(dados.hora_ocorrencia AS STRING))),
            r'\s', '') AS hora_txt
    FROM `basedosdados.br_ms_sinan.microdados_violencia` AS dados
    WHERE
        -- [v3] Filtros com os códigos atuais (antes: sexo = '0' e autoprovocada = '0', que hoje não existem)
        SAFE_CAST(dados.sexo_paciente AS STRING) = 'F'
        AND SAFE_CAST(dados.lesao_autoprovocada AS STRING) = '2'
        AND SAFE_CAST(dados.idade_paciente AS INT64) >= 14
),
base_hora AS (
    SELECT
        b.*,
        CASE
            WHEN hora_txt IS NULL OR hora_txt = '' THEN NULL
            WHEN REGEXP_CONTAINS(hora_txt, r'^\d{1,2}[:hH]') THEN SAFE_CAST(REGEXP_EXTRACT(hora_txt, r'^(\d{1,2})[:hH]') AS INT64)
            WHEN REGEXP_CONTAINS(hora_txt, r'^\d{3,4}$') THEN SAFE_CAST(SUBSTR(LPAD(hora_txt, 4, '0'), 1, 2) AS INT64)
            WHEN REGEXP_CONTAINS(hora_txt, r'^\d{1,2}$') THEN SAFE_CAST(hora_txt AS INT64)
            ELSE NULL
        END AS hora_num
    FROM base AS b
)
SELECT
    -- Identificadores Temporais
    dados.ano AS ano,
    dados.data_ocorrencia AS data_ocorrencia,
    EXTRACT(DAYOFWEEK FROM SAFE_CAST(dados.data_ocorrencia AS DATE)) AS dia_semana_ocorrencia,  -- 1 = domingo
    -- [v3] Hora não informada/ inválida vira NULL (antes era contada como "dia")
    CASE
        WHEN dados.hora_num IS NULL OR dados.hora_num NOT BETWEEN 0 AND 23 THEN NULL
        WHEN dados.hora_num >= 18 OR dados.hora_num < 6 THEN 1
        ELSE 0
    END AS ocorreu_noite_madrugada,

    -- Localização Geográfica
    SAFE_CAST(dados.id_municipio_ocorrencia AS STRING) AS id_municipio_ocorrencia,
    mun_ocorrencia.nome_municipio AS nome_municipio_ocorrencia,
    mun_ocorrencia.sigla_uf AS uf_ocorrencia,
    -- [v3] Município nulo -> NULL (antes virava 0 = "não reside")
    CASE
        WHEN dados.id_municipio_residencia IS NULL OR dados.id_municipio_ocorrencia IS NULL THEN NULL
        WHEN SAFE_CAST(dados.id_municipio_residencia AS STRING) = SAFE_CAST(dados.id_municipio_ocorrencia AS STRING) THEN 1
        ELSE 0
    END AS reside_municipio_ocorrencia,
    -- [v3] Local 99 (ignorado) ou nulo -> NULL
    CASE
        WHEN dados.local_ocorrencia IS NULL OR SAFE_CAST(dados.local_ocorrencia AS STRING) = '99' THEN NULL
        WHEN SAFE_CAST(dados.local_ocorrencia AS STRING) = '1' THEN 1
        ELSE 0
    END AS local_residencia,

    -- Perfil da Vítima
    SAFE_CAST(dados.idade_paciente AS INT64) AS idade_paciente,
    CASE WHEN SAFE_CAST(dados.gestante_paciente AS STRING) IN ('1', '2', '3', '4') THEN 1 ELSE 0 END AS gestante,
    CASE WHEN SAFE_CAST(dados.raca_paciente AS STRING) IN ('8', '9') THEN NULL ELSE SAFE_CAST(dados.raca_paciente AS STRING) END AS raca_paciente,
    -- [v3] 0 = analfabeta e 8 = superior completo são categorias válidas; só 9 e 10 viram NULL
    CASE WHEN SAFE_CAST(dados.escolaridade_paciente AS STRING) IN ('9', '10') THEN NULL ELSE SAFE_CAST(dados.escolaridade_paciente AS STRING) END AS escolaridade_paciente,
    CASE WHEN SAFE_CAST(dados.estado_civil_paciente AS STRING) IN ('8', '9') THEN NULL ELSE SAFE_CAST(dados.estado_civil_paciente AS STRING) END AS estado_civil_paciente,
    IF(SAFE_CAST(dados.deficiencia_transtorno_paciente AS STRING) = '1', 1, 0) AS possui_deficiencia,

    -- Tipologia da Violência (Binárias: 1 = sim; 2, 9 ou nulo = 0)
    IF(SAFE_CAST(dados.ocorreu_violencia_fisica AS STRING) = '1', 1, 0) AS violencia_fisica,
    IF(SAFE_CAST(dados.ocorreu_violencia_psicologica AS STRING) = '1', 1, 0) AS violencia_psicologica,
    IF(SAFE_CAST(dados.ocorreu_violencia_sexual AS STRING) = '1', 1, 0) AS violencia_sexual,
    IF(SAFE_CAST(dados.ocorreu_violencia_financeira AS STRING) = '1', 1, 0) AS violencia_financeira,
    IF(SAFE_CAST(dados.ocorreu_negligencia_abandono AS STRING) = '1', 1, 0) AS violencia_negligencia,
    IF(SAFE_CAST(dados.ocorreu_tortura AS STRING) = '1', 1, 0) AS violencia_tortura,

    -- Meios Utilizados (Binárias)
    IF(SAFE_CAST(dados.meio_forca AS STRING) = '1', 1, 0) AS meio_forca_corporal,
    IF(SAFE_CAST(dados.meio_enforcamento AS STRING) = '1', 1, 0) AS meio_enforcamento,
    IF(SAFE_CAST(dados.meio_objeto_contundente AS STRING) = '1', 1, 0) AS meio_objeto_contundente,
    IF(SAFE_CAST(dados.meio_objeto_perfurante AS STRING) = '1', 1, 0) AS meio_objeto_perfurante,
    IF(SAFE_CAST(dados.meio_arma_fogo AS STRING) = '1', 1, 0) AS meio_arma_fogo,
    IF(SAFE_CAST(dados.meio_ameaca AS STRING) = '1', 1, 0) AS meio_ameaca,

    -- Perfil do Agressor
    -- [v3] 9 (ignorado) -> NULL; o código 3 (207 casos, fora do dicionário) entra como "dois ou mais"
    CASE
        WHEN SAFE_CAST(dados.numero_envolvidos_violencia AS STRING) = '1' THEN 1
        WHEN SAFE_CAST(dados.numero_envolvidos_violencia AS STRING) IN ('2', '3') THEN 2
        ELSE NULL
    END AS numero_envolvidos,
    CASE
        WHEN SAFE_CAST(dados.autor_sexo AS STRING) = '1' THEN 'M'
        WHEN SAFE_CAST(dados.autor_sexo AS STRING) = '2' THEN 'F'
        WHEN SAFE_CAST(dados.autor_sexo AS STRING) = '3' THEN 'Ambos'
        ELSE NULL
    END AS autor_sexo,
    IF(SAFE_CAST(dados.autor_usou_alcool AS STRING) = '1', 1, 0) AS autor_alcoolizado,

    -- Vínculo com o Agressor
    -- [v3] Também é alvo: se os 4 campos de parceria forem ignorados/nulos, fica NULL
    --      (antes contava como "não é parceiro")
    CASE
        WHEN '1' IN (SAFE_CAST(dados.autor_conjugue AS STRING), SAFE_CAST(dados.autor_ex_conjugue AS STRING),
                     SAFE_CAST(dados.autor_namorado_a AS STRING), SAFE_CAST(dados.autor_ex_namorado_a AS STRING)) THEN 1
        WHEN IFNULL(SAFE_CAST(dados.autor_conjugue AS STRING), '9') = '9'
         AND IFNULL(SAFE_CAST(dados.autor_ex_conjugue AS STRING), '9') = '9'
         AND IFNULL(SAFE_CAST(dados.autor_namorado_a AS STRING), '9') = '9'
         AND IFNULL(SAFE_CAST(dados.autor_ex_namorado_a AS STRING), '9') = '9' THEN NULL
        ELSE 0
    END AS autor_parceiro_ou_ex,
    IF(
        SAFE_CAST(dados.autor_pai AS STRING) = '1'
        OR SAFE_CAST(dados.autor_mae AS STRING) = '1'
        OR SAFE_CAST(dados.autor_padrasto AS STRING) = '1'
        OR SAFE_CAST(dados.autor_madrasta AS STRING) = '1'
        OR SAFE_CAST(dados.autor_filho_a AS STRING) = '1'
        OR SAFE_CAST(dados.autor_irmao AS STRING) = '1',
        1, 0
    ) AS autor_familiar,
    IF(SAFE_CAST(dados.autor_conhecido AS STRING) = '1', 1, 0) AS autor_conhecido,
    IF(SAFE_CAST(dados.autor_desconhecido AS STRING) = '1', 1, 0) AS autor_desconhecido,

    -- Encaminhamentos
    IF(SAFE_CAST(dados.encaminhamento_assistencia_social AS STRING) = '1', 1, 0) AS encaminhamento_assistencia_social,
    IF(SAFE_CAST(dados.encaminhamento_atendimento_mulher AS STRING) = '1', 1, 0) AS encaminhamento_atendimento_mulher,
    -- [v3] Também é alvo: 9 (ignorado) e nulo (201 mil casos) -> NULL, em vez de "não encaminhada"
    CASE
        WHEN SAFE_CAST(dados.encaminhamento_delegacia_mulher AS STRING) = '1' THEN 1
        WHEN SAFE_CAST(dados.encaminhamento_delegacia_mulher AS STRING) = '2' THEN 0
        ELSE NULL
    END AS encaminhamento_delegacia_mulher,

    -- Target Principal (Ciclo de Reincidência)
    -- [v3] "Não" agora é 2 (antes 0). Com o mapeamento antigo o CSV sairia sem nenhum negativo.
    CASE
        WHEN SAFE_CAST(dados.outras_vezes_ocorrencia AS STRING) = '1' THEN 1
        WHEN SAFE_CAST(dados.outras_vezes_ocorrencia AS STRING) = '2' THEN 0
        ELSE NULL
    END AS target_reincidencia,

    -- [v3] Colunas novas
    CASE WHEN SAFE_CAST(dados.local_ocorrencia AS STRING) = '99' THEN NULL
         ELSE SAFE_CAST(dados.local_ocorrencia AS STRING) END AS local_ocorrencia_cod,   -- local completo (via pública, bar, escola...)
    EXTRACT(MONTH FROM SAFE_CAST(dados.data_ocorrencia AS DATE)) AS mes_ocorrencia,
    IF(EXTRACT(YEAR FROM SAFE_CAST(dados.data_ocorrencia AS DATE)) < dados.ano - 1, 1, 0) AS ocorrencia_antiga,
    dados.hora_num AS hora_ocorrencia_h,                                                  -- hora extraída (0–23)
    SAFE_CAST(dados.hora_ocorrencia AS STRING) AS hora_ocorrencia_raw                     -- para conferir o formato

FROM base_hora AS dados

LEFT JOIN diretorio_municipios AS mun_ocorrencia
    ON SAFE_CAST(dados.id_municipio_ocorrencia AS STRING) = mun_ocorrencia.id_municipio_7
    OR (dados.id_municipio_ocorrencia IS NULL AND SAFE_CAST(dados.id_municipio_6_ocorrencia AS STRING) = mun_ocorrencia.id_municipio_6)
