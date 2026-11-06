# Violência contra a mulher · modelos preditivos (SINAN) — v3

App Streamlit e notebook de treino de modelos supervisionados com notificações de violência interpessoal contra mulheres de 14 anos ou mais no SINAN (via Base dos Dados / BigQuery).

São três cenários, e cada um tem dois modelos: regressão logística e gradient boosting.

| Cenário | Alvo | Pergunta |
|---|---|---|
| Ciclo de Reincidência | `target_reincidencia` | A violência já havia ocorrido outras vezes? |
| Agressor Íntimo | `autor_parceiro_ou_ex` | O agressor é parceiro ou ex? |
| Encaminhamento DEAM | `encaminhamento_delegacia_mulher` | O caso foi encaminhado à Delegacia da Mulher? |

> Ferramenta de estudo. Os modelos aprendem padrões de **registro** e não de causas. Eles não substituem avaliação profissional e não devem orientar decisões sobre casos individuais.

## Arquivos

| Arquivo | O que é |
|---|---|
| `query_violencia_v3.sql` | Consulta no BigQuery que gera o CSV, já com os códigos atuais da Base dos Dados |
| `treino_modelos_v3.ipynb` | Notebook de treino para o Colab. Gera os modelos, as métricas e o `modelos_v3.zip` |
| `features.py` | Esquema de variáveis compartilhado entre o treino e o app. **Precisa ficar ao lado do `app.py`** |
| `app.py` | App Streamlit |
| `metrics.json`, `test_samples.json`, `model_*.joblib` | Artefatos gerados pelo notebook |
| `.streamlit/config.toml` | Tema do app |
| `requirements.txt` | Dependências. O notebook gera um com a versão exata do scikit-learn usada no treino |

## Como gerar os modelos (Colab)

1. Rode o `query_violencia_v3.sql` no BigQuery e salve o resultado em **Salvar resultados → CSV (Google Drive)**.
2. Abra o `treino_modelos_v3.ipynb` no Colab e ajuste o `CSV_PATH` na célula de configuração.
3. Rode tudo com *Ambiente de execução → Executar tudo*. O treino completo leva cerca de 40–60 min. Com `MODO_RAPIDO = True` leva uns 3 min, só para testar.
4. No final é baixado o `modelos_v3.zip`, e uma cópia vai para o Drive. Extraia o zip na raiz do repositório, substituindo os arquivos antigos.

Se a sessão cair no meio do treino, os cenários já concluídos ficam salvos. Rode de novo deixando em `CENARIOS` só os que faltaram.

> **Versão do scikit-learn:** o app precisa carregar os `.joblib` com a mesma versão usada no treino. O `requirements.txt` que vem no zip já fixa essa versão, então use esse.

## Como rodar o app

```bash
pip install -r requirements.txt
streamlit run app.py
```

No Streamlit Cloud basta fazer o commit dos arquivos acima. O `.python-version` usa 3.11.

## O que mudou na v3

**Dados**
- A consulta foi atualizada para os códigos atuais da Base dos Dados: sexo `F`, autoprovocada `2` = não, reincidência `2` = não.
- "Ignorado" deixou de virar "não". Isso vale para a hora, o local, o município e os alvos.
- A hora passou a ser lida do formato `HH:MM`.
- A escolaridade voltou a incluir os códigos 0 (analfabeta) e 8 (superior completo).

**Variáveis**
- São 41 candidatas: os 32 campos da ficha (incluindo hora, mês, local completo e vínculo do agressor) e 9 derivadas.
- Ficam de fora por vazamento os outros encaminhamentos, em todos os cenários, e o vínculo do agressor no cenário "Agressor Íntimo".

**Seleção**
- A seleção é feita separadamente para cada modelo em cada cenário. Usa eliminação recursiva guiada por importância de permutação.
- Fica o menor subconjunto cuja AUC está a até 1 erro-padrão da melhor.

**Modelos**
- Sem `class_weight`, então as probabilidades são calibradas.
- O limiar de decisão é escolhido pelo índice de Youden na validação.
- O GBDT usa categóricas nativas e early stopping.

**Avaliação**
- A configuração v2 é retreinada nos mesmos dados e na mesma divisão, como baseline.
- Há validação temporal: treino até 2021 e teste em 2022–2025.
- O desempenho também é medido por subgrupo: raça/cor, faixa etária e região.

**App**
- Tem 7 abas: visão geral (v2 × v3), desempenho com limiar interativo, seleção de variáveis, associações (com mapa em mosaico por UF), equidade, simulador e metodologia.
- O simulador mostra uma explicação local do caso e permite fazer "e se…?" em qualquer campo.

**Bug corrigido**
- Na v2, o simulador mandava `"4"` e o modelo esperava `"4.0"`. Por isso raça, escolaridade e situação conjugal não tinham efeito nas previsões.
- Agora a normalização acontece dentro do próprio pipeline (`features.prepare_features`).

## Lacunas conhecidas nos dados

- 2013 e 2014 estão incompletos no reprocessamento atual da Base dos Dados: 36 mil e 7 mil notificações, contra cerca de 77 mil e 84 mil em exportações anteriores.
- O campo de encaminhamento à DEAM só existe a partir de 2015, quando a ficha mudou.
- A hora não é informada em cerca de 38% dos casos.

---

Precisa de ajuda ou conhece alguém em situação de violência? **Ligue 180**, a Central de Atendimento à Mulher (gratuito, 24h). Em emergência, ligue **190**.
