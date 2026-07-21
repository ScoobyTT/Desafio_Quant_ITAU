"""Núcleo do pipeline: dados COTAHIST -> features -> modelo -> ranking.

Assume arquivo já processado pelo b3_cotahist.py (colunas: ticker, data_pregao,
tipo_mercado, preco_abertura/maximo/minimo/fechamento, volume_financeiro,
numero_negocios).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DB_PADRAO = "decisoes.db"

HORIZONTE_PADRAO = 5  # pregões à frente que o modelo tenta prever
MERCADO_VISTA = 10    # tipo_mercado da B3 para ações à vista

# Todos os parâmetros abaixo têm default 0 (custo/imposto desligado). Digite os
# valores reais na interface do app se quiser considerá-los na simulação.
# Referências de mercado (não são aplicadas automaticamente, é só consulta):
#   - Taxa B3 (negociação+CCP+TTA, não day trade, ADTV até R$3mi/mês): 0,0300%
#     Fonte: b3.com.br/pt_br/produtos-e-servicos/tarifas/.../a-vista (jul/2026)
#   - ISS sobre corretagem: 2% a 5% conforme município (LC 116/03); SP cobra 5%
#   - IRRF swing trade ("dedo-duro"): 0,005% sobre valor da venda, não retido
#     se o valor calculado for menor que R$1,00
#     Fonte: gov.br/receitafederal, seção Renda Variável > Isenções (jul/2026)
#   - IR swing trade: 15% sobre o ganho líquido mensal, isento se as vendas do
#     mês somarem até R$20.000 (Lei 11.033/2004, Art. 3º, II)
TAXA_B3_PADRAO = 0.0
CORRETAGEM_PADRAO = 0.0
CORRETAGEM_PERCENTUAL_PADRAO = 0.0
ISS_PADRAO = 0.0
IRRF_SWING_PCT = 0.0
IRRF_MINIMO = 1.0              # abaixo disso (em R$), a corretora não retém o IRRF
IR_SWING_ALIQUOTA = 0.0
ISENCAO_VENDAS_MENSAL = 0.0    # 0 = isenção nunca se aplica

FEATURE_COLS = [
    "retorno_1d", "retorno_5d", "retorno_20d",
    "volatilidade_20d", "media_5_sobre_20", "media_20_sobre_60",
    "volume_rel_20d", "amplitude_dia", "rsi_14", "momentum_10d",
]


def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_parquet(caminho) if caminho.endswith(".parquet") else pd.read_csv(
        caminho, parse_dates=["data_pregao"]
    )
    df = df[df["tipo_mercado"] == MERCADO_VISTA].copy()
    df = df.sort_values(["ticker", "data_pregao"])
    return df


def _rsi(precos: pd.Series, periodo: int = 14) -> pd.Series:
    delta = precos.diff()
    ganho = delta.clip(lower=0).rolling(periodo).mean()
    perda = (-delta.clip(upper=0)).rolling(periodo).mean()
    rs = ganho / perda.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def construir_features(df: pd.DataFrame, horizonte: int = HORIZONTE_PADRAO) -> pd.DataFrame:
    g = df.groupby("ticker", group_keys=False)
    fechamento = g["preco_fechamento"]

    df["retorno_1d"] = fechamento.pct_change(1)
    df["retorno_5d"] = fechamento.pct_change(5)
    df["retorno_20d"] = fechamento.pct_change(20)
    df["volatilidade_20d"] = g["preco_fechamento"].transform(lambda s: s.pct_change().rolling(20).std())
    media_5 = g["preco_fechamento"].transform(lambda s: s.rolling(5).mean())
    media_20 = g["preco_fechamento"].transform(lambda s: s.rolling(20).mean())
    df["media_5_sobre_20"] = media_5 / media_20 - 1
    media_60 = g["preco_fechamento"].transform(lambda s: s.rolling(60).mean())
    df["media_20_sobre_60"] = media_20 / media_60 - 1
    vol_medio_20 = g["volume_financeiro"].transform(lambda s: s.rolling(20).mean())
    df["volume_rel_20d"] = df["volume_financeiro"] / vol_medio_20 - 1
    df["amplitude_dia"] = (df["preco_maximo"] - df["preco_minimo"]) / df["preco_fechamento"]
    df["rsi_14"] = g["preco_fechamento"].transform(_rsi)
    df["momentum_10d"] = fechamento.pct_change(10)

    # alvo: retorno acumulado nos próximos `horizonte` pregões (shift negativo = futuro)
    df["retorno_futuro"] = g["preco_fechamento"].transform(
        lambda s: s.shift(-horizonte) / s - 1
    )
    return df


def _splits_walk_forward(datas_unicas: np.ndarray, n_splits: int = 5):
    """Gera cortes temporais crescentes (treino = passado, teste = bloco seguinte)."""
    blocos = np.array_split(datas_unicas, n_splits + 1)
    treino_ate = blocos[0]
    for bloco_teste in blocos[1:]:
        yield treino_ate, bloco_teste
        treino_ate = np.concatenate([treino_ate, bloco_teste])


def validar_walk_forward(
    dados_treino: pd.DataFrame, n_splits: int = 5, horizonte: int = HORIZONTE_PADRAO
) -> pd.DataFrame:
    """Validação honesta: treina só com o passado, testa no bloco futuro seguinte.

    Descarta as últimas `horizonte` datas de cada bloco de treino, porque o
    alvo (`retorno_futuro`) dessas linhas só se resolve depois do início do
    bloco de teste — sem esse corte, o treino "veria" indiretamente preços
    do período de teste através do próprio alvo.
    """
    datas = np.sort(dados_treino["data_pregao"].unique())
    resultados = []

    for datas_treino, datas_teste in _splits_walk_forward(datas, n_splits):
        datas_treino_seguras = datas_treino[:-horizonte] if horizonte > 0 else datas_treino
        treino = dados_treino[dados_treino["data_pregao"].isin(datas_treino_seguras)]
        teste = dados_treino[dados_treino["data_pregao"].isin(datas_teste)]
        if len(treino) < 200 or len(teste) < 20:
            continue

        modelo = HistGradientBoostingRegressor(random_state=42)
        modelo.fit(treino[FEATURE_COLS], treino["retorno_futuro"])
        previsto = modelo.predict(teste[FEATURE_COLS])
        real = teste["retorno_futuro"].to_numpy()

        acerto_direcao = float(np.mean(np.sign(previsto) == np.sign(real)))
        r2 = float(1 - np.sum((real - previsto) ** 2) / np.sum((real - real.mean()) ** 2))
        resultados.append({
            "treino_ate": pd.Timestamp(datas_treino_seguras.max()).date(),
            "n_linhas_treino": len(treino),
            "periodo_teste_inicio": pd.Timestamp(datas_teste.min()).date(),
            "periodo_teste_fim": pd.Timestamp(datas_teste.max()).date(),
            "n_amostras": len(teste),
            "acerto_direcao": round(acerto_direcao, 3),
            "r2": round(r2, 3),
        })

    return pd.DataFrame(resultados)


def treinar_modelo_final(dados_treino: pd.DataFrame) -> HistGradientBoostingRegressor:
    modelo = HistGradientBoostingRegressor(random_state=42)
    modelo.fit(dados_treino[FEATURE_COLS], dados_treino["retorno_futuro"])
    return modelo


def prever_ultimo_pregao(df_features: pd.DataFrame, modelo: HistGradientBoostingRegressor) -> pd.DataFrame:
    """Usa a linha mais recente de cada ticker (sem alvo, pois é o futuro real) para prever."""
    ultima_data = df_features["data_pregao"].max()
    atual = df_features[df_features["data_pregao"] == ultima_data].dropna(subset=FEATURE_COLS).copy()
    atual["retorno_previsto"] = modelo.predict(atual[FEATURE_COLS])
    return atual[["ticker", "data_pregao", "preco_fechamento", "retorno_previsto"]].sort_values(
        "retorno_previsto", ascending=False
    )


def _custo_operacional(
    valor_financeiro: float,
    taxa_b3: float,
    corretagem_fixa: float,
    corretagem_percentual: float,
    iss_pct: float,
) -> float:
    """Custo de UMA ponta (compra OU venda): taxa B3 + corretagem (fixa+%) + ISS sobre a corretagem."""
    corretagem = corretagem_fixa + valor_financeiro * corretagem_percentual
    return valor_financeiro * taxa_b3 + corretagem + corretagem * iss_pct


def _alocar_greedy(candidatos: pd.DataFrame, capital_disponivel: float, capital_max_por_acao: float) -> pd.DataFrame:
    """Percorre candidatos (já ordenados) comprando até `capital_max_por_acao` de cada,
    respeitando o capital disponível. Compra em lotes de ações inteiras."""
    linhas = []
    capital_restante = capital_disponivel
    for _, ativo in candidatos.iterrows():
        if capital_restante < ativo["preco_fechamento"]:
            continue
        valor_alvo = min(capital_max_por_acao, capital_restante)
        quantidade = int(valor_alvo // ativo["preco_fechamento"])
        if quantidade <= 0:
            continue
        valor_efetivo = quantidade * ativo["preco_fechamento"]
        capital_restante -= valor_efetivo
        linha = ativo.to_dict()
        linha["quantidade_acoes"] = quantidade
        linha["valor_alocado"] = round(valor_efetivo, 2)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def alocar_capital(
    ranking: pd.DataFrame,
    capital: float,
    top_n: int = 10,
    taxa_b3: float = TAXA_B3_PADRAO,
    corretagem_fixa: float = CORRETAGEM_PADRAO,
    corretagem_percentual: float = CORRETAGEM_PERCENTUAL_PADRAO,
    iss_pct: float = ISS_PADRAO,
    capital_max_por_acao: float | None = None,
) -> pd.DataFrame:
    """Aloca capital entre os top_n com retorno previsto positivo, líquido de custos.

    Compra até `capital_max_por_acao` por ativo (padrão: sem limite, usa todo o
    capital se necessário), na ordem do ranking, até esgotar o capital total.
    """
    custo_pct_ida_volta = 2 * (taxa_b3 + corretagem_percentual * (1 + iss_pct))
    selecionados = ranking.head(top_n).copy()
    selecionados["retorno_liquido_estimado"] = selecionados["retorno_previsto"] - custo_pct_ida_volta
    selecionados = selecionados[selecionados["retorno_liquido_estimado"] > 0]
    if selecionados.empty:
        return selecionados.assign(valor_alocado=[], quantidade_acoes=[], custo_estimado=[])

    limite = capital_max_por_acao if capital_max_por_acao is not None else capital
    carteira = _alocar_greedy(selecionados, capital, limite)
    if carteira.empty:
        return carteira.assign(custo_estimado=[])

    custo_por_ponta = carteira["valor_alocado"].apply(
        lambda v: _custo_operacional(v, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct)
    )
    carteira["custo_estimado"] = (2 * custo_por_ponta).round(2)  # compra + venda
    return carteira


def _conectar(caminho_db: str = DB_PADRAO) -> sqlite3.Connection:
    con = sqlite3.connect(caminho_db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS decisoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_decisao TEXT NOT NULL,
            ticker TEXT NOT NULL,
            preco_na_decisao REAL NOT NULL,
            retorno_previsto REAL NOT NULL,
            horizonte_pregoes INTEGER NOT NULL,
            valor_alocado REAL NOT NULL,
            quantidade_acoes INTEGER NOT NULL,
            preco_realizado REAL,
            retorno_realizado REAL,
            UNIQUE(data_decisao, ticker)
        )
    """)
    return con


def registrar_decisoes(carteira: pd.DataFrame, horizonte: int, caminho_db: str = DB_PADRAO) -> None:
    """Grava a carteira sugerida no histórico local, para conferir o acerto depois."""
    if carteira.empty:
        return
    with _conectar(caminho_db) as con:
        for _, linha in carteira.iterrows():
            con.execute(
                """INSERT OR IGNORE INTO decisoes
                   (data_decisao, ticker, preco_na_decisao, retorno_previsto,
                    horizonte_pregoes, valor_alocado, quantidade_acoes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(linha["data_pregao"].date()), linha["ticker"], float(linha["preco_fechamento"]),
                    float(linha["retorno_previsto"]), int(horizonte),
                    float(linha["valor_alocado"]), int(linha["quantidade_acoes"]),
                ),
            )


def reconciliar_historico(df_precos_atualizado: pd.DataFrame, caminho_db: str = DB_PADRAO) -> pd.DataFrame:
    """Para decisões antigas ainda sem resultado, busca o preço `horizonte` pregões à
    frente nos dados atualizados (se já disponível) e calcula o retorno realizado."""
    with _conectar(caminho_db) as con:
        pendentes = pd.read_sql(
            "SELECT * FROM decisoes WHERE retorno_realizado IS NULL", con,
            parse_dates=["data_decisao"],
        )
        if pendentes.empty:
            return pd.read_sql("SELECT * FROM decisoes", con, parse_dates=["data_decisao"])

        precos = df_precos_atualizado.sort_values(["ticker", "data_pregao"])
        for _, linha in pendentes.iterrows():
            serie = precos[
                (precos["ticker"] == linha["ticker"]) & (precos["data_pregao"] > linha["data_decisao"])
            ]
            if len(serie) < linha["horizonte_pregoes"]:
                continue  # ainda não passou tempo suficiente nos dados disponíveis
            preco_futuro = serie.iloc[int(linha["horizonte_pregoes"]) - 1]["preco_fechamento"]
            retorno = preco_futuro / linha["preco_na_decisao"] - 1
            con.execute(
                "UPDATE decisoes SET preco_realizado = ?, retorno_realizado = ? WHERE id = ?",
                (float(preco_futuro), float(retorno), int(linha["id"])),
            )

        return pd.read_sql("SELECT * FROM decisoes", con, parse_dates=["data_decisao"])


def backtest(
    df_features: pd.DataFrame,
    capital_total: float,
    capital_max_por_acao: float,
    horizonte: int = HORIZONTE_PADRAO,
    top_n: int = 10,
    taxa_b3: float = TAXA_B3_PADRAO,
    corretagem_fixa: float = CORRETAGEM_PADRAO,
    corretagem_percentual: float = CORRETAGEM_PERCENTUAL_PADRAO,
    iss_pct: float = ISS_PADRAO,
    irrf_pct: float = IRRF_SWING_PCT,
    min_dias_treino: int = 120,
    data_inicio: str | pd.Timestamp | None = None,
    data_fim: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula compra/venda ao longo do histórico disponível (ou de uma janela dele).

    A cada `horizonte` pregões, treina o modelo só com dados anteriores (sem
    vazamento), escolhe as ações com retorno previsto líquido positivo, compra
    limitado a `capital_max_por_acao` por ação até esgotar `capital_total`
    disponível naquele momento, e realiza o resultado usando o retorno real
    ocorrido (já calculado em `retorno_futuro`). Capital e lucro OPERACIONAL
    (antes de IR) são compostos entre os períodos. O IR/IRRF é calculado à
    parte por `calcular_ir_mensal`, pois no Brasil ele é apurado por mês, não
    por operação.

    Importante sobre o treino: só entram linhas cujo alvo (`retorno_futuro`,
    calculado com `horizonte` pregões de folga) já esteja totalmente resolvido
    ANTES da data de decisão. Sem esse corte, linhas de treino muito recentes
    carregariam informação de preço do próprio período de teste através do
    alvo, um vazamento sutil de dados do futuro. `treino_ate`, no resumo
    retornado, mostra exatamente até que data cada decisão usou dados.

    `data_inicio`/`data_fim` restringem as datas de DECISÃO (compra) simuladas:
    - Só `data_inicio` (== `data_fim`): simula um único dia de decisão.
    - Ambos preenchidos: simula todas as decisões dentro do intervalo (ainda
      espaçadas por `horizonte` pregões).
    - Nenhum: comportamento padrão, usa todo o histórico disponível.
    Em qualquer caso, o treino do modelo em cada decisão usa só dados
    anteriores à própria data de decisão (walk-forward), mesmo que
    `data_inicio` esteja no meio do histórico.

    Retorna (operacoes, resumo_por_periodo).
    """
    base = df_features.dropna(subset=FEATURE_COLS).copy()
    datas = np.sort(base["data_pregao"].unique())
    custo_pct_ida_volta = 2 * (taxa_b3 + corretagem_percentual * (1 + iss_pct))

    capital_disponivel = capital_total
    operacoes = []
    resumo = []

    i = min_dias_treino
    if data_inicio is not None:
        i = max(i, int(np.searchsorted(datas, np.datetime64(pd.Timestamp(data_inicio)))))
    limite_fim = np.datetime64(pd.Timestamp(data_fim)) if data_fim is not None else None

    while i < len(datas):
        data_decisao = datas[i]
        if limite_fim is not None and data_decisao > limite_fim:
            break

        corte_idx = i - horizonte  # última data cujo alvo (retorno_futuro) já está resolvido
        if corte_idx < 0:
            i += horizonte
            continue
        data_treino_ate = datas[corte_idx]
        treino = base[(base["data_pregao"] <= data_treino_ate) & base["retorno_futuro"].notna()]
        candidatos = base[base["data_pregao"] == data_decisao].dropna(subset=["retorno_futuro"])

        if len(treino) < 200 or candidatos.empty:
            i += horizonte
            continue

        modelo = HistGradientBoostingRegressor(random_state=42)
        modelo.fit(treino[FEATURE_COLS], treino["retorno_futuro"])
        candidatos = candidatos.copy()
        candidatos["retorno_previsto"] = modelo.predict(candidatos[FEATURE_COLS])
        candidatos["retorno_liquido_estimado"] = candidatos["retorno_previsto"] - custo_pct_ida_volta
        candidatos = candidatos[candidatos["retorno_liquido_estimado"] > 0].sort_values(
            "retorno_previsto", ascending=False
        ).head(top_n)

        selecionados = _alocar_greedy(candidatos, capital_disponivel, capital_max_por_acao)
        data_venda = pd.Timestamp(datas[i + horizonte]) if i + horizonte < len(datas) else pd.NaT

        lucro_periodo = 0.0
        for _, ativo in selecionados.iterrows():
            preco_compra = ativo["preco_fechamento"]
            valor_compra = ativo["valor_alocado"]
            preco_venda_real = preco_compra * (1 + ativo["retorno_futuro"])
            valor_venda = ativo["quantidade_acoes"] * preco_venda_real

            custo = _custo_operacional(valor_compra, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct) + \
                _custo_operacional(valor_venda, taxa_b3, corretagem_fixa, corretagem_percentual, iss_pct)
            irrf = valor_venda * irrf_pct
            irrf = irrf if irrf >= IRRF_MINIMO else 0.0

            lucro = valor_venda - valor_compra - custo
            lucro_periodo += lucro
            operacoes.append({
                "data_compra": pd.Timestamp(data_decisao).date(),
                "data_venda": data_venda.date() if pd.notna(data_venda) else None,
                "ticker": ativo["ticker"],
                "preco_compra": round(preco_compra, 2),
                "quantidade": int(ativo["quantidade_acoes"]),
                "preco_venda": round(preco_venda_real, 2),
                "valor_venda": round(valor_venda, 2),
                "retorno_previsto": round(ativo["retorno_previsto"], 4),
                "retorno_realizado": round(float(ativo["retorno_futuro"]), 4),
                "custo": round(custo, 2),
                "irrf": round(irrf, 2),
                "lucro": round(lucro, 2),
            })

        capital_disponivel += lucro_periodo
        resumo.append({
            "treino_de": pd.Timestamp(datas[0]).date(),
            "treino_ate": pd.Timestamp(data_treino_ate).date(),
            "n_linhas_treino": len(treino),
            "data_decisao": pd.Timestamp(data_decisao).date(),
            "data_venda_prevista": data_venda.date() if pd.notna(data_venda) else None,
            "n_operacoes": len(selecionados),
            "lucro_periodo": round(lucro_periodo, 2),
            "capital_apos_periodo": round(capital_disponivel, 2),
        })
        i += horizonte

    return pd.DataFrame(operacoes), pd.DataFrame(resumo)


def calcular_ir_mensal(
    operacoes: pd.DataFrame,
    ir_aliquota: float = IR_SWING_ALIQUOTA,
    isencao_vendas_mensal: float = ISENCAO_VENDAS_MENSAL,
) -> pd.DataFrame:
    """Apura IR mês a mês (regra brasileira de swing trade em ações).

    Regras aplicadas (Lei 11.033/2004 art. 3º, II + regulamentação da Receita
    Federal): lucro/prejuízo de cada mês é somado ao saldo de prejuízo acumulado
    de meses anteriores (compensação sem prazo de validade); se o total vendido
    no mês for menor ou igual a `isencao_vendas_mensal`, o IR do mês fica
    isento (mas o resultado ainda é usado para compensação de prejuízo);
    caso contrário, aplica-se `ir_aliquota` sobre o ganho líquido do mês, e o
    IRRF já retido nas vendas é abatido do valor a pagar (DARF).

    Simplificação assumida: eventual IRRF que exceda o IR devido no mês não é
    modelado como restituição (na prática, seria compensado/restituído na
    declaração anual). Retorna DataFrame vazio se `operacoes` estiver vazio.
    """
    if operacoes.empty:
        return pd.DataFrame()

    op = operacoes.dropna(subset=["data_venda"]).copy()
    op["data_venda"] = pd.to_datetime(op["data_venda"])
    op["mes"] = op["data_venda"].dt.to_period("M")

    mensal = op.groupby("mes").agg(vendas_mes=("valor_venda", "sum"), lucro_mes=("lucro", "sum"),
                                    irrf_retido_mes=("irrf", "sum")).reset_index().sort_values("mes")

    linhas = []
    prejuizo_acumulado = 0.0
    for _, row in mensal.iterrows():
        resultado = row["lucro_mes"] - prejuizo_acumulado
        if resultado <= 0:
            prejuizo_acumulado = -resultado
            ganho_tributavel = 0.0
        else:
            prejuizo_acumulado = 0.0
            ganho_tributavel = resultado

        isento = row["vendas_mes"] <= isencao_vendas_mensal
        ir_devido = 0.0 if isento else ganho_tributavel * ir_aliquota
        darf = max(0.0, ir_devido - row["irrf_retido_mes"])

        linhas.append({
            "mes": str(row["mes"]),
            "vendas_mes": round(row["vendas_mes"], 2),
            "lucro_mes": round(row["lucro_mes"], 2),
            "isento": isento,
            "prejuizo_acumulado_apos": round(prejuizo_acumulado, 2),
            "irrf_retido_mes": round(row["irrf_retido_mes"], 2),
            "ir_devido": round(ir_devido, 2),
            "darf_a_pagar": round(darf, 2),
        })

    return pd.DataFrame(linhas)