import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Análise B3 (ML)", layout="wide")
st.title("Análise de carteira B3 com machine learning")

st.warning(
    "⚠️ **Aviso:** este é um modelo estatístico experimental, não uma recomendação de "
    "investimento. Retornos previstos podem não se realizar. Estudos acadêmicos (FGV EESP) "
    "mostram que a maioria dos investidores de curto prazo perde dinheiro na bolsa "
    "brasileira. Use a aba **Validação** abaixo para julgar se o modelo tem algum poder "
    "preditivo real antes de confiar nos números."
)

arquivo = st.file_uploader("Arquivo processado (parquet ou csv, saída do b3_cotahist.py)", type=["parquet", "csv"])
col1, col2, col3, col4 = st.columns(4)
capital = col1.number_input("Capital total (R$)", min_value=0.0, value=100000.0, step=1000.0)
capital_max_por_acao = col2.number_input("Máximo por ação (R$)", min_value=0.0, value=1000.0, step=100.0)
horizonte = col3.number_input("Horizonte de previsão (pregões à frente)", min_value=1, value=core.HORIZONTE_PADRAO)
top_n = col4.number_input("Nº máximo de ativos por período", min_value=1, value=10)

st.caption(
    "Todos os custos e impostos abaixo começam **zerados**. Preencha com os valores da "
    "sua corretora e da sua situação fiscal; o app não assume nenhum valor por conta própria."
)

with st.expander("Custos de transação (B3 + corretagem + ISS)"):
    c1, c2, c3 = st.columns(3)
    taxa_b3_pct = c1.number_input(
        "Taxa B3 por operação (%)", min_value=0.0, value=core.TAXA_B3_PADRAO * 100,
        step=0.001, format="%.4f",
        help="Negociação + CCP + TTA, por ponta (compra ou venda). Consulte b3.com.br/tarifas.",
    )
    corretagem_fixa = c2.number_input(
        "Corretagem fixa por ordem (R$)", min_value=0.0, value=core.CORRETAGEM_PADRAO, step=1.0,
        help="Muitas corretoras cobram R$0 para ações. Confira a tabela da sua corretora.",
    )
    corretagem_percentual = c3.number_input(
        "Corretagem percentual (%)", min_value=0.0, value=core.CORRETAGEM_PERCENTUAL_PADRAO * 100,
        step=0.01, format="%.4f",
        help="Se sua corretora cobrar um % do valor da ordem em vez de (ou além de) valor fixo.",
    )
    iss_pct = st.number_input(
        "ISS sobre a corretagem (%)", min_value=0.0, value=core.ISS_PADRAO * 100, step=0.1,
        help="Imposto municipal sobre o serviço de corretagem (varia de 2% a 5% conforme o município). "
             "Se a corretagem é zero, o ISS também é zero.",
    )

with st.expander("Impostos sobre o resultado (IRRF + IR mensal, regra de swing trade)"):
    d1, d2, d3 = st.columns(3)
    irrf_pct = d1.number_input(
        "IRRF sobre venda / \"dedo-duro\" (%)", min_value=0.0, value=core.IRRF_SWING_PCT * 100,
        step=0.001, format="%.5f",
        help="Retenção na fonte a cada venda, abatida do IR devido no mês. Regra oficial (swing trade): 0,005%.",
    )
    ir_aliquota_pct = d2.number_input(
        "Alíquota de IR sobre o lucro mensal (%)", min_value=0.0, value=core.IR_SWING_ALIQUOTA * 100,
        step=1.0, help="Regra oficial (swing trade): 15% sobre o ganho líquido do mês, após compensar prejuízos.",
    )
    isencao_mensal = d3.number_input(
        "Isenção de IR se vendas do mês ≤ (R$)", min_value=0.0, value=core.ISENCAO_VENDAS_MENSAL,
        step=1000.0, help="Regra oficial (swing trade): R$ 20.000/mês em vendas totais.",
    )
    st.caption(
        "Cálculo mês a mês: prejuízo de um mês é automaticamente compensado com lucro de meses "
        "seguintes (sem prazo de validade). Simplificação: IRRF que exceda o IR devido no mês não "
        "é modelado como restituição."
    )

if arquivo and st.button("Treinar e analisar"):
    with st.spinner("Carregando dados e construindo features..."):
        caminho = f"/tmp/{arquivo.name}"
        with open(caminho, "wb") as f:
            f.write(arquivo.getbuffer())
        df = core.carregar_dados(caminho)
        df = core.construir_features(df, horizonte=horizonte)
        dados_treino = df.dropna(subset=core.FEATURE_COLS + ["retorno_futuro"])

    if len(dados_treino) < 500:
        st.error("Poucos dados após limpeza para treinar com confiança. Baixe um período maior.")
        st.stop()

    with st.spinner("Rodando validação walk-forward..."):
        validacao = core.validar_walk_forward(dados_treino, horizonte=int(horizonte))

    with st.spinner("Treinando modelo final e gerando ranking..."):
        modelo = core.treinar_modelo_final(dados_treino)
        ranking = core.prever_ultimo_pregao(df, modelo)
        carteira = core.alocar_capital(
            ranking, capital=capital, top_n=int(top_n),
            taxa_b3=taxa_b3_pct / 100, corretagem_fixa=corretagem_fixa,
            corretagem_percentual=corretagem_percentual / 100, iss_pct=iss_pct / 100,
            capital_max_por_acao=capital_max_por_acao,
        )
        core.registrar_decisoes(carteira, horizonte=int(horizonte))

    # Guarda tudo em session_state: botões dentro das abas (ex: "Rodar backtest")
    # disparam um rerun do script, e sem isso o resultado do treino se perderia.
    st.session_state.resultado = dict(
        df=df, validacao=validacao, ranking=ranking, carteira=carteira,
        capital=capital, capital_max_por_acao=capital_max_por_acao,
        horizonte=int(horizonte), top_n=int(top_n),
        taxa_b3=taxa_b3_pct / 100, corretagem_fixa=corretagem_fixa,
        corretagem_percentual=corretagem_percentual / 100, iss_pct=iss_pct / 100,
        irrf_pct=irrf_pct / 100, ir_aliquota=ir_aliquota_pct / 100,
        isencao_mensal=isencao_mensal,
    )

if "resultado" in st.session_state:
    r = st.session_state.resultado
    df, validacao, ranking, carteira = r["df"], r["validacao"], r["ranking"], r["carteira"]

    aba_carteira, aba_validacao, aba_historico, aba_backtest = st.tabs([
        "Carteira sugerida (hoje)", "Validação (honestidade do modelo)",
        "Histórico de decisões", "Backtest completo",
    ])

    with aba_carteira:
        st.subheader(f"Ranking completo ({ranking['data_pregao'].max().date()})")
        st.dataframe(
            ranking.rename(columns={"preco_fechamento": "preço", "retorno_previsto": "retorno previsto"}),
            use_container_width=True,
        )

        st.subheader("Alocação sugerida do capital informado")
        if carteira.empty:
            st.info(
                "Nenhum ativo com retorno previsto **líquido de custos** positivo hoje. "
                "O modelo sugere não alocar capital."
            )
        else:
            st.dataframe(
                carteira[[
                    "ticker", "preco_fechamento", "retorno_previsto", "retorno_liquido_estimado",
                    "valor_alocado", "quantidade_acoes", "custo_estimado",
                ]],
                use_container_width=True,
            )
            st.caption(
                f"Total alocado: R$ {carteira['valor_alocado'].sum():.2f} de R$ {r['capital']:.2f} | "
                f"Custo estimado (B3 + corretagem + ISS, ida e volta): R$ {carteira['custo_estimado'].sum():.2f}. "
                "Ativos cujo retorno previsto não cobre os custos de ida e volta já foram excluídos. "
                "Impostos sobre o lucro (IR/IRRF) não entram aqui, só no Backtest (são apurados por mês)."
            )
            st.caption("Essa carteira foi registrada no histórico local (decisoes.db) para conferência futura.")

    with aba_validacao:
        st.write(
            "Cada linha treina o modelo só com dados **até** aquele ponto e testa no bloco "
            "seguinte (nunca viu esses dados). `acerto_direcao` = 0.5 é equivalente a "
            "cara-ou-coroa; `r2` negativo significa que o modelo é pior que simplesmente "
            "prever a média histórica."
        )
        st.dataframe(validacao, use_container_width=True)
        media_acerto = validacao["acerto_direcao"].mean() if not validacao.empty else float("nan")
        st.metric("Acerto direcional médio (fora da amostra)", f"{media_acerto:.1%}")

    with aba_historico:
        st.write(
            "Decisões sugeridas em execuções anteriores deste app. O retorno realizado só "
            "aparece quando você já tiver dados novos o suficiente (passados `horizonte` "
            "pregões da data da decisão) no arquivo carregado."
        )
        historico = core.reconciliar_historico(df)
        if historico.empty:
            st.info("Ainda não há decisões registradas.")
        else:
            st.dataframe(historico, use_container_width=True)
            concluidas = historico.dropna(subset=["retorno_realizado"])
            if not concluidas.empty:
                acerto = float((concluidas["retorno_realizado"] > 0).mean())
                st.metric("Decisões concluídas com retorno realizado positivo", f"{acerto:.1%}")
                st.caption(f"Baseado em {len(concluidas)} decisão(ões) com resultado já conhecido.")

    with aba_backtest:
        modo_periodo = st.radio(
            "Período a simular", ["Histórico completo", "Um dia específico", "Intervalo de datas"],
            horizontal=True,
        )
        data_min = pd.Timestamp(df["data_pregao"].min()).date()
        data_max = pd.Timestamp(df["data_pregao"].max()).date()
        data_inicio = data_fim = None
        if modo_periodo == "Um dia específico":
            dia = st.date_input("Data da decisão de compra", value=data_max, min_value=data_min, max_value=data_max)
            data_inicio = data_fim = dia
        elif modo_periodo == "Intervalo de datas":
            intervalo = st.date_input(
                "Intervalo de datas de decisão", value=(data_min, data_max),
                min_value=data_min, max_value=data_max,
            )
            if isinstance(intervalo, tuple) and len(intervalo) == 2:
                data_inicio, data_fim = intervalo
            else:
                st.info("Selecione a data final do intervalo para continuar.")

        st.write(
            f"Simula compra a cada {r['horizonte']} pregões {'no dia selecionado' if modo_periodo == 'Um dia específico' else 'dentro do período selecionado' if modo_periodo == 'Intervalo de datas' else 'ao longo de todo o histórico'} "
            f"disponível: compra até R$ {r['capital_max_por_acao']:.2f} por ativo, respeitando o "
            f"capital de R$ {r['capital']:.2f}, mantém por {r['horizonte']} pregões e vende, usando o "
            "retorno **real** que aconteceu (não é hipotético). Pode demorar, pois retreina o "
            "modelo a cada período, sem olhar o futuro em nenhum momento."
        )
        if st.button("Rodar backtest completo"):
            with st.spinner("Simulando compra e venda ao longo do histórico..."):
                operacoes, resumo = core.backtest(
                    df, capital_total=r["capital"], capital_max_por_acao=r["capital_max_por_acao"],
                    horizonte=r["horizonte"], top_n=r["top_n"],
                    taxa_b3=r["taxa_b3"], corretagem_fixa=r["corretagem_fixa"],
                    corretagem_percentual=r["corretagem_percentual"], iss_pct=r["iss_pct"],
                    irrf_pct=r["irrf_pct"], data_inicio=data_inicio, data_fim=data_fim,
                )
                ir_mensal = core.calcular_ir_mensal(
                    operacoes, ir_aliquota=r["ir_aliquota"], isencao_vendas_mensal=r["isencao_mensal"]
                )
                st.session_state.backtest_resultado = (operacoes, resumo, ir_mensal, r["capital"])

        if "backtest_resultado" in st.session_state:
            operacoes, resumo, ir_mensal, capital_inicial = st.session_state.backtest_resultado
            if resumo.empty:
                st.info("Histórico insuficiente para simular nenhum período completo.")
            else:
                capital_final_bruto = resumo["capital_apos_periodo"].iloc[-1]
                total_darf = ir_mensal["darf_a_pagar"].sum() if not ir_mensal.empty else 0.0
                capital_final_liquido = capital_final_bruto - total_darf

                m1, m2, m3 = st.columns(3)
                m1.metric("Capital final (antes de IR)", f"R$ {capital_final_bruto:,.2f}",
                          f"{(capital_final_bruto/capital_inicial-1)*100:.2f}%")
                m2.metric("DARF total pago no período", f"R$ {total_darf:,.2f}")
                m3.metric("Capital final (líquido de IR)", f"R$ {capital_final_liquido:,.2f}",
                          f"{(capital_final_liquido/capital_inicial-1)*100:.2f}%")

                st.line_chart(resumo.set_index("data_decisao")["capital_apos_periodo"])
                st.subheader("Resumo por período (custos operacionais já descontados, IR ainda não)")
                st.dataframe(resumo, use_container_width=True)

                st.subheader("Apuração de IR mensal (DARF)")
                if ir_mensal.empty:
                    st.info("Nenhuma venda com data suficiente para apurar IR mensal.")
                else:
                    st.dataframe(ir_mensal, use_container_width=True)

                st.subheader("Todas as operações simuladas")
                st.dataframe(operacoes, use_container_width=True)