import yfinance as yf
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

def baixar_e_salvar(ticker, nome_arquivo, period="2y", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    # o nome da coluna de data muda dependendo do interval (Date pra diário, Datetime pra intraday)
    col_data = "Datetime" if "Datetime" in df.columns else "Date"

    caminho = os.path.join("data/raw", nome_arquivo)
    df.to_csv(caminho, index=False)

    print(f"{ticker} | Linhas: {len(df)} | {df[col_data].min()} até {df[col_data].max()}")
    print(f"Salvo em: {caminho}")
    print()
    return df

df_vale = baixar_e_salvar("VALE3.SA", "vale3_1h.csv")
df_ibov = baixar_e_salvar("^BVSP", "ibovespa_1h.csv")










