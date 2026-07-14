 7.2                                                                                                  download_arquivos.R                                                                                                           
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
import zipfile
import pandas as pd

# baixa bid
dl(year='2024', month='6', pair='eurusd', platform=P.NINJA_TRADER, time_frame=TF.TICK_DATA_BID)

# baixa ask (ano corrigido pra 2024)
dl(year='2024', month='6', pair='eurusd', platform=P.NINJA_TRADER, time_frame=TF.TICK_DATA_ASK)

# extrai bid
with zipfile.ZipFile("DAT_NT_EURUSD_T_BID_202406.zip", "r") as zip_ref:
    nome_bid = zip_ref.namelist()[0]
    zip_ref.extractall(".")

# extrai ask
with zipfile.ZipFile("DAT_NT_EURUSD_T_ASK_202406.zip", "r") as zip_ref:
    nome_ask = zip_ref.namelist()[0]
    zip_ref.extractall(".")

print("Arquivo bid:", nome_bid)
print("Arquivo ask:", nome_ask)

# lê os dois
df_bid = pd.read_csv(nome_bid, sep=";", header=None, names=["timestamp", "bid", "volume_B"])
df_ask = pd.read_csv(nome_ask, sep=";", header=None, names=["timestamp", "ask", "volume_A"])

df_bid["timestamp"] = pd.to_datetime(df_bid["timestamp"], format="%Y%m%d %H%M%S%f")
df_ask["timestamp"] = pd.to_datetime(df_ask["timestamp"], format="%Y%m%d %H%M%S%f")

df = pd.merge(df_bid[["timestamp", "bid","volume_B"]], df_ask[["timestamp", "ask","volume_A"]], on="timestamp", how="outer")
df = df.sort_values("timestamp")

print(df.head())


