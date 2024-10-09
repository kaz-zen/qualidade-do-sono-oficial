import pandas as pd

def retornarDataframeTratado():
    url = "Dataset\\Sleep_health_and_lifestyle_dataset.csv"
    df = pd.read_csv(url)
    dfTratado = tratarDataframe(df)
    return dfTratado

def tratarDataframe(df):
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No disorder')
    return df
