import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #scikit-learn
from sklearn.preprocessing import StandardScaler
import tratamentododataset as tratamento
import graficos.graficos as graficos
import graficos.correlacao as corr

df = tratamento.retornarDataframeTratado()

graficos.plotarQualidadePorOcupacao(df)
graficos.plotarDuracaoPorIdade(df)
graficos.plotarGraficoDispersao(df, 'Age', 'Quality of Sleep')
graficos.plotarGraficoApneaPorOcupacao(df)
graficos.plotarGraficoInsomniaPorOcupacao(df)
graficos.plotarGraficoNonePorOcupacao(df)

corr.plotarGraficoCorrelacao(df)



print(df.head())