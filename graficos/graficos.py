import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotarQualidadePorOcupacao(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Occupation', y='Quality of Sleep', data=df, ci=None)
    plt.title('Qualidade do Sono por Ocupação')
    plt.xlabel('Ocupação')
    plt.ylabel('Qualidade do Sono')
    plt.xticks(rotation=45)
    plt.show()

def plotarDuracaoPorIdade(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Age', y='Sleep Duration', data=df, ci=None)
    plt.title('Relação entre Duração do Sono e Idade')
    plt.xlabel('Idade')
    plt.ylabel('Duração do Sono')
    plt.show()

def plotarGraficoDispersao(df ,x, y):
    plt.scatter(df[x], df[y])

    plt.title(f'Gráfico de Dispersão: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()

def plotarGraficoPizzaDisturbioSono(df):
    contagem = df['Occupation'].value_counts()
    plt.pie(contagem, labels = contagem.index, autopct='%1.1f%%')

    sleepDisorderName = df['Sleep Disorder'].iloc[0]
    plt.title('Profissões com '+ sleepDisorderName)
    plt.show()

def plotarGraficoApneaPorOcupacao(df):
    df_apneia = df[df['Sleep Disorder'] == 'Sleep Apnea']
    plotarGraficoPizzaDisturbioSono(df_apneia)

def plotarGraficoInsomniaPorOcupacao(df):
    df_insomnia = df[df['Sleep Disorder'] == 'Insomnia']
    plotarGraficoPizzaDisturbioSono(df_insomnia)

def plotarGraficoNonePorOcupacao(df):
    df_none = df[df['Sleep Disorder'] == 'No disorder']
    plotarGraficoPizzaDisturbioSono(df_none)
