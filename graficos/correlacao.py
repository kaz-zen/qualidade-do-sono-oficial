import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder #scikit-learn
from sklearn.preprocessing import StandardScaler

def plotarGraficoCorrelacao(dataframe):
    ### Normalizando valores e imprimindo matriz de correlação
    # Copiar o DataFrame original
    df_normalized = dataframe.copy()

    # Mapear valores categóricos para números
    gender_mapping = {'Male': 0, 'Female': 1}
    occupation_mapping = {'Accountant': 0, 'Doctor': 1, 'Engineer': 2, 'Lawyer': 3, 'Manager': 4, 'Nurse': 5, 'Sales Representative': 6, 'Salesperson': 7, 'Scientist': 8, 'Software Engineer': 9, 'Teacher': 10}
    bmi_mapping = {'Normal': 0, 'Normal Weight': 1, 'Obese': 2, 'Overweight': 3}
    blood_pressure_mapping = {'115/75': 0, '115/78': 1, '117/76': 2, '118/75': 3, '115/76': 4, '119/77': 5, '120/80': 6, '121/79': 7, '122/80': 8, '125/80': 9, '125/82': 10, '126/83': 11, '128/84': 12, '128/85': 13, '129/84': 14, '130/85': 15, '130/86': 16, '131/86': 17, '132/87': 18, '135/88': 19, '136/89': 20, '137/89': 21, '135/90': 22, '139/91': 23, '140/90': 24, '140/95': 25, '142/92': 26}
    sleep_disorder_mapping = {'Insomnia': 0, 'No disorder': 1, 'Sleep Apnea': 2}

    # Aplicar mapeamento para as colunas correspondentes
    df_normalized['Gender'] = df_normalized['Gender'].map(gender_mapping)
    df_normalized['Occupation'] = df_normalized['Occupation'].map(occupation_mapping)
    df_normalized['BMI Category'] = df_normalized['BMI Category'].map(bmi_mapping)
    df_normalized['Blood Pressure'] = df_normalized['Blood Pressure'].map(blood_pressure_mapping)
    df_normalized['Sleep Disorder'] = df_normalized['Sleep Disorder'].map(sleep_disorder_mapping)

    # Normalizar todas as colunas
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

    # Remover a coluna 'Person ID'
    df_corr = df_normalized.drop(columns=['Person ID'])

    # Calcular a matriz de correlação
    correlation_matrix = df_corr.corr()

    # Gerar o gráfico de correlação (heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    return plt.show()

