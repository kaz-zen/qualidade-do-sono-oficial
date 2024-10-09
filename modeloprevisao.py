import pandas as pd
import tratamentododataset as tratamento
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = tratamento.retornarDataframeTratado()
# Substituir valores NaN na coluna 'Sleep Disorder' por 'No disorder'
df['Sleep Disorder'].fillna('No disorder', inplace=True)

# Mapear valores categóricos para números
occupation_mapping = {
    'Accountant': 0, 'Doctor': 1, 'Engineer': 2, 'Lawyer': 3, 'Manager': 4, 
    'Nurse': 5, 'Sales Representative': 6, 'Salesperson': 7, 'Scientist': 8, 
    'Software Engineer': 9, 'Teacher': 10
}

df['Occupation'] = df['Occupation'].map(occupation_mapping)

# Transformar outras colunas categóricas em números
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])
df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])

# Definir características e alvos
X = df[['Age', 'Occupation', 'Stress Level', 'Sleep Disorder']]
y = df[['Sleep Duration', 'Quality of Sleep']]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'R^2 Score: {r2}')
print(f'Mean Squared Error: {mse}')

# Função para fazer previsões baseadas em entrada do usuário
def predict_sleep(age, occupation, stress_level, sleep_disorder):
    # Codificar a entrada categórica
    occupation_encoded = occupation_mapping[occupation]
    sleep_disorder_encoded = label_encoder.transform([sleep_disorder])[0]
    
    # Criar um dataframe para a entrada
    input_data = pd.DataFrame([[age, occupation_encoded, stress_level, sleep_disorder_encoded]], 
                              columns=['Age', 'Occupation', 'Stress Level', 'Sleep Disorder'])
    
    # Fazer a previsão
    prediction = model.predict(input_data)
    
    return prediction

# Exemplo de uso da função de previsão
age = 25
occupation = 'Software Engineer'
stress_level = 3
sleep_disorder = 'No disorder'

predicted_sleep_duration, predicted_quality_of_sleep = predict_sleep(age, occupation, stress_level, sleep_disorder)[0]
print(f'Predicted Sleep Duration: {predicted_sleep_duration}')
print(f'Predicted Quality of Sleep: {predicted_quality_of_sleep}')