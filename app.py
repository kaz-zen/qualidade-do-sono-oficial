import streamlit as st
import modeloprevisao as mp

# Interface do Streamlit
st.title('Previsão da Qualidade do Sono')

age = st.number_input('Idade', min_value=0, max_value=100, value=25)
occupation = st.selectbox('Ocupação', list(mp.occupation_mapping.keys()))
stress_level = st.slider('Nível de Estresse', min_value=0, max_value=10, value=3)
sleep_disorder = st.selectbox('Transtorno do Sono', ['No disorder', 'Insomnia', 'Sleep Apnea'])

if st.button('Prever'):
    prediction = mp.predict_sleep(age, occupation, stress_level, sleep_disorder)
    predicted_sleep_duration, predicted_quality_of_sleep = prediction[0]
    st.write(f'Duração Prevista do Sono: {predicted_sleep_duration:.2f} horas')
    st.write(f'Qualidade Prevista do Sono: {predicted_quality_of_sleep:.2f}')

# Avaliações do modelo
st.write(f'R^2 Score: {mp.r2}')
st.write(f'Mean Squared Error: {mp.mse}')