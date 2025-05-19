import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from datetime import timedelta
import ipeadatapy as ip

# Função para carregar os dados da API
@st.cache_data
def carregar_dados():
    df = ip.timeseries('EIA366_PBRENT366', yearGreaterThan=1999)
    df = df.rename(columns={'VALUE (US$)': 'Preco'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    df = df[['Preco']]
    return df

# Função para exibir a série histórica

def mostrar_serie_historica(df):
    st.subheader("\U0001F4C8 Série Histórica do Preço do Petróleo (USD)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['Preco'], label='Preço Real', linewidth=1.5, color='#1f77b4')
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (USD)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

# Gráfico com eventos
def mostrar_insights_com_grafico(df):
    st.subheader("\U0001F4CA Impacto de Eventos no Preço do Petróleo")
    st.write('Considerando os últimos 6 anos de dados, notamos aqui alguns eventos impactantes e uma certa estabilização na variância do valor:')
    df = df[df.index >= df.index.max() - pd.DateOffset(years=6)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df['Preco'], label='Preço Brent', linewidth=1.5, color='#1f77b4')

    eventos = {
        "Invasão da Ucrânia": "2022-02-24",
        "Corte OPEP+": "2020-04-12",
        "Início da Pandemia": "2020-03-11",
        "COP26": "2021-11-01"
    }

    y_max = df['Preco'].max()
    y_min = df['Preco'].min()
    altura_anotacao = y_min + (y_max - y_min) * 0.6

    for i, (evento, data) in enumerate(eventos.items()):
        data = pd.to_datetime(data)
        if data >= df.index.min():
            ax.axvline(data, color='red', linestyle='--', alpha=0.7)
            deslocamento_dias = 20 * ((i % 2) + 1)
            ax.annotate(evento,
                        xy=(data, y_max * 0.7),
                        xytext=(data + pd.Timedelta(days=deslocamento_dias), altura_anotacao - i * 10),
                        textcoords='data',
                        ha='left',
                        va='bottom',
                        fontsize=9,
                        color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.set_title("Preço do Petróleo Brent com Eventos Relevantes (6 anos)")
    ax.set_xlabel("Ano")
    ax.set_ylabel("USD/barril")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

# Gráficos de médias móveis
def mostrar_medias_moveis(df):
    df_plot = df.copy()
    df_plot['MM_30'] = df_plot['Preco'].rolling(window=30).mean()
    df_plot['MM_90'] = df_plot['Preco'].rolling(window=90).mean()

    st.subheader("📊 Médias Móveis (Tendência)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_plot.index, df_plot['Preco'], label='Preço Real', alpha=0.5, color='#1f77b4')
    ax.plot(df_plot.index, df_plot['MM_30'], label='Média Móvel 30 dias', color='orange')
    ax.plot(df_plot.index, df_plot['MM_90'], label='Média Móvel 90 dias', color='green')
    ax.set_title("Tendência do Preço com Médias Móveis")
    ax.set_xlabel("Ano")
    ax.set_ylabel("USD/barril")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

    st.subheader("📊 Médias Móveis (Últimos 5 anos)")
    fig, ax = plt.subplots(figsize=(14, 4))
    df_plot = df_plot[df_plot.index >= df_plot.index.max() - pd.DateOffset(years=5)]
    ax.plot(df_plot.index, df_plot['Preco'], label='Preço Real', alpha=0.5, color='#1f77b4')
    ax.plot(df_plot.index, df_plot['MM_30'], label='Média Móvel 30 dias', color='orange')
    ax.plot(df_plot.index, df_plot['MM_90'], label='Média Móvel 90 dias', color='green')
    ax.set_title("Tendência Recente com Médias Móveis")
    ax.set_xlabel("Ano")
    ax.set_ylabel("USD/barril")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
    st.markdown("""
    📌 **Insight adicional:**
    No gráfico das médias móveis, é possível observar que após a invasão da Ucrânia, o petróleo se manteve em um patamar elevado, mas com alta volatilidade. 
    As curvas de 30 e 90 dias demonstram tendência de convergência apenas a partir de meados de 2023, indicando estabilidade relativa após choques externos.
    """)

# Função para exibir variação percentual mensal
def mostrar_variacao_percentual(df):
    df_mensal = df.resample('M').mean()
    df_mensal['Variação %'] = df_mensal['Preco'].pct_change() * 100

    st.subheader("📉 Variação Percentual Mensal")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df_mensal.index, df_mensal['Variação %'], color='#ff7f0e', width=20)
    ax.set_title("Variação Mensal do Preço do Petróleo")
    ax.set_xlabel("Mês")
    ax.set_ylabel("% de Variação")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

# Função para exibir variação percentual mensal (últimos 5 anos)
    st.subheader("📉 Variação Percentual Mensal (Últimos 5 anos)")
    fig, ax = plt.subplots(figsize=(14, 4))
    df_mensal = df_mensal[df_mensal.index >= df_mensal.index.max() - pd.DateOffset(years=5)]
    ax.bar(df_mensal.index, df_mensal['Variação %'], color='#ff7f0e', width=20)
    ax.set_title("Variação Mensal do Preço do Petróleo (Recente)")
    ax.set_xlabel("Mês")
    ax.set_ylabel("% de Variação")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

    # Insight adicional
    st.markdown("""
    📌 **Insight complementar sobre volatilidade:**
    A variação percentual mensal evidencia períodos de forte oscilação, especialmente em 2020 e 2022. 
    Isso reflete o impacto direto de eventos como a pandemia e a guerra na Ucrânia.
    Em contrapartida, entre 2023 e 2024, observa-se uma tendência de normalização com flutuações mensais menos extremas, 
    indicando uma possível acomodação do mercado internacional diante de choques anteriores.
    """)

# Função pra exportar previsão para csv
def gerar_csv_previsao(datas, previsoes):
    df = pd.DataFrame({
        'Data': datas.strftime('%d/%m/%Y'),
        'Preço Previsto (USD)': previsoes
    })
    return df.to_csv(index=False).encode('utf-8')

# Função para prever usando o modelo XGBoost
def prever_xgboost(modelo, df):
    scaler = joblib.load("scaler_xgb.pkl")
    dados = df['Preco'].values.reshape(-1, 1)
    dados_norm = scaler.transform(dados)
    ultimos = dados_norm[-11:].flatten()
    previsoes = []
    for _ in range(15):
        pred = modelo.predict([ultimos])[0]
        previsoes.append(pred)
        ultimos = np.append(ultimos[1:], pred)
    previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1)).flatten()
    datas_previstas = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=15)
    return datas_previstas, previsoes

# Função para prever usando LSTM
def prever_lstm(modelo, df):
    scaler = joblib.load("scaler_lstm.pkl")
    dados = df['Preco'].values.reshape(-1, 1)
    dados_norm = scaler.transform(dados)
    ultimos = dados_norm[-30:].reshape(1, 30, 1)
    previsoes = []
    for _ in range(15):
        pred = modelo.predict(ultimos)[0][0]
        previsoes.append(pred)
        ultimos = np.append(ultimos[:, 1:, :], [[[pred]]], axis=1)
    previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1)).flatten()
    datas_previstas = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=15)
    return datas_previstas, previsoes

# Streamlit App
st.set_page_config(layout="wide")
st.title("\U0001F4CA Dashboard de Previsão do Preço do Petróleo")
dados = carregar_dados()

with st.sidebar:
    aba_selecionada = option_menu(
        menu_title="Navegação",
        options=["📘 Contexto e Insights", "📊 Histórico", "🔮 Previsão"],
        icons=["Ball", "Ball", "Ball"],
        menu_icon=None,
        default_index=0,
        orientation="vertifcal"
    )
    st.markdown("""
    Grupo 8:
    - Marlon Jobim Fernandez (RM353490)
    - Roberto Cavedon Muller (RM353491)
    """)

# ABA 1: CONTEXTO
if aba_selecionada == "📘 Contexto e Insights":
    st.title("\U0001F4D8 Contexto do Projeto e Insights")
    st.markdown("""
    Este dashboard foi desenvolvido como parte do **POSTECH Tech Challenge – Fase 4**, com o objetivo de fornecer **insights estratégicos** sobre o preço do petróleo Brent e permitir **previsões automatizadas** para apoio à tomada de decisão.

    ### ✨ Objetivo
    - Visualizar a evolução histórica do preço do petróleo
    - Observar comportamentos sazonais e anomalias
    - Realizar previsões com modelos de Machine Learning (XGBoost e LSTM)

    ### 🔎 Insights Relevantes com Impacto no Preço

    1. **📌 Crise Geopolítica: Guerra na Ucrânia (fev/2022)**  
       A invasão da Ucrânia pela Rússia desencadeou sanções, corte na oferta e especulações. O petróleo Brent ultrapassou **120 USD**.

    2. **📉 Queda de Demanda: COVID-19 (mar/2020)**  
       A pandemia provocou colapso na demanda por energia. O preço caiu abaixo de **20 USD** devido à retração global.

    3. **⚖️ Ação Coordenada: Corte da OPEP+ (abr/2020)**  
       Para conter a queda de preços, a OPEP+ cortou a produção, promovendo estabilização gradativa.

    4. **🌱 Pressão Verde: COP26 e ESG (nov/2021)**  
       Incentivos à transição energética e pressão por fontes limpas geraram volatilidade nas expectativas futuras.
    """)

    mostrar_insights_com_grafico(dados)

# ABA 2: HISTÓRICO
elif aba_selecionada == "📊 Histórico":
    st.subheader("📈 Evolução Histórica do Preço do Petróleo Brent")
    st.markdown("""
    Esta aba apresenta a série histórica do petróleo Brent desde 2000 até o momento atual.
    Com isso, podemos observar o comportamento do mercado global de energia ao longo dos anos, 
    destacando tendências de curto e médio prazo, bem como períodos de alta volatilidade.
    """)

    mostrar_serie_historica(dados)

    st.markdown("---")
    mostrar_medias_moveis(dados)

    st.markdown("---")
    mostrar_variacao_percentual(dados)


# ABA 3: PREVISAO
elif aba_selecionada == "🔮 Previsão":
    st.markdown("## 🔮 Previsão do Preço do Petróleo Brent")
    st.markdown("""
    Aqui você pode gerar uma previsão para os próximos dias com base nos modelos treinados.
    Utilize o seletor abaixo para escolher entre dois modelos de Machine Learning:
    - **XGBoost**: rápido, baseado em árvores de decisão.
    - **LSTM**: mais robusto, ideal para capturar padrões temporais.
    """)

    modelo_escolhido = st.selectbox("Selecione o modelo para previsão:", ["XGBoost", "LSTM"], index=0)

    if modelo_escolhido == "XGBoost":
        rmse_modelo = 1.74
    else:
        rmse_modelo = 2.12

    st.markdown(f"**📉 RMSE do modelo selecionado:** {rmse_modelo:.2f}")

    dias_historico_prev = st.slider("Selecione o período a ser exibido no gráfico histórico (dias)", min_value=30, max_value=365, value=365, step=30)

    if st.button("Gerar Previsão"):
        with st.spinner("Carregando modelo e gerando previsão..."):
            if modelo_escolhido == "XGBoost":
                modelo = joblib.load("modelo_xgboost.pkl")
                datas, previsoes = prever_xgboost(modelo, dados)
            else:
                modelo = load_model("modelo_lstm.keras")
                datas, previsoes = prever_lstm(modelo, dados)

            st.success("Previsão gerada com sucesso!")

            fig, ax = plt.subplots(figsize=(14, 4))
            dados_recorte = dados.iloc[-dias_historico_prev:]
            ax.plot(dados_recorte.index, dados_recorte['Preco'], label='Histórico')
            # Destacar subida/descida com cor
            for i in range(1, len(previsoes)):
                cor = 'green' if previsoes[i] > previsoes[i-1] else 'red'
                ax.plot(datas[i-1:i+1], previsoes[i-1:i+1], color=cor, linewidth=2, alpha=0.8, label='_nolegend_')
            ax.plot(datas, previsoes, label='Previsão (15 dias)', linestyle='--', color='black', linewidth=1)
            ax.set_xlabel("Data")
            ax.set_ylabel("Preço (USD)")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            # Interpretação automática
            if previsoes[-1] > previsoes[0]:
                st.success("📈 Tendência de alta observada nas previsões.")
            else:
                st.warning("📉 Tendência de queda observada nas previsões.")
            # Tabela
            tabela_previsao = pd.DataFrame({
                'Data': datas.strftime('%d/%m/%Y'),
                'Preço Previsto (USD)': previsoes
            })
            tabela_formatada = tabela_previsao.style.format({'Preço Previsto (USD)': '${:,.2f}'})
            st.markdown("### 📋 Tabela com Previsão")
            st.dataframe(tabela_formatada, use_container_width=False, hide_index=True, height=565)

            # Botão de download
            csv = gerar_csv_previsao(datas, previsoes)
            st.download_button("📥 Baixar previsão como CSV", data=csv, file_name='previsao_petroleo.csv', mime='text/csv')