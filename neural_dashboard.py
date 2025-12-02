import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Neural Brain Monitor", layout="wide", page_icon="📈")
MODEL_FILE = "cloudwalk_brain.pth"

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3 { color: #00ffbf !important; }
</style>
""", unsafe_allow_html=True)

# --- DEFINIÇÃO DA REDE (Necessária para carregar os pesos) ---
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# --- CARREGAMENTO DE DADOS ---
@st.cache_data(ttl=2) # Cache de 2 segundos para permitir "Real-time" refresh manual
def load_data():
    if not os.path.exists(MODEL_FILE):
        return None
    
    # Carrega na CPU para visualização
    checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    return checkpoint

def get_model_weights(checkpoint):
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    output_size = checkpoint["output_size"]
    
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

# --- INTERFACE ---
st.title("🧠 Neural Brain Monitor")
st.markdown("Visualize o estado interno do cérebro do Chatbot. Atualize após treinar para ver as mudanças nos pesos.")

if st.button("🔄 Atualizar Dados Agora"):
    st.cache_data.clear()
    st.rerun()

data = load_data()

if not data:
    st.error(f"Arquivo '{MODEL_FILE}' não encontrado. Treine o modelo primeiro.")
    st.stop()

model = get_model_weights(data)
all_words = data["all_words"]
tags = data["tags"]
loss_history = data.get("loss_history", [])

# --- 1. VISÃO GERAL (KPIs) ---
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Vocabulário Aprendido", f"{len(all_words)} palavras")
with col2:
    st.metric("Tópicos (Classes)", f"{len(tags)}")
with col3:
    st.metric("Neurônios na Camada Oculta", data["hidden_size"])
with col4:
    last_loss = loss_history[-1] if loss_history else 0
    st.metric("Perda Final (Treino)", f"{last_loss:.4f}")

# --- 2. CURVA DE APRENDIZADO ---
st.divider()
st.subheader("📉 Curva de Convergência (Treinamento Inicial)")
if loss_history:
    chart_data = pd.DataFrame(loss_history, columns=["Loss"])
    st.line_chart(chart_data, color="#00ffbf")
else:
    st.info("Sem histórico de perda disponível.")

# --- 3. RAIO-X NEURAL (PESOS) ---
st.divider()
st.subheader("🧬 Raio-X das Conexões Neurais")
st.caption("Aqui vemos como o cérebro conecta palavras (Input) aos Tópicos (Output).")

tab1, tab2 = st.tabs(["Camada de Entrada (Palavras -> Neurônios)", "Camada de Saída (Neurônios -> Tópicos)"])

with tab1:
    st.markdown("**O que isso significa?** Mostra quais palavras ativam quais neurônios ocultos.")
    
    # Extrair pesos da camada 1
    weights_l1 = model.l1.weight.detach().numpy() # Shape: (Hidden, Vocab)
    
    # Vamos mostrar apenas as top 20 palavras mais impactantes (pela soma absoluta dos pesos)
    # para o gráfico não ficar gigante e ilegível
    word_impact = np.sum(np.abs(weights_l1), axis=0)
    top_indices = np.argsort(word_impact)[-20:] # Top 20 palavras
    
    filtered_weights = weights_l1[:, top_indices]
    filtered_words = [all_words[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(filtered_weights, xticklabels=filtered_words, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_xlabel("Top 20 Palavras Mais Fortes")
    ax.set_ylabel("Neurônios Ocultos (16)")
    st.pyplot(fig)

with tab2:
    st.markdown("**O que isso significa?** Mostra como os neurônios ocultos votam para decidir o Tópico Final.")
    st.markdown("Se você punir (**Bad**) uma resposta no Chat, verá as cores desta matriz mudarem (esfriarem) para aquele tópico.")
    
    # Extrair pesos da camada 3 (Final)
    weights_l3 = model.l3.weight.detach().numpy() # Shape: (Classes, Hidden)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(weights_l3, yticklabels=tags, xticklabels=False, cmap="coolwarm", center=0, ax=ax2)
    ax2.set_ylabel("Tópicos (Classes)")
    ax2.set_xlabel("Ativação vinda da Camada Oculta")
    st.pyplot(fig2)

# --- 4. LISTA DE VOCABULÁRIO ---
st.divider()
with st.expander("📚 Ver Vocabulário Completo (Dicionário do Robô)"):
    st.write(all_words)

# --- 5. VISUALIZAÇÃO DE BIAS (VIESES) ---
st.divider()
st.subheader("⚖️ Bias (Vieses) dos Tópicos")
st.caption("Quais tópicos o modelo tem tendência natural a escolher (independente da entrada)?")

bias_l3 = model.l3.bias.detach().numpy()
bias_df = pd.DataFrame({"Tópico": tags, "Viés (Bias)": bias_l3})

st.bar_chart(bias_df.set_index("Tópico"), color="#ff4b4b")
