import streamlit as st
import torch
import torch.nn as nn
import os
import nltk
import numpy as np
import pandas as pd
import time
from nltk.stem.porter import PorterStemmer

# --- CONFIGURATION ---
DOCS_FOLDER = "cloudwalk_docs"
MODEL_FILE = "cloudwalk_brain.pth"

# --- NLTK SETUP ---
resources = ['punkt', 'punkt_tab']
for res in resources:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res)

stemmer = PorterStemmer()

# --- NEURAL NETWORK ARCHITECTURE ---
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

# --- NLP HELPER FUNCTIONS ---
def tokenize(sentence):
    return nltk.word_tokenize(sentence, language='english')

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# --- BRAIN MANAGEMENT (LOAD & SAVE) ---
def load_brain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(MODEL_FILE):
        return None
    
    checkpoint = torch.load(MODEL_FILE)
    
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    output_size = checkpoint["output_size"]
    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    
    # We switch to train mode momentarily if we need to update gradients, 
    # but eval mode for inference.
    model.eval() 
    
    return {
        "model": model,
        "all_words": checkpoint["all_words"],
        "tags": checkpoint["tags"],
        "params": checkpoint.get("params", {}),
        "loss_history": checkpoint.get("loss_history", []),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "device": device
    }

def update_brain(brain_data, model_state_dict):
    """Overwrites the .pth file with the new learned weights"""
    data = {
        "model_state": model_state_dict,
        "input_size": brain_data["input_size"],
        "hidden_size": brain_data["hidden_size"],
        "output_size": brain_data["output_size"],
        "all_words": brain_data["all_words"],
        "tags": brain_data["tags"],
        "loss_history": brain_data["loss_history"],
        "params": brain_data["params"]
    }
    torch.save(data, MODEL_FILE)

# --- REINFORCEMENT LEARNING STEP ---
def teach_brain(brain_data, input_tensor, target_idx, action):
    """
    Online Learning: Performs a single step of optimization based on user feedback.
    action: 'reward' (minimize loss) or 'punish' (maximize loss)
    """
    model = brain_data["model"]
    device = brain_data["device"]
    
    # Switch to training mode to enable gradient calculation
    model.train()
    
    # Optimizer specifically for this online step
    # We use a slightly higher Learning Rate (0.01) for immediate feedback impact
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare target
    target = torch.tensor([target_idx], dtype=torch.long).to(device)
    
    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    
    if action == "reward":
        # Standard Backpropagation: Reinforce this connection
        loss.backward()
        message = "Brain reinforced! Connection strengthened."
    elif action == "punish":
        # Gradient Ascent: We want to INCREASE the loss for this specific wrong prediction
        # Effectively saying "Don't predict this label for this input"
        (-loss).backward() 
        message = "Brain corrected! Connection weakened."
        
    optimizer.step()
    
    # Save the updated brain to file immediately
    update_brain(brain_data, model.state_dict())
    
    return message

# --- STREAMLIT UI ---
st.set_page_config(page_title="CloudWalk RL Chat", layout="wide", page_icon="🧠")

st.title("🧠 CloudWalk Self-Learning Chat")
st.markdown("Experimental: This chatbot uses **Online Reinforcement Learning**. Use the buttons to train it in real-time.")

# --- SIDEBAR ---
brain_data = load_brain()
confidence_threshold = 0.55

with st.sidebar:
    st.header("⚙️ Control Panel")
    if brain_data:
        st.success("Brain Online")
        confidence_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.55, 0.05)
    else:
        st.error("Model missing. Run `treinar_modelo.py`.")
        st.stop()

# --- SESSION STATE FOR RL ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = None 

# --- DISPLAY HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT ---
if prompt := st.chat_input("Teach me about CloudWalk..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. PREPARE INPUT
    model = brain_data["model"]
    all_words = brain_data["all_words"]
    tags = brain_data["tags"]
    device = brain_data["device"]

    sentence = tokenize(prompt)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X).to(device) # Keep tensor for RL later

    # 2. PREDICT
    output = model(X_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    confidence_score = prob.item()

    # 3. GENERATE RESPONSE
    with st.chat_message("assistant"):
        # Attempt to retrieve content regardless of confidence
        response_file = os.path.join(DOCS_FOLDER, f"{tag}.txt")
        file_content = "⚠️ File content missing or deleted."
        if os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                file_content = f.read()

        # Build Response Structure
        if confidence_score > confidence_threshold:
            # High Confidence Response
            status_icon = "✅"
            status_msg = f"**Confident Answer** ({confidence_score:.0%})"
            response_body = file_content
        else:
            # Low Confidence Response (Draft)
            status_icon = "⚠️"
            status_msg = f"**Low Confidence** ({confidence_score:.0%}) - Is this what you meant?"
            response_body = f"> _I'm not sure, but my neural paths point to `{tag}`. Here is the content for that topic:_\n\n{file_content}"
            
        # Final formatted message
        final_markdown = f"""
{status_icon} {status_msg}
**Predicted Topic:** `{tag}`
**Source File:** `{tag}.txt`

---

{response_body}
        """
        
        st.markdown(final_markdown)
        
        # SAVE CONTEXT FOR RL BUTTONS
        st.session_state.last_interaction = {
            "input_tensor": X_tensor,
            "predicted_idx": predicted.item(),
            "tag": tag,
            "prompt": prompt
        }

    st.session_state.messages.append({"role": "assistant", "content": final_markdown})
    # Force rerun to show buttons immediately below (Streamlit logic)
    st.rerun()

# --- REINFORCEMENT BUTTONS (APPEAR AFTER RESPONSE) ---
if st.session_state.last_interaction:
    last = st.session_state.last_interaction
    
    st.markdown("---")
    st.caption(f"Was the classification **{last['tag']}** correct for: *'{last['prompt']}'*?")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # REWARD BUTTON
        if st.button("👍 Good (+1 Point)"):
            with st.spinner("Reinforcing neural pathways..."):
                msg = teach_brain(brain_data, last['input_tensor'], last['predicted_idx'], "reward")
                st.success(msg)
                time.sleep(1)
                st.session_state.last_interaction = None # Clear state
                st.rerun()

    with col2:
        # PUNISH BUTTON
        if st.button("👎 Bad (-1 Point)"):
            with st.spinner("Weakening incorrect connection..."):
                msg = teach_brain(brain_data, last['input_tensor'], last['predicted_idx'], "punish")
                st.error(msg)
                time.sleep(1)
                st.session_state.last_interaction = None # Clear state
                st.rerun()