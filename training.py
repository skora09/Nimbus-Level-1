import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

# --- 1. CONFIGURATION AND PARAMETERS ---
DOCS_FOLDER = "cloudwalk_docs"
MODEL_FILE = "cloudwalk_brain.pth"
PLOT_FILE = "curva_aprendizado.png" # Keeping filename, content will be English

# Neural Network Hyperparameters
HIDDEN_SIZE = 16       # Neurons in hidden layer
BATCH_SIZE = 4         # Batch size
LEARNING_RATE = 0.001  # Learning rate
NUM_EPOCHS = 500       # Training cycles

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize Stemmer (English)
stemmer = PorterStemmer()

# --- 2. NLP UTILS ---
def tokenize(sentence):
    """Splits sentence into array of words/tokens."""
    return nltk.word_tokenize(sentence, language='english')

def stem(word):
    """Finds the root of the word."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """Converts a sentence into a binary vector (0s and 1s)."""
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# --- 3. DATA PREPARATION ---
def prepare_data():
    # Create folder and mock data if not exists (now in English)
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        # Creating 3 example files in English
        ex_data = {
            "culture.txt": "Our culture is customer centric. Autonomy and ownership are key. Speed is essential.",
            "tech.txt": "We use Python, Go and Rust. Our stack is Cloud Native. We use AI for fraud detection.",
            "infinitepay.txt": "InfinitePay is our payment solution. Best rates in the market. Receive instantly."
        }
        for k, v in ex_data.items():
            with open(os.path.join(DOCS_FOLDER, k), "w", encoding="utf-8") as f:
                f.write(v)

    all_words = []
    tags = [] # Tags are filenames
    xy = [] # Pairs (sentence, tag)

    print("📚 Reading files and processing sentences...")
    
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            tag = filename.replace(".txt", "") # Ex: culture
            tags.append(tag)
            
            filepath = os.path.join(DOCS_FOLDER, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split text into sentences for training
                sentences = content.split('.') 
                
                for s in sentences:
                    if len(s.strip()) > 3: # Ignore very short sentences
                        w = tokenize(s)
                        all_words.extend(w)
                        xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, all_words, tags

# --- 4. NEURAL NETWORK ARCHITECTURE ---
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

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# --- 5. TRAINING LOOP ---
def train():
    X_train, y_train, all_words, tags = prepare_data()
    
    input_size = len(X_train[0])
    output_size = len(tags)
    
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, HIDDEN_SIZE, output_size).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    # Save Model Data
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
        "loss_history": loss_history,
        "params": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE
        }
    }
    torch.save(data, MODEL_FILE)
    print(f"✅ Training finished. Model saved to {MODEL_FILE}")
    
    # Generate Plot (In English)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Error (Loss)')
    plt.title('Neural Network Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is better)')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"📊 Learning curve chart saved to {PLOT_FILE}")

if __name__ == "__main__":
    train()