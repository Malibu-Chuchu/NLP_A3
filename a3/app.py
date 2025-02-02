from flask import Flask, request, jsonify, render_template
import torch
from torchtext.data.utils import get_tokenizer
from pythainlp.tokenize import word_tokenize
import pickle
from models.seq2seq import select_model

# Set up device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocab_transform from pickle file
with open('/Users/maliboochuchu/Desktop/AIT/nlp/a3/code/vocab_transform.pkl', 'rb') as f:
    vocab_transform = pickle.load(f)

# Tokenizer for Thai and English
token_transform = {
    "en": get_tokenizer('spacy', language='en_core_web_sm'),
    "th": lambda text: word_tokenize(text, engine="newmm")
}

# Load the best model (ensure to select the right attention type)
best_attention = 'additive'
best_model = select_model(best_attention)
save_path = f'/Users/maliboochuchu/Desktop/AIT/nlp/a3/models/{best_attention}_attention.pt'
best_model.load_state_dict(torch.load(save_path))
best_model.eval()  # Set model to evaluation mode

def translate_sentence(model, sentence):
    tokenized_sentence = token_transform['en'](sentence)
    src_tokens = vocab_transform['en'].lookup_indices(tokenized_sentence)
    src_tensor = torch.tensor([src_tokens]).to(device)

    bos_idx = vocab_transform['th']["<bos>"]
    eos_idx = vocab_transform['th']["<eos>"]

    generated_tokens = [bos_idx]
    with torch.no_grad():
        for _ in range(50):
            trg_tensor = torch.tensor([generated_tokens]).to(device)
            output, _ = model(src_tensor, trg_tensor)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == eos_idx:
                break
            generated_tokens.append(next_token)

    mapping = vocab_transform['th'].get_itos()
    translated_sentence = " ".join([mapping[token] for token in generated_tokens[1:]])

    return translated_sentence

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():
    input_data = request.get_json()
    sentence = input_data.get('sentence')

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        translated_text = translate_sentence(best_model, sentence)
        return jsonify({"translation": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
