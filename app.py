from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
import torch

app = Flask(__name__)

# Define available translation models
MODEL_MAPPING = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",   # English → Hindi
    "mr": "Helsinki-NLP/opus-mt-en-mr"    # English → Marathi
}

# Load models and tokenizers for Hindi and Marathi
models = {}
tokenizers = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for lang, model_name in MODEL_MAPPING.items():
    tokenizers[lang] = MarianTokenizer.from_pretrained(model_name)
    models[lang] = MarianMTModel.from_pretrained(model_name).to(device)

# Google Translate API for Kannada
translator = Translator()

def translate_text(text, target_lang):
    """Translates input text from English to the selected target language."""
    if not text.strip():
        return "Please enter some text to translate."

    if target_lang == "kn":  # Use Google Translate for Kannada
        return translator.translate(text, src="en", dest="kn").text

    if target_lang not in models:
        return "Unsupported language selected."

    try:
        tokenizer = tokenizers[target_lang]
        model = models[target_lang]

        # Tokenize input and move tensors to device
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated_ids = model.generate(**inputs)

        # Decode translation
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        return translated_text[0]

    except Exception as e:
        return f"Error in translation: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    selected_lang = "hi"  # Default to Hindi

    if request.method == 'POST':
        input_text = request.form.get("input_text", "").strip()
        selected_lang = request.form.get("language", "hi")

        if input_text:
            translation = translate_text(input_text, selected_lang)
        else:
            translation = "Please enter some text to translate."
    
    return render_template('index.html', translation=translation, selected_lang=selected_lang)

if __name__ == '__main__':
    app.run(debug=True)
