from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
import torch

app = Flask(__name__)

# Translation model mappings
MODEL_MAPPING = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "mr": "Helsinki-NLP/opus-mt-en-mr"
}

models = {}
tokenizers = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for lang, model_name in MODEL_MAPPING.items():
    tokenizers[lang] = MarianTokenizer.from_pretrained(model_name)
    models[lang] = MarianMTModel.from_pretrained(model_name).to(device)

translator = Translator()

def translate_text(text, target_lang):
    if not text.strip():
        return "Please enter some text to translate."

    # Languages supported via Google Translate
    google_langs = ["hi", "kn", "mr", "ta", "te", "ml", "bn"]

    if target_lang in google_langs:
        try:
            return translator.translate(text, src="en", dest=target_lang).text
        except Exception as e:
            return f"Google Translate error: {str(e)}"

    # If a huggingface model is available (currently for hi and mr only)
    if target_lang in models:
        try:
            tokenizer = tokenizers[target_lang]
            model = models[target_lang]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_ids = model.generate(**inputs)
            translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            return translated_text[0]
        except Exception as e:
            return f"Model translation error: {str(e)}"

    return "Unsupported language selected."

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save("static/translated_audio.mp3")
        return True
    except Exception as e:
        print("TTS Error:", e)
        return False



@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    selected_lang = "hi"

    if request.method == 'POST':
        input_text = request.form.get("input_text", "").strip()
        selected_lang = request.form.get("language", "hi")

        if input_text:
            translation = translate_text(input_text, selected_lang)

    return render_template('index.html', translation=translation, selected_lang=selected_lang)




if __name__ == '__main__':
    app.run(debug=True)
