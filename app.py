from flask import Flask, jsonify, request, render_template
import pandas as pd 
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')


EVAL_PATH = os.path.join(os.path.dirname(__file__),"evaluation")
LANG_MAP = {
    "as": "Assamese",
    "bd": "Bodo",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

lang_summary_path = os.path.join(EVAL_PATH,"language_confidence.csv")
predictions_path = os.path.join(EVAL_PATH,"test_predictions.csv")

lang_summary = pd.read_csv(lang_summary_path)
lang_summary["language"] = lang_summary["language"].map(LANG_MAP).fillna(lang_summary["language"])
predictions = pd.read_csv(predictions_path)

@app.route('/')
def dashboard():
    lang_summary_records = lang_summary.to_dict(orient='records')
    return render_template(
        "index.html",
        lang_summary=lang_summary_records,
        plots=[
            "confidence_distribution.png",
            "confidence_per_language.png",
            "tsne_embeddings.png"
        ]
    )

@app.route('/api/lang-summary')
def get_lang_summary():
    return jsonify(lang_summary.to_dict(orient='records'))

@app.route('/api/predictions')
def get_predictions():
    return jsonify(predictions.head(50).to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True,
            host='0.0.0.0',
            port=5000)
