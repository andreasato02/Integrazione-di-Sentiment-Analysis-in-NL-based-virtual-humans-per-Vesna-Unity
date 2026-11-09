import os
import tempfile
import threading

from flask import Flask, request, jsonify

# Importare questo modulo carica Whisper e i due modelli di emozioni, una volta sola
from analizza_audio import analizza, text_clf, normalizza, top, VH_EMOZIONE, MODI_FUSIONE

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # max 20 MB per richiesta

# Evita che due richieste usino i modelli nello stesso momento
lock = threading.Lock()


@app.route("/mood", methods=["POST"])
def mood_testo():
    """Solo testo: JSON {"text": "..."}"""
    dati = request.get_json(silent=True) or {}
    testo = dati.get("text", "").strip()
    if not testo:
        return jsonify({"errore": "campo 'text' mancante o vuoto"}), 400

    with lock:
        emo = normalizza(text_clf(testo))
    mood = top(emo)
    return jsonify({
        "mood": mood,
        "emozione_vh": VH_EMOZIONE.get(mood, "neutral"),
        "score": round(emo[mood], 4),
        "tutte": {k: round(v, 4) for k, v in emo.items()},
    })


@app.route("/mood/audio", methods=["POST"])
def mood_audio():
    """Audio: multipart/form-data con un campo 'audio' (wav, m4a, mp3...).

    Campi facoltativi: 'peso_audio' (0-1, default 0.6) e
    'fusione' ("semplice" o "equa", default "semplice").
    """
    if "audio" not in request.files:
        return jsonify({"errore": "campo 'audio' mancante"}), 400

    f = request.files["audio"]
    ext = os.path.splitext(f.filename or "")[1] or ".wav"
    try:
        peso = float(request.form.get("peso_audio", 0.6))
    except ValueError:
        peso = None
    if peso is None or not 0 <= peso <= 1:
        return jsonify({"errore": "'peso_audio' deve essere un numero tra 0 e 1"}), 400
    modo = request.form.get("fusione", default="semplice")
    if modo not in MODI_FUSIONE:
        return jsonify({"errore": f"'fusione' deve essere uno di {list(MODI_FUSIONE)}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        percorso = tmp.name

    try:
        with lock:
            r = analizza(percorso, peso_audio=peso, modo=modo)
    except Exception as e:
        return jsonify({"errore": str(e)}), 500
    finally:
        os.remove(percorso)

    finale = r["finale"]
    return jsonify({
        "mood": finale,
        "emozione_vh": r["emozione_vh"],
        "score": r["punteggi_finali"].get(finale),
        "testo": r["testo"],
        "da_testo": r["da_testo"],
        "da_voce": r["da_voce"],
        "fusione": modo,
        "peso_audio": peso,
        "tutte": r["punteggi_finali"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)