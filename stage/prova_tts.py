import os
from openai import OpenAI
from analizza_audio import audio_clf, normalizza

client = OpenAI()  # legge la chiave dalla variabile OPENAI_API_KEY

# Stessa frase usata in casi.csv, così i risultati sono confrontabili
FRASE = "Non ci credo."

EMOZIONI = {
    "anger": "Voice affect: tense and sharp. Tone: furious, hostile. Pacing: fast and clipped. Emotion: intense anger.",
    "sadness": "Voice affect: low and heavy. Tone: sorrowful. Pacing: slow, with pauses. Emotion: deep sadness.",
    "joy": "Voice affect: bright and lively. Tone: delighted. Pacing: quick and bouncy. Emotion: joy and excitement.",
    "fear": "Voice affect: shaky and breathy. Tone: anxious. Pacing: hurried and uneven. Emotion: fear.",
}

os.makedirs("tts", exist_ok=True)

for emo, istruzioni in EMOZIONI.items():
    percorso = f"tts/{emo}.wav"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=FRASE,
        instructions=istruzioni,
        response_format="wav",
    ) as risposta:
        risposta.stream_to_file(percorso)

    punteggi = normalizza(audio_clf(percorso))
    top = max(punteggi, key=punteggi.get)
    esito = "OK" if top == emo else "diverso"
    print(f"{emo:8s} -> {top:9s} ({punteggi[top]:.2f})  {esito}", flush=True)