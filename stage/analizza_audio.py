import sys
from transformers import pipeline

from calibrazione_voce import calibra as calibra_voce, DISPONIBILE as CALIBRAZIONE_DISPONIBILE

AUDIO_MODEL = "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0"
TEXT_MODEL = "MilaNLProc/feel-it-italian-emotion"

print("Carico i modelli (la prima volta scarica Whisper, circa 1 GB)...", flush=True)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30)
text_clf = pipeline("text-classification", model=TEXT_MODEL, top_k=None)
audio_clf = pipeline("audio-classification", model=AUDIO_MODEL, top_k=None)

# Porta le etichette dei due modelli su un vocabolario comune
NORMALIZZA = {
    "anger": "anger", "angry": "anger",
    "joy": "joy", "happy": "joy",
    "sadness": "sadness", "sad": "sadness",
    "fear": "fear",
}

# Stringhe accettate dallo switch in AbstractAvatarSocial.cs (branch CateMiki)
VH_EMOZIONE = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprised",
    "disgust": "disgust",
    "neutral": "neutral",
}

def appiattisci(risultati):
    # La pipeline a volte restituisce una lista annidata
    return risultati[0] if risultati and isinstance(risultati[0], list) else risultati


def normalizza(risultati):
    out = {}
    for r in appiattisci(risultati):
        etichetta = NORMALIZZA.get(r["label"], r["label"])
        out[etichetta] = out.get(etichetta, 0) + r["score"]
    return out


MODI_FUSIONE = ("semplice", "equa")


def top(d):
    return max(d, key=d.get) if d else None


def fondi(emo_testo, emo_audio, peso_audio=0.8, modo="equa"):
    """Combina i punteggi di testo e voce. Il risultato somma sempre a 1.

    "semplice": media pesata delle due distribuzioni. Il testo conosce solo
    4 emozioni, quindi disgust/neutral/surprise ricevono al massimo
    peso_audio e sono svantaggiate.

    "equa": il testo si esprime solo sulle emozioni che conosce. La sua
    distribuzione viene riscalata sulla massa che la voce assegna a quelle
    stesse emozioni; per le altre emozioni "si astiene" e vale la voce.
    """
    if not 0 <= peso_audio <= 1:
        raise ValueError("peso_audio deve essere tra 0 e 1")
    if modo not in MODI_FUSIONE:
        raise ValueError(f"modo deve essere uno di {MODI_FUSIONE}")
    if not emo_testo:  # trascrizione vuota: conta solo la voce
        return dict(emo_audio)

    if modo == "equa":
        massa_comune = sum(emo_audio.get(e, 0) for e in emo_testo)
        emo_testo = {
            **emo_audio,
            **{e: v * massa_comune for e, v in emo_testo.items()},
        }

    etichette = set(emo_testo) | set(emo_audio)
    return {
        e: peso_audio * emo_audio.get(e, 0) + (1 - peso_audio) * emo_testo.get(e, 0)
        for e in etichette
    }


def analizza(percorso, peso_audio=0.8, modo="equa", calibra=True):
    testo = asr(
        percorso, generate_kwargs={"language": "italian", "task": "transcribe"}
    )["text"].strip()

    emo_testo = normalizza(text_clf(testo)) if testo else {}
    emo_audio_grezza = normalizza(audio_clf(percorso))
    # Corregge la distorsione del modello vocale verso "joy" (se calibrazione.npz c'e')
    emo_audio = calibra_voce(emo_audio_grezza) if calibra else emo_audio_grezza

    combinato = fondi(emo_testo, emo_audio, peso_audio, modo)

    return {
        "testo": testo,
        "da_testo": top(emo_testo),
        "da_voce": top(emo_audio),
        "da_voce_grezza": top(emo_audio_grezza),
        "calibrata": bool(calibra and CALIBRAZIONE_DISPONIBILE),
        "finale": top(combinato),
        "emozione_vh": VH_EMOZIONE.get(top(combinato), "neutral"),
        "punteggi_testo": {k: round(v, 3) for k, v in emo_testo.items()},
        "punteggi_voce": {k: round(v, 3) for k, v in emo_audio.items()},
        "punteggi_finali": {k: round(v, 3) for k, v in combinato.items()},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analizza_audio.py /percorso/del/file.m4a")
        sys.exit(1)

    r = analizza(sys.argv[1])
    print("\nTrascrizione:", r["testo"])
    print("Emozione dal testo :", r["da_testo"], r["punteggi_testo"])
    print("Emozione dalla voce:", r["da_voce"], r["punteggi_voce"])
    print("Emozione finale    :", r["finale"], r["punteggi_finali"])