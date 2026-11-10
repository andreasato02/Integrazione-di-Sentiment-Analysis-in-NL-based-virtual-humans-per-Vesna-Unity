"""
Testa il modello vocale del prototipo su un sottoinsieme di Emozionalmente.

Uso:
    python valuta_emozionalmente.py CARTELLA_DATASET [--n 20] [--csv percorso.csv] [--seed 0]

CARTELLA_DATASET = cartella in cui hai estratto emozionalmente.zip.
Lo script cerca da solo un CSV con le colonne "file" ed "emotion"
(preferisce quello del test set con le emozioni "desired", cioe' quelle
volute dall'attore). Se ne trova piu' di uno o nessuno, te lo dice e puoi
indicarlo tu con --csv.

Usa audio_clf di analizza_audio.py (stesso modello vocale del prototipo),
quindi va lanciato nella stessa cartella e con lo stesso venv.
Misura SOLO la voce: niente Whisper e niente modello di testo.
"""
import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict

# Nomi unificati usati nei risultati
NORMALIZZA = {
    "anger": "anger", "angry": "anger", "rabbia": "anger",
    "disgust": "disgust", "disgusto": "disgust",
    "fear": "fear", "fearful": "fear", "paura": "fear",
    "joy": "joy", "happy": "joy", "happiness": "joy", "gioia": "joy",
    "sadness": "sadness", "sad": "sadness", "tristezza": "sadness",
    "surprise": "surprise", "surprised": "surprise", "sorpresa": "surprise",
    "neutral": "neutral", "neutrality": "neutral", "neutro": "neutral",
}
CLASSI = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Versione grossolana: utile se per il VH basta positivo / negativo / neutro
GROSSA = {"joy": "positivo", "anger": "negativo", "disgust": "negativo",
          "fear": "negativo", "sadness": "negativo", "neutral": "neutro"}
# "surprise" resta fuori dalla versione grossolana: puo' essere positiva o negativa


def norm(label):
    return NORMALIZZA.get(str(label).strip().lower())


def col(intest, tipo):
    """Trova il nome della colonna: 'file'/'file_name' oppure 'emotion'/'emotion_expressed'."""
    nomi = {"file": ["file_name", "file"],
            "emotion": ["emotion_expressed", "emotion"]}[tipo]
    for n in nomi:
        if n in intest:
            return n
    return None


def trova_csv(cartella):
    candidati = []
    for radice, _, files in os.walk(cartella):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            p = os.path.join(radice, f)
            try:
                with open(p, newline="", encoding="utf-8-sig") as fh:
                    intest = next(csv.reader(fh), [])
            except Exception:
                continue
            if col(intest, "file") and col(intest, "emotion"):
                candidati.append(p)
    return candidati


def scegli_csv(candidati):
    if not candidati:
        return None
    # preferisci il test set (speaker mai visti in addestramento)
    def punteggio(p):
        s = p.lower()
        return ("test" in s) * 2 + ("desired" in s) - ("samples" in s) * 0.5
    candidati = sorted(candidati, key=punteggio, reverse=True)
    return candidati[0]


def trova_audio(cartella, rel):
    """Il valore in 'file' puo' essere relativo alla cartella o a una sottocartella."""
    for base in (cartella, os.path.join(cartella, "audio")):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    nome = os.path.basename(rel)
    for radice, _, files in os.walk(cartella):
        if nome in files:
            return os.path.join(radice, nome)
    return None


def predici(audio_clf, percorso):
    out = audio_clf(percorso, top_k=None)
    if out and isinstance(out[0], list):  # alcune versioni annidano
        out = out[0]
    punteggi = defaultdict(float)
    for r in out:
        n = norm(r["label"])
        if n:
            punteggi[n] += float(r["score"])
    if not punteggi:
        return None, {}
    return max(punteggi, key=punteggi.get), dict(punteggi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cartella")
    ap.add_argument("--n", type=int, default=20, help="clip per emozione (default 20)")
    ap.add_argument("--csv", default=None, help="CSV con le colonne file_name ed emotion_expressed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    percorso_csv = a.csv
    if not percorso_csv:
        cand = trova_csv(a.cartella)
        percorso_csv = scegli_csv(cand)
        print("CSV candidati trovati:")
        for c in cand:
            print("  -", c)
        if not percorso_csv:
            raise SystemExit("Nessun CSV con colonne file_name ed emotion_expressed: usa --csv.")
    print("Uso il CSV:", percorso_csv)

    righe = defaultdict(list)
    with open(percorso_csv, newline="", encoding="utf-8-sig") as fh:
        lettore = csv.DictReader(fh)
        cf, ce = col(lettore.fieldnames, "file"), col(lettore.fieldnames, "emotion")
        for r in lettore:
            e = norm(r[ce])
            if e:  # scarta etichette come no_agreement
                righe[e].append(r[cf])
    print("Clip per emozione nel CSV:", {k: len(v) for k, v in sorted(righe.items())})

    random.seed(a.seed)
    campione = []
    for e in CLASSI:
        lista = righe.get(e, [])
        random.shuffle(lista)
        campione += [(f, e) for f in lista[: a.n]]
    print(f"Campione: {len(campione)} clip ({a.n} per emozione al massimo, seed {a.seed})")

    print("Carico il modello (come nel prototipo)...")
    from analizza_audio import audio_clf  # noqa: E402

    risultati, tempi, mancanti = [], [], 0
    for i, (rel, vero) in enumerate(campione, 1):
        p = trova_audio(a.cartella, rel)
        if not p:
            mancanti += 1
            continue
        t0 = time.time()
        pred, punti = predici(audio_clf, p)
        tempi.append(time.time() - t0)
        if pred is None:
            continue
        risultati.append((rel, vero, pred, punti.get(pred, 0.0)))
        if i % 20 == 0:
            print(f"  {i}/{len(campione)}")
    if mancanti:
        print(f"ATTENZIONE: {mancanti} file non trovati (controlla il percorso del dataset)")
    if not risultati:
        raise SystemExit("Nessun risultato.")

    # --- metriche ---
    ok = sum(v == p for _, v, p, _ in risultati)
    print(f"\nClip valutate: {len(risultati)}")
    print(f"Accuratezza: {ok / len(risultati):.1%}")

    conf = defaultdict(Counter)
    for _, v, p, _ in risultati:
        conf[v][p] += 1
    richiami = []
    print("\nRichiamo per emozione (quante clip vere sono riconosciute):")
    for e in CLASSI:
        tot = sum(conf[e].values())
        if tot:
            r = conf[e][e] / tot
            richiami.append(r)
            print(f"  {e:9s} {r:6.1%}  ({conf[e][e]}/{tot})")
    print(f"UAR (media dei richiami): {sum(richiami) / len(richiami):.1%}")
    print("Confronto: umani 66% UAR, wav2vec2 fine-tuned dal paper 82,45% UAR, caso 14,3%")

    print("\nMatrice di confusione (righe = vero, colonne = predetto):")
    print("          " + " ".join(f"{c[:5]:>6s}" for c in CLASSI))
    for v in CLASSI:
        if sum(conf[v].values()):
            print(f"{v:9s} " + " ".join(f"{conf[v][p]:6d}" for p in CLASSI))

    # --- versione grossolana ---
    gr = [(GROSSA[v], GROSSA[p]) for _, v, p, _ in risultati
          if v in GROSSA and p in GROSSA]
    if gr:
        okg = sum(a_ == b_ for a_, b_ in gr)
        print(f"\nVersione grossolana (positivo/negativo/neutro, esclusa 'surprise'): "
              f"{okg / len(gr):.1%} su {len(gr)} clip")

    print(f"\nTempo medio per clip: {sum(tempi) / len(tempi):.2f} s")

    with open("risultati_emozionalmente.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "vero", "predetto", "score"])
        w.writerows(risultati)
    print("Salvato risultati_emozionalmente.csv")


if __name__ == "__main__":
    main()
