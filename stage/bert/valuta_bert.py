"""
Valuta il modello boltuix/bert-emotion sul test set di MELD (solo testo, inglese).

Prima scarica il test set (una volta sola, ~300 KB):
    curl -L -o test_sent_emo.csv https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv
Poi, nel venv:
    python valuta_bert.py                 # tutto il test set (~2600 frasi, qualche minuto)
    python valuta_bert.py --n 300         # prova veloce su 300 frasi

Misura:
  - emozioni (7 classi di MELD): accuratezza, UAR, F1 pesata, richiamo per classe, matrice di confusione
  - sentiment (positivo/negativo/neutro), ricavato dalle stesse probabilita'
  - tempo per frase (il nostro vincolo e' il "quasi real time")
Salva le predizioni in risultati_bert_meld.csv.
"""
import argparse
import csv
import statistics
import time
from collections import Counter, defaultdict

from transformers import pipeline

MODELLO = "boltuix/bert-emotion"

# Le 13 etichette del modello, nell'ordine della scheda (serve se il modello
# restituisce LABEL_0, LABEL_1, ... invece dei nomi)
ETICHETTE_MODELLO = ["sadness", "anger", "love", "surprise", "fear", "happiness", "neutral",
                     "disgust", "shame", "guilt", "confusion", "desire", "sarcasm"]

# 13 etichette del modello -> 7 emozioni di MELD.
# Scelta nostra, da dichiarare nella relazione: si puo' cambiare qui.
A_MELD = {
    "happiness": "joy", "love": "joy", "desire": "joy",
    "sadness": "sadness", "shame": "sadness", "guilt": "sadness",
    "anger": "anger", "fear": "fear", "disgust": "disgust",
    "surprise": "surprise", "confusion": "surprise",
    "neutral": "neutral", "sarcasm": "neutral",
}
EMOZIONI = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# 7 emozioni -> sentiment. "surprise" puo' essere positiva o negativa:
# non la contiamo per il sentiment (vince la classe piu' probabile tra le altre).
A_SENTIMENT = {"joy": "positive", "anger": "negative", "disgust": "negative",
               "fear": "negative", "sadness": "negative", "neutral": "neutral"}
SENTIMENT = ["negative", "neutral", "positive"]


def nome_etichetta(label):
    l = label.lower()
    if l.startswith("label_"):
        return ETICHETTE_MODELLO[int(l.split("_")[1])]
    return l


def predici(clf, frase):
    out = clf(frase)
    if out and isinstance(out[0], list):
        out = out[0]
    emo = defaultdict(float)
    for r in out:
        emo[A_MELD.get(nome_etichetta(r["label"]), "neutral")] += r["score"]
    sent = defaultdict(float)
    for e, p in emo.items():
        if e in A_SENTIMENT:
            sent[A_SENTIMENT[e]] += p
    return max(emo, key=emo.get), max(sent, key=sent.get), out


def report(titolo, classi, veri, pred):
    n = len(veri)
    print(f"\n=== {titolo} ({n} frasi) ===")
    acc = sum(v == p for v, p in zip(veri, pred)) / n
    richiami, f1, supporto = {}, {}, Counter(veri)
    for c in classi:
        tp = sum(v == c and p == c for v, p in zip(veri, pred))
        np_ = sum(p == c for p in pred)
        nv = supporto[c]
        r = tp / nv if nv else 0.0
        pr = tp / np_ if np_ else 0.0
        richiami[c], f1[c] = r, (2 * pr * r / (pr + r) if pr + r else 0.0)
    uar = statistics.mean(richiami[c] for c in classi if supporto[c])
    f1w = sum(f1[c] * supporto[c] for c in classi) / n
    magg = supporto.most_common(1)[0]
    print(f"Accuratezza: {acc:.1%}   UAR: {uar:.1%}   F1 pesata: {f1w:.1%}")
    print(f"(classe piu' frequente '{magg[0]}' = {magg[1] / n:.1%}: e' l'accuratezza di chi risponde sempre quella)")
    print("Richiamo per classe:")
    for c in classi:
        print(f"  {c:9s} {richiami[c]:6.1%}  (su {supporto[c]})")
    print("Volte che il modello dice ciascuna classe:", dict(Counter(pred)))
    print("Matrice di confusione (righe = vero, colonne = predetto):")
    print("          " + " ".join(f"{c[:5]:>6s}" for c in classi))
    for v in classi:
        print(f"{v:9s} " + " ".join(f"{sum(a == v and b == p for a, b in zip(veri, pred)):6d}" for p in classi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test_sent_emo.csv")
    ap.add_argument("--n", type=int, default=0, help="usa solo le prime N frasi (0 = tutte)")
    a = ap.parse_args()

    with open(a.csv, newline="", encoding="utf-8", errors="replace") as fh:
        righe = [r for r in csv.DictReader(fh) if r["Emotion"] in EMOZIONI]
    if a.n:
        righe = righe[: a.n]
    print(f"Frasi di MELD: {len(righe)}")

    print("Carico il modello...")
    t0 = time.perf_counter()
    clf = pipeline("text-classification", model=MODELLO, top_k=None)
    print(f"Modello caricato in {time.perf_counter() - t0:.1f} s")

    _, _, grezzo = predici(clf, "I love you")  # riscaldamento + controllo etichette
    print("Esempio di uscita grezza:", [(r["label"], round(r["score"], 3)) for r in grezzo[:3]])

    veri_e, pred_e, veri_s, pred_s, tempi, salvate = [], [], [], [], [], []
    for i, r in enumerate(righe, 1):
        frase = r["Utterance"].replace("\x92", "'").strip()
        t = time.perf_counter()
        e, s, _ = predici(clf, frase)
        tempi.append((time.perf_counter() - t) * 1000)
        veri_e.append(r["Emotion"]); pred_e.append(e)
        veri_s.append(r["Sentiment"]); pred_s.append(s)
        salvate.append([frase, r["Emotion"], e, r["Sentiment"], s])
        if i % 300 == 0:
            print(f"  {i}/{len(righe)}")

    report("EMOZIONI (7 classi)", EMOZIONI, veri_e, pred_e)
    report("SENTIMENT (3 classi)", SENTIMENT, veri_s, pred_s)

    tempi.sort()
    print("\n=== VELOCITA' ===")
    print(f"Tempo per frase: media {statistics.mean(tempi):.1f} ms, mediana {statistics.median(tempi):.1f} ms, "
          f"95% delle frasi sotto {tempi[int(len(tempi) * 0.95) - 1]:.1f} ms")

    with open("risultati_bert_meld.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frase", "emozione_vera", "emozione_predetta", "sentiment_vero", "sentiment_predetto"])
        w.writerows(salvate)
    print("Predizioni salvate in risultati_bert_meld.csv")


if __name__ == "__main__":
    main()
