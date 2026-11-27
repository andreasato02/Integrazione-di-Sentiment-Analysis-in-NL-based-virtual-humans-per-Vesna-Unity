"""
Valuta un modello di classificazione delle emozioni (default:
j-hartmann/emotion-english-distilroberta-base, si cambia con --modello) su uno
di tre dataset (solo testo, inglese).

Modelli provati e dataset visti in addestramento (il risultato su quei dataset
non dice nulla sulla generalizzazione):
  j-hartmann/emotion-english-distilroberta-base  7 etichette (anger, disgust, fear,
      joy, neutral, sadness, surprise); addestrato anche su MELD e sull'Emotions
      dataset -> l'unico dataset davvero nuovo e' dailydialog.
      Non ha "love": su emotion quella classe non viene mai predetta.
  boltuix/bert-emotion  13 etichette; addestrato sull'Emotions dataset.

Dataset:

  meld     test set di MELD (test_sent_emo.csv): 7 emozioni + sentiment a 3 classi.
           Si scarica una volta sola (~300 KB) con:
           curl -L -o test_sent_emo.csv https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv
  emotion  "Emotions dataset for NLP" (test.txt, una riga "frase;emozione"):
           6 emozioni, niente "neutral" e niente sentiment.
  dailydialog  test set di DailyDialog (dialoghi quotidiani scritti), cartella dailydialog/:
           dialogues_test.txt (un dialogo per riga, turni separati da "__eou__") e
           dialogues_emotion_test.txt (un numero per turno). Stesse 7 emozioni di MELD
           (0 = nessuna emozione -> neutral, 4 = happiness -> joy), niente sentiment.
           Copia del test set originale scaricata da
           https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/test.zip
           (il link ufficiale http://yanran.li/files/ijcnlp_dailydialog.zip non funziona piu').

Per avviare il venv, digitare: source venv/bin/activate
Poi, nel venv:
    python valuta_bert.py --dataset meld              # tutto il test set di MELD (~2600 frasi)
    python valuta_bert.py --dataset emotion           # tutto test.txt (2000 frasi)
    python valuta_bert.py --dataset emotion --n 300   # prova veloce su 300 frasi
    python valuta_bert.py --dataset emotion --file val.txt   # un altro file nello stesso formato
    python valuta_bert.py --dataset dailydialog       # tutto il test set di DailyDialog (7740 turni)
Misura:
  - emozioni (classi del dataset): accuratezza, UAR, F1 pesata, richiamo per classe, matrice di confusione
  - sentiment, ricavato dalle stesse probabilita':
      meld        -> positivo/negativo/neutro, confrontato con il sentiment annotato
      dailydialog -> positivo/negativo/neutro, con il sentiment vero ricavato dall'emozione
      emotion     -> positivo/negativo, con la polarita' vera ricavata dall'emozione
      (dove il sentiment e' ricavato, le frasi "surprise" sono escluse perche' ambigue)
  - tempo per frase (il nostro vincolo e' il "quasi real time")
Salva le predizioni in risultati_<nome modello>_<dataset>.csv
(es. risultati_emotion-english-distilroberta-base_meld.csv).
"""
import argparse
import csv
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from transformers import pipeline

MODELLO = "j-hartmann/emotion-english-distilroberta-base"

# Le 13 etichette di boltuix/bert-emotion, nell'ordine della scheda (serve se il
# modello restituisce LABEL_0, LABEL_1, ... invece dei nomi). j-hartmann
# restituisce gia' i nomi, che le mappature qui sotto coprono tutti.
ETICHETTE_MODELLO = ["sadness", "anger", "love", "surprise", "fear", "happiness", "neutral",
                     "disgust", "shame", "guilt", "confusion", "desire", "sarcasm"]


# ---------------------------------------------------------------------------
# Lettura dei due formati. Ogni lettore restituisce una lista di
# (frase, emozione vera, sentiment vero) con le sole righe valide.
# ---------------------------------------------------------------------------

def leggi_meld(percorso, emozioni, a_sentiment):
    """Legge il CSV di MELD; il sentiment vero e' quello annotato nel dataset."""
    righe = []
    with open(percorso, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r["Emotion"] in emozioni:
                # "\x92" e' l'apostrofo "curvo" di Windows che compare in MELD
                frase = r["Utterance"].replace("\x92", "'").strip()
                righe.append((frase, r["Emotion"], r["Sentiment"]))
    return righe

def leggi_emotion(percorso, emozioni, a_sentiment):
    """
    Legge un file "frase;emozione" (senza intestazione). Divide sull'ultimo ";"
    nel caso la frase stessa contenga un punto e virgola. Il dataset non ha il
    sentiment: la polarita' vera si ricava dall'emozione ("" per surprise).
    """
    righe = []
    with open(percorso, encoding="utf-8", errors="replace") as fh:
        for riga in fh:
            riga = riga.strip()
            if ";" not in riga:
                continue
            frase, emozione = riga.rsplit(";", 1)
            emozione = emozione.strip().lower()
            if emozione in emozioni:
                righe.append((frase.strip(), emozione, a_sentiment.get(emozione, "")))
    return righe

# Codici delle emozioni di DailyDialog, gia' tradotti nei nomi di MELD
CODICI_DAILYDIALOG = {"0": "neutral", "1": "anger", "2": "disgust", "3": "fear",
                      "4": "joy", "5": "sadness", "6": "surprise"}

def ripulisci_dailydialog(frase):
    """
    Il testo di DailyDialog e' tokenizzato ("I ’ ll go , ok ?"): rimette
    apostrofi e punteggiatura attaccati come in una frase normale.
    """
    frase = frase.replace("’", "'").replace("‘", "'")
    frase = re.sub(r"\s+'\s*", "'", frase)                 # "I ' ll" -> "I'll"
    frase = re.sub(r"\s+([,.!?;:%)])", r"\1", frase)       # "ok ?" -> "ok?"
    frase = re.sub(r"([($])\s+", r"\1", frase)             # "( ciao" -> "(ciao"
    return re.sub(r"\s+", " ", frase).strip()

def leggi_dailydialog(percorso, emozioni, a_sentiment):
    """
    Legge il test set di DailyDialog: `percorso` e' dialogues_test.txt e accanto
    deve esserci dialogues_emotion_test.txt. Ogni riga e' un dialogo; i turni
    sono separati da "__eou__" e a ciascuno corrisponde un codice di emozione.
    Ogni turno diventa una frase a se'. Il sentiment vero si ricava dall'emozione
    ("" per surprise).
    """
    cartella, nome = os.path.split(percorso)
    percorso_emo = os.path.join(cartella, nome.replace("dialogues_", "dialogues_emotion_", 1))
    righe = []
    with open(percorso, encoding="utf-8", errors="replace") as fd, open(percorso_emo, encoding="utf-8") as fe:
        for n, (dialogo, codici) in enumerate(zip(fd, fe), 1):
            turni = [t for t in dialogo.split("__eou__") if t.strip()]
            codici = codici.split()
            if len(turni) != len(codici):
                print(f"  dialogo {n} saltato: {len(turni)} turni ma {len(codici)} emozioni")
                continue
            for turno, codice in zip(turni, codici):
                emozione = CODICI_DAILYDIALOG.get(codice)
                if emozione in emozioni:
                    righe.append((ripulisci_dailydialog(turno), emozione, a_sentiment.get(emozione, "")))
    return righe


# ---------------------------------------------------------------------------
# Configurazione dei due dataset
# ---------------------------------------------------------------------------

DATASET = {
    "meld": {
        "file": "test_sent_emo.csv",
        "leggi": leggi_meld,
        "emozioni": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        # 13 etichette del modello -> 7 emozioni di MELD
        "mappa": {
            "happiness": "joy", "love": "joy", "desire": "joy", "joy": "joy",
            "sadness": "sadness", "shame": "sadness", "guilt": "sadness",
            "anger": "anger", "fear": "fear", "disgust": "disgust",
            "surprise": "surprise", "confusion": "surprise",
            "neutral": "neutral", "sarcasm": "neutral",
        },
        # etichette sconosciute al dizionario: finiscono qui (None = ignorate)
        "default": "neutral",
        # emozioni -> sentiment ("surprise" e' ambigua: esclusa)
        "sentiment": {"joy": "positive", "anger": "negative", "disgust": "negative",
                      "fear": "negative", "sadness": "negative", "neutral": "neutral"},
        "classi_sentiment": ["negative", "neutral", "positive"],
        "titolo_sentiment": "SENTIMENT (3 classi)",
    },
    "emotion": {
        "file": "test.txt",
        "leggi": leggi_emotion,
        "emozioni": ["anger", "fear", "joy", "love", "sadness", "surprise"],
        # 13 etichette del modello -> 6 emozioni del dataset.
        # "neutral" e "sarcasm" non hanno un corrispettivo: le loro probabilita'
        # vengono ignorate e si sceglie fra le 6 emozioni rimaste.
        "mappa": {
            "happiness": "joy", "joy": "joy",
            "love": "love", "desire": "love",
            "sadness": "sadness", "shame": "sadness", "guilt": "sadness",
            "anger": "anger", "disgust": "anger",
            "fear": "fear",
            "surprise": "surprise", "confusion": "surprise",
        },
        "default": None,
        "sentiment": {"joy": "positive", "love": "positive",
                      "anger": "negative", "fear": "negative", "sadness": "negative"},
        "classi_sentiment": ["negative", "positive"],
        "titolo_sentiment": "POLARITA' (2 classi, senza surprise)",
    },
}
# DailyDialog ha le stesse 7 emozioni di MELD: stessa mappatura e stesso sentiment,
# cambiano solo il file e il modo di leggerlo (il sentiment vero e' ricavato).
DATASET["dailydialog"] = {
    **DATASET["meld"],
    "file": "dailydialog/dialogues_test.txt",
    "leggi": leggi_dailydialog,
    "titolo_sentiment": "SENTIMENT (3 classi, ricavato dall'emozione, senza surprise)",
}


def nome_etichetta(label):
    """
    Converte l'etichetta restituita dal modello nel nome dell'emozione, in minuscolo.
    Se il modello restituisce etichette generiche ("LABEL_0", "LABEL_1", ...),
    usa il numero come indice in ETICHETTE_MODELLO; altrimenti restituisce
    il nome cosi' com'e' (es. "Happiness" -> "happiness").
    """
    l = label.lower()
    if l.startswith("label_"):
        return ETICHETTE_MODELLO[int(l.split("_")[1])]
    return l

def predici(clf, frase, cfg):
    """
    Classifica una frase e restituisce (emozione, sentiment, uscita grezza).
    - Passa la frase al classificatore, che da' una probabilita' per ciascuna
      delle 13 etichette del modello.
    - Somma le probabilita' delle etichette che finiscono nella stessa emozione
      del dataset (es. sadness + shame + guilt -> sadness) e sceglie l'emozione
      con la probabilita' totale piu' alta. Le etichette senza corrispettivo
      finiscono in cfg["default"] (o vengono ignorate se e' None).
    - Allo stesso modo somma le probabilita' delle emozioni per sentiment
      (surprise e' esclusa perche' ambigua) e sceglie il sentiment piu' probabile.
    - L'uscita grezza del modello viene restituita per poterla ispezionare.
    """
    out = clf(frase)
    # Alcune versioni di transformers restituiscono [[...]] invece di [...]
    if out and isinstance(out[0], list):
        out = out[0]
    emo = defaultdict(float)
    for r in out:
        e = cfg["mappa"].get(nome_etichetta(r["label"]), cfg["default"])
        if e:
            emo[e] += r["score"]
    sent = defaultdict(float)
    for e, p in emo.items():
        if e in cfg["sentiment"]:
            sent[cfg["sentiment"][e]] += p
    return max(emo, key=emo.get), max(sent, key=sent.get), out

def report(titolo, classi, veri, pred):
    """
    Stampa le metriche di valutazione confrontando le etichette vere con quelle predette.
    Calcola e stampa:
      - accuratezza: frazione di frasi classificate correttamente;
      - UAR (Unweighted Average Recall): media dei richiami delle classi presenti,
        utile perche' i dataset sono sbilanciati;
      - F1 pesata: media degli F1 per classe, pesata per il numero di esempi;
      - la baseline "rispondo sempre la classe piu' frequente";
      - richiamo per classe e quante volte il modello predice ciascuna classe;
      - matrice di confusione (righe = classe vera, colonne = classe predetta).
    """
    n = len(veri)
    print(f"\n=== {titolo} ({n} frasi) ===")
    acc = sum(v == p for v, p in zip(veri, pred)) / n
    richiami, f1, supporto = {}, {}, Counter(veri)
    for c in classi:
        tp = sum(v == c and p == c for v, p in zip(veri, pred))  # veri positivi
        np_ = sum(p == c for p in pred)  # quante volte il modello ha predetto c
        nv = supporto[c]                 # quante frasi sono davvero di classe c
        r = tp / nv if nv else 0.0       # richiamo (recall)
        pr = tp / np_ if np_ else 0.0    # precisione
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
    """
    Punto di ingresso: legge gli argomenti, carica dati e modello, classifica
    ogni frase del test set misurando il tempo, stampa le metriche
    e salva le predizioni su CSV.
    """
    # --- Argomenti da riga di comando ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASET, default="meld",
                    help="quale dataset valutare (default: meld)")
    ap.add_argument("--file", help="file del dataset (default: test_sent_emo.csv per meld, test.txt per emotion, dailydialog/dialogues_test.txt per dailydialog)")
    ap.add_argument("--n", type=int, default=0, help="usa solo le prime N frasi (0 = tutte)")
    ap.add_argument("--modello", default=MODELLO, help="nome del modello su Hugging Face")
    ap.add_argument("--out", help="file in cui salvare le predizioni (default: risultati_<modello>_<dataset>.csv)")
    a = ap.parse_args()
    cfg = DATASET[a.dataset]
    percorso = a.file or cfg["file"]
    uscita = a.out or f"risultati_{a.modello.rstrip('/').split('/')[-1]}_{a.dataset}.csv"

    # --- Lettura del dataset ---
    righe = cfg["leggi"](percorso, cfg["emozioni"], cfg["sentiment"])
    if a.n:
        righe = righe[: a.n]
    print(f"Dataset {a.dataset}: {len(righe)} frasi da {percorso}")
    print("Distribuzione delle emozioni:", dict(Counter(e for _, e, _ in righe)))

    # --- Caricamento del modello (top_k=None = restituisce tutte le etichette con la loro probabilita') ---
    print("Carico il modello", a.modello, "...")
    t0 = time.perf_counter()
    clf = pipeline("text-classification", model=a.modello, top_k=None)
    print(f"Modello caricato in {time.perf_counter() - t0:.1f} s")

    _, _, grezzo = predici(clf, "I love you", cfg)  # riscaldamento + controllo etichette
    print("Esempio di uscita grezza:", [(r["label"], round(r["score"], 3)) for r in grezzo[:3]])

    # --- Classificazione di tutte le frasi, misurando il tempo di ciascuna ---
    veri_e, pred_e, veri_s, pred_s, tempi, salvate = [], [], [], [], [], []
    for i, (frase, emozione, sent_vero) in enumerate(righe, 1):
        t = time.perf_counter()
        e, s, _ = predici(clf, frase, cfg)
        tempi.append((time.perf_counter() - t) * 1000)
        veri_e.append(emozione); pred_e.append(e)
        if sent_vero:  # in "emotion" le frasi surprise non hanno polarita' vera
            veri_s.append(sent_vero); pred_s.append(s)
        salvate.append([frase, emozione, e, sent_vero, s])
        if i % 300 == 0:
            print(f"  {i}/{len(righe)}")

    # --- Metriche ---
    report(f"EMOZIONI ({len(cfg['emozioni'])} classi)", cfg["emozioni"], veri_e, pred_e)
    if veri_s:
        report(cfg["titolo_sentiment"], cfg["classi_sentiment"], veri_s, pred_s)

    # --- Velocita': media, mediana e 95° percentile del tempo per frase ---
    tempi.sort()
    print("\n=== VELOCITA' ===")
    print(f"Tempo per frase: media {statistics.mean(tempi):.1f} ms, mediana {statistics.median(tempi):.1f} ms, "
          f"95% delle frasi sotto {tempi[max(int(len(tempi) * 0.95) - 1, 0)]:.1f} ms")

    # --- Salvataggio delle predizioni frase per frase ---
    with open(uscita, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frase", "emozione_vera", "emozione_predetta", "sentiment_vero", "sentiment_predetto"])
        w.writerows(salvate)
    print("Predizioni salvate in", uscita)

if __name__ == "__main__":
    main()
