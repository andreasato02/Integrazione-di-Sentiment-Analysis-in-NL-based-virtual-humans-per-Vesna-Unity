"""
Corregge le uscite del modello vocale con una regressione logistica
addestrata sulle sue probabilita' (train di Emozionalmente) e la valuta
sul test set (speaker mai visti).

Uso (nella cartella con analizza_audio.py e valuta_emozionalmente.py):
    python calibra_emozionalmente.py ~/Downloads/emozionalmente

Passi:
 1. fa passare le clip di train/dev/test dal modello vocale e salva le
    probabilita' in cache_<split>.npz (la seconda volta non rifa' il calcolo);
 2. addestra la correzione sul train, sceglie l'iperparametro sul dev;
 3. confronta sul test: modello originale vs modello corretto;
 4. salva i parametri in calibrazione.npz.
Tempo stimato: circa 12 minuti la prima volta (6.900 clip a ~0,18 s).
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np

from valuta_emozionalmente import CLASSI, GROSSA, col, norm, predici, trova_audio

K = len(CLASSI)
IDX = {c: i for i, c in enumerate(CLASSI)}


# ---------------------------------------------------------------- estrazione
def leggi_split(cartella, split):
    p = os.path.join(cartella, "metadata", "split", f"{split}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Non trovo {p}")
    out = []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        cf, ce = col(r.fieldnames, "file"), col(r.fieldnames, "emotion")
        for riga in r:
            e = norm(riga[ce])
            if e:
                out.append((riga[cf], e))
    return out


def estrai(cartella, split, audio_clf):
    cache = f"cache_{split}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"[{split}] letto dalla cache: {len(d['y'])} clip")
        return d["X"], d["y"]
    righe = leggi_split(cartella, split)
    parziale = f"cache_{split}_parziale.npz"
    fatti, X, y = set(), [], []
    saltati = []
    if os.path.exists(parziale):  # riprende da dove si era fermato
        d = np.load(parziale, allow_pickle=True)
        fatti = set(d["files"].tolist())
        X, y = d["X"].tolist(), d["y"].tolist()
        saltati = d["saltati"].tolist()
        print(f"[{split}] ripreso dal salvataggio parziale: {len(fatti)} clip gia' fatte")

    def salva_parziale(files_fatti):
        np.savez(parziale, files=np.array(sorted(files_fatti)), X=np.array(X, dtype=float),
                 y=np.array(y, dtype=int), saltati=np.array(saltati))

    t0 = time.time()
    nuovi = 0
    for i, (rel, vero) in enumerate(righe, 1):
        if rel in fatti or rel in saltati:
            continue
        p = trova_audio(cartella, rel)
        if not p:
            saltati.append(rel)
            continue
        try:
            pred, punti = predici(audio_clf, p)
        except Exception as ex:  # file audio difettoso: lo salto e lo segnalo
            print(f"[{split}] SALTO {rel}: {type(ex).__name__}: {ex}")
            saltati.append(rel)
            continue
        if pred is None:
            saltati.append(rel)
            continue
        X.append([punti.get(c, 0.0) for c in CLASSI])
        y.append(IDX[vero])
        fatti.add(rel)
        nuovi += 1
        if nuovi % 300 == 0:
            salva_parziale(fatti)
            print(f"[{split}] {i}/{len(righe)}  ({time.time() - t0:.0f} s)")
    X, y = np.array(X, dtype=float), np.array(y, dtype=int)
    np.savez(cache, X=X, y=y)
    if os.path.exists(parziale):
        os.remove(parziale)
    print(f"[{split}] {len(y)} clip salvate in {cache}; saltate: {len(saltati)}")
    return X, y


# ------------------------------------------------------------ regressione
def feat(X):
    return np.log(np.clip(X, 1e-6, 1.0))


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def addestra(F, y, l2, iters=600, lr=0.05):
    """Regressione logistica multinomiale, pesi di classe bilanciati, Adam."""
    n, d = F.shape
    Y = np.eye(K)[y]
    conteggi = np.bincount(y, minlength=K).astype(float)
    pc = np.where(conteggi > 0, n / (K * np.maximum(conteggi, 1)), 0.0)
    pesi = pc[y][:, None]
    W, b = np.zeros((d, K)), np.zeros(K)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    for t in range(1, iters + 1):
        P = softmax(F @ W + b)
        G = (P - Y) * pesi / n
        gW = F.T @ G + l2 * W
        gb = G.sum(axis=0)
        for par, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            par -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def predici_cal(F, W, b):
    return (F @ W + b).argmax(axis=1)


# ---------------------------------------------------------------- metriche
def uar(y, p):
    r = [np.mean(p[y == c] == c) for c in range(K) if np.any(y == c)]
    return float(np.mean(r))


def report(titolo, y, p):
    print(f"\n=== {titolo} ===")
    print(f"Accuratezza: {np.mean(p == y):.1%}   UAR: {uar(y, p):.1%}")
    print("Richiamo per emozione:")
    for c in range(K):
        m = y == c
        if m.any():
            print(f"  {CLASSI[c]:9s} {np.mean(p[m] == c):6.1%}  ({int(np.sum(p[m] == c))}/{int(m.sum())})")
    print("Volte che il modello dice ciascuna etichetta:",
          {CLASSI[c]: int(np.sum(p == c)) for c in range(K)})
    conf = np.zeros((K, K), dtype=int)
    for a, b_ in zip(y, p):
        conf[a, b_] += 1
    print("Matrice di confusione (righe = vero, colonne = predetto):")
    print("          " + " ".join(f"{c[:5]:>6s}" for c in CLASSI))
    for i in range(K):
        print(f"{CLASSI[i]:9s} " + " ".join(f"{conf[i, j]:6d}" for j in range(K)))
    gr = [(GROSSA[CLASSI[a]], GROSSA[CLASSI[b_]]) for a, b_ in zip(y, p)
          if CLASSI[a] in GROSSA and CLASSI[b_] in GROSSA]
    if gr:
        print(f"Grossolana (positivo/negativo/neutro): {np.mean([a == b_ for a, b_ in gr]):.1%} su {len(gr)} clip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cartella")
    a = ap.parse_args()

    from analizza_audio import audio_clf  # stesso modello del prototipo

    Xtr, ytr = estrai(a.cartella, "train", audio_clf)
    Xdv, ydv = estrai(a.cartella, "dev", audio_clf)
    Xte, yte = estrai(a.cartella, "test", audio_clf)

    Ftr, Fdv, Fte = feat(Xtr), feat(Xdv), feat(Xte)
    media, dev_std = Ftr.mean(axis=0), Ftr.std(axis=0) + 1e-8
    Ftr, Fdv, Fte = (Ftr - media) / dev_std, (Fdv - media) / dev_std, (Fte - media) / dev_std

    print("\nScelta dell'iperparametro sul dev set:")
    migliore = None
    for l2 in (0.0, 0.001, 0.01, 0.1, 1.0):
        W, b = addestra(Ftr, ytr, l2)
        u = uar(ydv, predici_cal(Fdv, W, b))
        print(f"  l2={l2:<6} UAR dev = {u:.1%}")
        if migliore is None or u > migliore[0]:
            migliore = (u, l2)
    l2 = migliore[1]
    print(f"Scelgo l2={l2}")
    W, b = addestra(Ftr, ytr, l2)

    report("TEST - modello originale (argmax delle probabilita')", yte, Xte.argmax(axis=1))
    report("TEST - modello corretto", yte, predici_cal(Fte, W, b))

    np.savez("calibrazione.npz", W=W, b=b, media=media, std=dev_std,
             classi=np.array(CLASSI), l2=l2)
    print("\nParametri salvati in calibrazione.npz")
    print("Confronto: umani 66% UAR, wav2vec2 fine-tuned del paper 82,45% UAR, caso 14,3%")


if __name__ == "__main__":
    main()
