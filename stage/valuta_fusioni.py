"""
Confronta diverse fusioni testo+voce sulle registrazioni salvate in prove/,
senza ricaricare i modelli (usa i punteggi gia' salvati).

Uso:  cd ~/stage && python valuta_fusioni.py
La prima volta crea prove/etichette.csv: scrivi nella colonna "voluta"
l'emozione che volevi esprimere (rabbia, gioia, tristezza, paura, sorpresa,
disgusto, neutro, oppure in inglese) e rilancia.
"""
import csv, glob, json, os

from calibrazione_voce import calibra as calibra_voce

CART = "prove"
ETICH = os.path.join(CART, "etichette.csv")
NOMI = {"rabbia": "anger", "gioia": "joy", "tristezza": "sadness", "paura": "fear",
        "sorpresa": "surprise", "disgusto": "disgust", "neutro": "neutral",
        "anger": "anger", "joy": "joy", "sadness": "sadness", "fear": "fear",
        "surprise": "surprise", "disgust": "disgust", "neutral": "neutral"}


def fondi(t, a, peso, modo):  # stessa logica di analizza_audio.fondi
    if not t:
        return dict(a)
    if modo == "equa":
        m = sum(a.get(e, 0) for e in t)
        t = {**a, **{e: v * m for e, v in t.items()}}
    return {e: peso * a.get(e, 0) + (1 - peso) * t.get(e, 0) for e in set(t) | set(a)}


def top(d):
    return max(d, key=d.get) if d else None


prove = []
for f in sorted(glob.glob(os.path.join(CART, "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    if "punteggi_voce_grezza" in d:
        prove.append(d)
if not prove:
    raise SystemExit("Nessuna prova in prove/: registra qualche frase con il servizio aggiornato.")

etichette = {}
if os.path.exists(ETICH):
    for r in csv.DictReader(open(ETICH, encoding="utf-8")):
        if r.get("voluta", "").strip():
            etichette[r["audio"]] = NOMI.get(r["voluta"].strip().lower())
nuove = [p for p in prove if p["audio"] not in etichette]
if nuove:
    righe = [{"audio": p["audio"], "voluta": "", "trascrizione": p["testo"]} for p in prove]
    for r in righe:
        if r["audio"] in etichette:
            r["voluta"] = etichette[r["audio"]]
    with open(ETICH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["audio", "voluta", "trascrizione"])
        w.writeheader(); w.writerows(righe)
    print(f"{len(nuove)} registrazioni senza etichetta: compila la colonna 'voluta' in {ETICH} e rilancia.")
    if not etichette:
        raise SystemExit()

dati = [p for p in prove if etichette.get(p["audio"])]
y = [etichette[p["audio"]] for p in dati]
n = len(dati)
print(f"Registrazioni etichettate: {n}\n")


def acc(pred):
    return sum(a == b for a, b in zip(pred, y))


grezza = [p["punteggi_voce_grezza"] for p in dati]
corretta = [calibra_voce(g) for g in grezza]
testo = [p["punteggi_testo"] for p in dati]

print("Componenti da sole:")
print(f"  voce grezza    {acc([top(g) for g in grezza]):2d}/{n}")
print(f"  voce corretta  {acc([top(c) for c in corretta]):2d}/{n}")
print(f"  testo          {acc([top(t) for t in testo]):2d}/{n}\n")

print("Fusioni (giuste su", n, "):")
print("peso voce   " + "  ".join(f"{w/10:.1f}" for w in range(11)))
risultati = []
for nome_v, voce in (("grezza", grezza), ("corretta", corretta)):
    for modo in ("semplice", "equa"):
        riga = []
        for w in range(11):
            pred = [fondi(t, a, w / 10, modo) for t, a in zip(testo, voce)]
            k = acc([top(d) for d in pred])
            riga.append(k)
            risultati.append((k, w / 10, modo, nome_v, pred))
        print(f"{modo:8s} {nome_v:8s} " + "  ".join(f"{k:3d}" for k in riga))

k, w, modo, nome_v, pred = max(risultati, key=lambda r: (r[0], r[1]))
print(f"\nMigliore: modo={modo}, voce {nome_v}, peso voce={w}: {k}/{n}")
print("Attuale  : modo=semplice, voce corretta, peso voce=0.6")

print("\nSoglia di confidenza per la migliore (sopra soglia: quante, di cui giuste):")
for s in (0.3, 0.4, 0.5, 0.6):
    sopra = [(top(d) == v) for d, v in zip(pred, y) if max(d.values()) >= s]
    print(f"  soglia {s:.1f}: {len(sopra):2d}/{n} risposte, {sum(sopra)} giuste")

print("\nDettaglio (migliore):")
for p, d, v in zip(dati, pred, y):
    print(f"  {'OK ' if top(d) == v else '   '} voluta {v:8s} -> {top(d):8s} {max(d.values()):.2f}  {p['testo'][:45]}")
