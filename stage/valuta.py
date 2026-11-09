import csv
import os
import sys
import time
from statistics import mean

from analizza_audio import (
    asr, text_clf, audio_clf, normalizza, NORMALIZZA, MODI_FUSIONE, fondi, top,
)

# Le uniche emozioni che il modello di testo può produrre
ETICHETTE_TESTO = {"anger", "joy", "sadness", "fear"}

# Pesi della voce provati nella scelta del peso
PESI = [i / 10 for i in range(11)]


def etichetta(x):
    x = x.strip().lower()
    return NORMALIZZA.get(x, x)


def esegui(percorso):
    t0 = time.perf_counter()
    testo = asr(
        percorso, generate_kwargs={"language": "italian", "task": "transcribe"}
    )["text"].strip()
    t1 = time.perf_counter()
    emo_testo = normalizza(text_clf(testo)) if testo else {}
    t2 = time.perf_counter()
    emo_audio = normalizza(audio_clf(percorso))
    t3 = time.perf_counter()
    return {
        "testo": testo,
        "emo_testo": emo_testo,
        "emo_audio": emo_audio,
        "t_asr": t1 - t0,
        "t_testo": t2 - t1,
        "t_voce": t3 - t2,
        "t_tot": t3 - t0,
    }


def giusti(casi, peso, modo):
    return sum(top(fondi(r["emo_testo"], r["emo_audio"], peso, modo)) == r["atteso"] for r in casi)


def miglior_peso(casi, modo):
    # A parità di risultato preferisce il peso più vicino a 0.5 (nessuna preferenza)
    return max(PESI, key=lambda w: (giusti(casi, w, modo), -abs(w - 0.5)))


def leave_one_out(risultati, modo):
    """Per ogni caso sceglie il peso sugli ALTRI casi e poi lo prova su quello escluso.

    Così l'accuratezza non è gonfiata dall'aver scelto il peso sugli stessi dati.
    """
    corretti, pesi_scelti = 0, []
    for i, escluso in enumerate(risultati):
        altri = risultati[:i] + risultati[i + 1:]
        w = miglior_peso(altri, modo)
        pesi_scelti.append(w)
        corretti += giusti([escluso], w, modo)
    return corretti / len(risultati), pesi_scelti


def main(csv_in, peso=0.6):
    cartella = os.path.dirname(os.path.abspath(csv_in))
    with open(csv_in, newline="", encoding="utf-8") as f:
        casi = list(csv.DictReader(f))

    def percorso(caso):
        p = caso["file"].strip()
        return p if os.path.isabs(p) else os.path.join(cartella, p)

    print("Riscaldamento (la prima chiamata è più lenta, non viene contata)...", flush=True)
    esegui(percorso(casi[0]))

    risultati = []
    for i, c in enumerate(casi, 1):
        r = esegui(percorso(c))
        r["file"] = c["file"]
        r["atteso"] = etichetta(c["atteso"])
        r["da_testo"] = top(r["emo_testo"])
        r["da_voce"] = top(r["emo_audio"])
        for modo in MODI_FUSIONE:
            r[f"finale_{modo}"] = top(fondi(r["emo_testo"], r["emo_audio"], peso, modo))
        risultati.append(r)
        print(
            f"[{i}/{len(casi)}] {c['file']}: atteso={r['atteso']} "
            f"testo={r['da_testo']} voce={r['da_voce']} "
            f"semplice={r['finale_semplice']} equa={r['finale_equa']} "
            f"({r['t_tot']:.2f}s)",
            flush=True,
        )

    # Salva tutto in una tabella, accanto al file dei casi
    csv_out = os.path.join(cartella, "risultati.csv")
    campi = ["file", "atteso", "testo", "da_testo", "da_voce", "finale_semplice",
             "finale_equa", "t_asr", "t_testo", "t_voce", "t_tot"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi, extrasaction="ignore")
        w.writeheader()
        for r in risultati:
            w.writerow({**r, **{k: round(r[k], 3) for k in ("t_asr", "t_testo", "t_voce", "t_tot")}})

    n = len(risultati)

    def acc(chiave, sottoinsieme=None):
        rr = [r for r in risultati if sottoinsieme is None or r["atteso"] in sottoinsieme]
        if not rr:
            return float("nan"), 0
        return sum(r[chiave] == r["atteso"] for r in rr) / len(rr), len(rr)

    print(f"\n=== ACCURATEZZA (peso voce fisso {peso}) ===")
    a, k = acc("da_testo", ETICHETTE_TESTO)
    print(f"Testo           : {a:.0%} (su {k} casi con emozioni note al modello di testo)")
    a, _ = acc("da_voce")
    print(f"Voce            : {a:.0%} (su {n})")
    for modo in MODI_FUSIONE:
        a, _ = acc(f"finale_{modo}")
        print(f"Fusione {modo:8s}: {a:.0%} (su {n})")

    print("\n=== FUSIONE AL VARIARE DEL PESO DELLA VOCE ===")
    print("(solo descrittivo: il peso migliore qui è scelto sugli stessi dati, quindi è ottimistico)")
    print("peso  " + "  ".join(f"{m:>8s}" for m in MODI_FUSIONE))
    for w in PESI:
        print(f"{w:.1f}   " + "  ".join(f"{giusti(risultati, w, m) / n:>8.0%}" for m in MODI_FUSIONE))

    print("\n=== STIMA ONESTA: PESO SCELTO CON LEAVE-ONE-OUT ===")
    for modo in MODI_FUSIONE:
        a, pesi_scelti = leave_one_out(risultati, modo)
        print(f"Fusione {modo:8s}: {a:.0%} (pesi scelti: {', '.join(f'{w:.1f}' for w in pesi_scelti)})")
    if n < 30:
        print(f"Attenzione: con {n} casi ogni file sposta l'accuratezza di {1 / n:.0%}.")

    disc = [r for r in risultati if r["da_testo"] != r["da_voce"]]
    print(f"\n=== TESTO E VOCE DISCORDANO: {len(disc)} casi su {n} ===")
    if disc:
        print("Ha ragione il testo   :", sum(r["da_testo"] == r["atteso"] for r in disc))
        print("Ha ragione la voce    :", sum(r["da_voce"] == r["atteso"] for r in disc))
        for modo in MODI_FUSIONE:
            print(f"Ha ragione f. {modo:8s}:", sum(r[f"finale_{modo}"] == r["atteso"] for r in disc))

    print("\n=== TEMPI MEDI (secondi) ===")
    for chiave, nome in [("t_asr", "Whisper"), ("t_testo", "Emozione dal testo"),
                         ("t_voce", "Emozione dalla voce"), ("t_tot", "Totale")]:
        print(f"{nome:22s}: {mean(r[chiave] for r in risultati):.2f}")

    print(f"\nRisultati completi salvati in {csv_out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python valuta.py casi.csv [peso_voce]")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.6)
