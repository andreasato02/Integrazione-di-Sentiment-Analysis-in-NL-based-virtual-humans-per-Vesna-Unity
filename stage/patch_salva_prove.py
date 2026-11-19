"""
Fa salvare ad analizza_audio.py ogni registrazione analizzata, con tutti i punteggi,
nella cartella prove/ (accanto al file). Aggiunge anche i punteggi della voce grezza.
Uso:  cd ~/stage && python patch_salva_prove.py
Crea analizza_audio.py.bak_prove. Si puo' rilanciare senza danni.
"""
import os, shutil

p = "analizza_audio.py"
s = open(p, encoding="utf-8").read()
if "salva_prova" in s:
    raise SystemExit("analizza_audio.py e' gia' modificato")
shutil.copy(p, p + ".bak_prove")

def sostituisci(s, vecchio, nuovo):
    assert s.count(vecchio) == 1, "non trovo: " + vecchio[:60]
    return s.replace(vecchio, nuovo)

s = sostituisci(s, "import sys\n", "import json\nimport os\nimport shutil\nimport sys\nimport time\n")

s = sostituisci(s, "def analizza(", '''CARTELLA_PROVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prove")


def salva_prova(percorso, r):
    """Copia l'audio e salva tutti i punteggi in prove/, per tarare la fusione dopo."""
    try:
        os.makedirs(CARTELLA_PROVE, exist_ok=True)
        nome = time.strftime("%Y%m%d-%H%M%S")
        ext = os.path.splitext(percorso)[1] or ".wav"
        shutil.copy(percorso, os.path.join(CARTELLA_PROVE, nome + ext))
        with open(os.path.join(CARTELLA_PROVE, nome + ".json"), "w", encoding="utf-8") as f:
            json.dump({"audio": nome + ext, **r}, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print("Non riesco a salvare la prova:", e, flush=True)


def analizza(''')

s = sostituisci(s, '    return {\n        "testo": testo,',
                '    r = {\n        "testo": testo,\n'
                '        "punteggi_voce_grezza": {k: round(v, 3) for k, v in emo_audio_grezza.items()},')

s = sostituisci(s, '        "punteggi_finali": {k: round(v, 3) for k, v in combinato.items()},\n    }\n',
                '        "punteggi_finali": {k: round(v, 3) for k, v in combinato.items()},\n    }\n'
                '    salva_prova(percorso, r)\n    return r\n')

open(p, "w", encoding="utf-8").write(s)
print("Fatto: ogni analisi ora viene salvata in", os.path.abspath("prove"))
