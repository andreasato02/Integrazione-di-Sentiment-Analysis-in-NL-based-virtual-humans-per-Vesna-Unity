# Integrazione di Sentiment Analysis in NL-based virtual humans per Vesna-Unity

Prototipo per rilevare l'emozione con cui un utente parla a un virtual human (VH) in un ambiente di realtà virtuale, così che il VH possa reagire con espressioni facciali e risposte coerenti. Lavoro di stage, in corso.

## Cosa fa

Dato un testo o un file audio, restituisce l'emozione dell'utente:

1. **Audio → testo:** Whisper (`openai/whisper-small`) trascrive la frase.
2. **Emozione dal testo:** `MilaNLProc/feel-it-italian-emotion` (anger, joy, sadness, fear).
3. **Emozione dalla voce:** `hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0` (anche disgust, surprise, neutral).
4. **Correzione della voce** (`calibrazione_voce.py`): il modello vocale tende a rispondere "joy" troppo spesso. Una regressione logistica addestrata sulle sue probabilità (train di Emozionalmente) le riequilibra. Attiva di default, disattivabile (vedi sotto).
5. **Fusione:** media pesata dei punteggi (peso della voce 0,6 di default, non tarato).
6. **Vocabolario del VH:** il campo `emozione_vh` traduce l'etichetta nelle stringhe accettate dal corpo Unity (`happy`, `sad`, `angry`, `fear`, `surprised`, `disgust`, `neutral`).

## Requisiti

- Python 3 e `ffmpeg` (su Mac: `brew install ffmpeg`)
- Circa 3-4 GB di spazio per i modelli, scaricati al primo avvio
- `numpy` (installato insieme a `torch`)

## Installazione

```bash
cd stage
python3 -m venv venv
source venv/bin/activate
pip install torch "transformers<5" sentencepiece flask
```

`transformers` va tenuto sotto la versione 5: con la 5 il tokenizer del modello FEEL-it dà errore.

## Uso

Ogni volta in un nuovo terminale, riattiva prima l'ambiente: `source venv/bin/activate`.

**Analizzare un file audio dal terminale**

```bash
python analizza_audio.py percorso/del/file.m4a
```

**Avviare il server** (porta 5001)

```bash
python app.py
```

Aspetta la riga `Running on http://127.0.0.1:5001`, poi da un altro terminale:

```bash
# Solo testo
curl -X POST http://127.0.0.1:5001/mood \
  -H "Content-Type: application/json" \
  -d '{"text": "Che bella notizia, sono felicissimo!"}'

# Audio
curl -X POST http://127.0.0.1:5001/mood/audio \
  -F "audio=@percorso/del/file.m4a"
```

Esempio di risposta (testo):

```json
{"emozione_vh": "happy", "mood": "joy", "score": 0.9994,
 "tutte": {"anger": 0.0001, "fear": 0.0002, "joy": 0.9994, "sadness": 0.0002}}
```

La risposta audio contiene anche `testo` (trascrizione), `da_testo`, `da_voce`, `da_voce_grezza` (voce senza correzione) e `calibrata`.

Campi facoltativi di `/mood/audio`:

| Campo | Valori | Default | Significato |
|---|---|---|---|
| `peso_audio` | 0-1 | 0.6 | peso della voce nella fusione |
| `fusione` | `semplice`, `equa` | `semplice` | modo di combinare testo e voce |
| `calibra` | `1`, `0` | `1` | applica la correzione alla voce (`0` = modello originale) |

```bash
# Stesso audio senza la correzione
curl -X POST http://127.0.0.1:5001/mood/audio \
  -F "audio=@percorso/del/file.m4a" -F "calibra=0"
```

## Valutazione

`valuta.py` esegue il prototipo su un insieme di registrazioni e misura accuratezza e tempi.

1. Metti le registrazioni in una cartella `audio/` e scrivi `casi.csv`, una riga per file, con l'emozione che volevi esprimere:

   ```
   file,atteso
   audio/01.m4a,anger
   audio/02.m4a,sadness
   ```

2. Lancia:

   ```bash
   python valuta.py casi.csv          # peso della voce 0.6
   python valuta.py casi.csv 0.7      # oppure un altro peso
   ```

Salva `risultati.csv` e stampa accuratezza di testo, voce e fusione, l'effetto del peso, i casi di discordanza e i tempi medi per fase.

## Valutazione della voce su Emozionalmente

Per misurare il modello vocale su più dati si usa [Emozionalmente](https://doi.org/10.5281/zenodo.12616095), un corpus italiano di parlato emotivo recitato (6.902 clip, 431 parlanti, 7 emozioni, licenza CC BY 4.0).

1. Scarica `emozionalmente.zip` da Zenodo (circa 560 MB) ed estrailo, ad esempio in `~/Downloads/emozionalmente` (contiene `audio/` e `metadata/`).
2. **Valutazione del modello originale** su un campione del test set (parlanti mai visti in addestramento):

   ```bash
   python valuta_emozionalmente.py ~/Downloads/emozionalmente --n 150
   ```

   `--n` è il numero di clip per emozione. Stampa accuratezza, richiamo per emozione, matrice di confusione e tempi.
3. **Correzione e confronto** (circa 12 minuti la prima volta; i calcoli intermedi sono salvati in `cache_*.npz`):

   ```bash
   python calibra_emozionalmente.py ~/Downloads/emozionalmente
   ```

   Addestra la correzione sul train, sceglie l'iperparametro sul dev, confronta sul test e salva `calibrazione.npz`.

Risultati sul test set (1.334 clip, solo voce):

| | Modello originale | Con correzione |
|---|---|---|
| UAR (media dei richiami) | 46,8% | 56,2% |
| Accuratezza | 47,1% | 56,4% |
| Positivo / negativo / neutro | 68,1% | 80,5% |
| Clip classificate "joy" | 532 | 211 |

Per confronto: caso 14,3%, ascoltatori umani 66%, wav2vec2 fine-tuned sul dataset (paper) 82,45% UAR. Richiamo per emozione dopo la correzione: anger 68%, disgust 62%, fear 61%, joy 59%, neutral 67%, sadness 50%, surprise 27%.

Dopo la correzione la confidenza è affidabile: con confidenza della voce almeno 0,5 la risposta è giusta nel 70% dei casi (su circa il 58% delle clip), con almeno 0,8 nel 91% (su circa il 10%). Il modello originale era molto sovra-confidente (94% di confidenza media per il 55% di correttezza).

## Prime registrazioni personali (indicative)

Misurate sul modello originale, prima della correzione. 8 registrazioni di una sola persona, quindi sono solo un'indicazione:

- Voce: 5/8 corrette. Testo: 2/8. Fusione (peso 0,6): 4/8.
- La voce distingue frasi uguali dette con toni diversi; il testo no.
- La rabbia è l'emozione più difficile da riconoscere dalla voce.
- Il modello di testo non ha una classe "neutro" e tende a rispondere `sadness` a frasi neutre.
- Tempo medio di elaborazione: circa 1,3 s per frase (Whisper è la fase più lenta).

## File

| File | Descrizione |
|---|---|
| `analizza_audio.py` | Modelli, fusione, mappatura verso il vocabolario del VH |
| `app.py` | Server Flask con gli endpoint `/mood` e `/mood/audio` |
| `valuta.py` | Valutazione su `casi.csv`, produce `risultati.csv` |
| `valuta_emozionalmente.py` | Valutazione del modello vocale su un campione di Emozionalmente |
| `calibra_emozionalmente.py` | Addestra la correzione della voce e la confronta con il modello originale |
| `calibrazione_voce.py` | Applica la correzione (legge `calibrazione.npz`) |
| `calibrazione.npz` | Parametri della correzione (2 KB). Senza questo file la voce non viene corretta |
| `test_audio.py` | Prova del solo modello vocale |
| `prova_tts.py` | Esperimento facoltativo di sintesi vocale (richiede `openai` e una chiave API a pagamento) |

## Limiti noti

- Le registrazioni personali sono poche e di una sola voce.
- Emozionalmente è parlato recitato: non si sa quanto le cifre valgano per voce spontanea in VR, con altri microfoni.
- Surprise resta la classe peggiore (27% di richiamo dopo la correzione, spesso scambiata per joy).
- Il modello di testo (FEEL-it) è il punto debole: non ha la classe "neutro" e tende a rispondere `sadness` anche con alta confidenza. La correzione riguarda solo la voce.
- Il peso testo/voce non è stato tarato, e la correzione cambia la scala delle probabilità vocali: andrebbe rivalutato.
- Il prototipo non è ancora collegato a Unity né ad agenti Jason/JaCaMo.

## Riferimenti

- F. Catania, J. W. Wilke, F. Garzotto, *Emozionalmente: A Crowdsourced Corpus of Simulated Emotional Speech in Italian*, IEEE Transactions on Audio, Speech and Language Processing, vol. 33, 2025. DOI 10.1109/TASLPRO.2025.3540662. Dataset: DOI 10.5281/zenodo.12616095 (CC BY 4.0). La correzione della voce è addestrata su questo dataset.
