# Integrazione di Sentiment Analysis in NL-based virtual humans per Vesna-Unity

Prototipo per rilevare l'emozione con cui un utente parla a un virtual human (VH) in un ambiente di realtà virtuale, così che il VH possa reagire con espressioni facciali e risposte coerenti. Lavoro di stage, in corso.

## Cosa fa

Dato un testo o un file audio, restituisce l'emozione dell'utente:

1. **Audio → testo:** Whisper (`openai/whisper-small`) trascrive la frase.
2. **Emozione dal testo:** `MilaNLProc/feel-it-italian-emotion` (anger, joy, sadness, fear).
3. **Emozione dalla voce:** `hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0` (anche disgust, surprise, neutral).
4. **Fusione:** media pesata dei punteggi (peso della voce 0,6 di default).
5. **Vocabolario del VH:** il campo `emozione_vh` traduce l'etichetta nelle stringhe accettate dal corpo Unity (`happy`, `sad`, `angry`, `fear`, `surprised`, `disgust`, `neutral`).

## Requisiti

- Python 3 e `ffmpeg` (su Mac: `brew install ffmpeg`)
- Circa 3-4 GB di spazio per i modelli, scaricati al primo avvio

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

La risposta audio contiene anche `testo` (trascrizione), `da_testo` e `da_voce`.

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

## Primi risultati (indicativi)

8 registrazioni di una sola persona, quindi sono solo un'indicazione:

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
| `test_audio.py` | Prova del solo modello vocale |
| `prova_tts.py` | Esperimento facoltativo di sintesi vocale (richiede `openai` e una chiave API a pagamento) |

## Limiti noti

- Valutazione su pochi campioni e una sola voce.
- I modelli non sono stati testati su voce spontanea in VR.
- Il peso testo/voce non è stato tarato.
- Il prototipo non è ancora collegato a Unity né ad agenti Jason/JaCaMo.
