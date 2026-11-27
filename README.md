# Integrazione di Sentiment Analysis in NL-based virtual humans per Vesna-Unity

Prototipo per rilevare l'emozione con cui un utente parla a un virtual human (VH) in un ambiente di realtà virtuale, così che il VH possa reagire con espressioni facciali e risposte coerenti. Lavoro di stage, in corso.

> **Stato (novembre 2026):** il prototipo audio + testo è completo e collegato a un agente Jason in Unity. D'accordo con la relatrice, la parte audio è ora accantonata (pochi dataset adatti) e il lavoro prosegue sull'emozione dal **testo**: valutazione di due modelli inglesi (`boltuix/bert-emotion` e `j-hartmann/emotion-english-distilroberta-base`) su MELD, DailyDialog e Emotions dataset (vedi sotto).

## Cosa fa

Dato un testo o un file audio, restituisce l'emozione dell'utente:

1. **Audio → testo:** Whisper (`openai/whisper-small`) trascrive la frase.
2. **Emozione dal testo:** `MilaNLProc/feel-it-italian-emotion` (anger, joy, sadness, fear).
3. **Emozione dalla voce:** `hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0` (anche disgust, surprise, neutral).
4. **Correzione della voce** (`calibrazione_voce.py`): il modello vocale tende a rispondere "joy" troppo spesso. Una regressione logistica addestrata sulle sue probabilità (train di Emozionalmente) le riequilibra. Attiva di default, disattivabile (vedi sotto).
5. **Fusione:** modo `equa` con peso della voce 0,8 (default). Il testo vota solo sulle 4 emozioni che conosce; su disgust, surprise e neutral decide la voce. Scelta in base alle prove con la voce reale (vedi sotto).
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
| `peso_audio` | 0-1 | 0.8 | peso della voce nella fusione |
| `fusione` | `semplice`, `equa` | `equa` | modo di combinare testo e voce |
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
   python valuta.py casi.csv          # peso della voce 0.6 (default dello script)
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

## Prove dal vivo in Unity (voce reale, una persona)

Il servizio è stato collegato a un agente Jason nel ramo CateMiki di VEsNA-Unity: l'emozione rilevata arriva all'agente come evento `user_emotion(Emozione, Score)` e l'avatar la mostra sul volto. Con la fusione precedente (`semplice`, peso 0,6), su 15 frasi recitate:

| Componente | Giuste |
|---|---|
| Voce (originale) | 7/15 |
| Voce (corretta) | 7/15 |
| Fusione semplice 0,6 | 5/15 |
| Testo (FEEL-it) | 4/15 |

La fusione andava peggio della voce da sola: il testo rispondeva `sadness` anche a frasi neutre ("Domani vado all'università alle 9") e trascinava il risultato. Da qui il passaggio a `equa` con peso 0,8, che sulle 8 registrazioni di `casi.csv` dà 5/8 (testo da solo 2/8). Campione piccolo: i numeri sono indicativi.

Per tarare la fusione su nuove registrazioni:

```bash
python patch_salva_prove.py   # una volta: ogni analisi viene salvata in prove/
python valuta_fusioni.py      # confronta tutte le fusioni sulle prove salvate
```

`valuta_fusioni.py` chiede di etichettare le registrazioni in `prove/etichette.csv` e non ricarica i modelli.

## Emozioni dal testo: confronto di modelli su tre dataset

`bert/valuta_bert.py` valuta un modello di emozioni dal testo (inglese) su uno di tre dataset, frase per frase e senza il contesto del dialogo. Misura emozioni e sentiment (accuratezza, UAR, F1 pesata, richiamo per classe, matrice di confusione) e il tempo per frase.

**Modelli**

- [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) (default): DistilRoBERTa, 7 etichette (le stesse di MELD). Addestrato su 6 dataset, **tra cui MELD e l'Emotions dataset**.
- [`boltuix/bert-emotion`](https://huggingface.co/boltuix/bert-emotion): BERT molto piccolo (~11M parametri, 45 MB), 13 etichette. Addestrato **sull'Emotions dataset**.

**Dataset**

| Dataset | Testo | Emozioni | Sentiment |
|---|---|---|---|
| `meld` | 2.610 battute di *Friends* (test set) | 7 | annotato (3 classi) |
| `dailydialog` | 7.740 turni di dialoghi quotidiani scritti (test set) | 7, come MELD (82% neutral) | ricavato dall'emozione, surprise esclusa |
| `emotion` | 2.000 frasi tipo blog/tweet (test set) | 6, con love e senza neutral | polarità ricavata dall'emozione, surprise esclusa |

DailyDialog è l'unico dei tre che nessuno dei due modelli ha visto in addestramento, quindi è il confronto più affidabile. I risultati su MELD e soprattutto sull'Emotions dataset sono ottimistici.

```bash
cd bert
# MELD
curl -L -o test_sent_emo.csv https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv
# DailyDialog (il link ufficiale http://yanran.li/files/ijcnlp_dailydialog.zip non funziona più: copia su Hugging Face)
curl -L -o dd.zip https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/test.zip
unzip -j dd.zip test/dialogues_test.txt test/dialogues_emotion_test.txt -d dailydialog
# Emotions dataset: test.txt da Kaggle ("Emotions dataset for NLP"), da mettere in bert/

python valuta_bert.py --dataset dailydialog                       # modello di default (j-hartmann)
python valuta_bert.py --dataset meld --modello boltuix/bert-emotion
python valuta_bert.py --dataset emotion --n 300                   # prova veloce
```

Le etichette dei modelli sono ridotte a quelle del dataset (per boltuix: love, desire → joy su MELD/DailyDialog e → love sull'Emotions dataset; shame, guilt → sadness; confusion → surprise; sarcasm → neutral, ignorato dove neutral non esiste). Le predizioni sono salvate in `risultati_<modello>_<dataset>.csv`.

**Risultati sulle emozioni** (accuratezza / UAR)

| Dataset | Sempre la classe più frequente | boltuix | j-hartmann |
|---|---|---|---|
| **DailyDialog** (non visto) | 81,7% (neutral) | 56,1% / 40,4% | **61,1% / 52,1%** |
| MELD (visto da j-hartmann) | 48,1% (neutral) | 44,6% / 27,3% | 49,8% / 39,4% |
| Emotions dataset (visto da entrambi) | 34,8% (joy) | 94,7% / 93,5% | 84,0% / 73,9% |

**Risultati sul sentiment** (accuratezza / UAR)

| Dataset | boltuix | j-hartmann |
|---|---|---|
| DailyDialog (3 classi) | 67,1% / 63,6% | 65,2% / 65,8% |
| MELD (3 classi) | 50,9% / 48,5% | 60,7% / 57,8% |
| Emotions dataset (2 classi) | 99,5% / 99,5% | 95,3% / 94,9% |

Osservazioni:

- **DailyDialog:** j-hartmann riconosce molto meglio le emozioni minori (anger 32% contro 14%, disgust 49% contro 19%, surprise 80% contro 52%) e scambia meno spesso il neutro per joy (richiamo di neutral 64% contro 54%), ma riconosce meno joy (45% contro 78%). Entrambi restano sotto chi risponde sempre "neutral": il problema principale è che danno un'emozione a frasi neutre. Nel sentiment j-hartmann trova molto meglio il negativo (81% contro 46%), boltuix il positivo (79% contro 49%).
- **MELD:** boltuix va peggio di chi risponde sempre "neutral"; delle 345 frasi arrabbiate, 67 le classifica `joy`.
- **Emotions dataset:** il 94,7% di boltuix riflette l'addestramento sullo stesso dataset. j-hartmann non ha la classe love (richiamo 0%, finisce quasi tutta in joy).
- **Velocità** (MacBook Air M2, mediana per frase): boltuix ~3 ms, j-hartmann ~7 ms. Entrambi adatti al tempo reale.

## File

| File | Descrizione |
|---|---|
| `analizza_audio.py` | Modelli, fusione, mappatura verso il vocabolario del VH |
| `app.py` | Server Flask con gli endpoint `/mood` e `/mood/audio` |
| `valuta.py` | Valutazione su `casi.csv`, produce `risultati.csv` |
| `patch_salva_prove.py` | Fa salvare a `analizza_audio.py` ogni analisi in `prove/` |
| `valuta_fusioni.py` | Confronta fusioni e pesi sulle prove salvate, senza ricaricare i modelli |
| `bert/valuta_bert.py` | Valuta un modello di emozioni dal testo su MELD, DailyDialog o Emotions dataset (emozioni, sentiment, tempi) |
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
- Il peso testo/voce (equa, 0,8) è scelto su poche registrazioni di una sola persona.
- Il codice di integrazione con Unity/Jason (messaggio `mood`, piani dell'agente, registrazione dal microfono) è nel progetto VEsNA-Unity locale, non in questo repository.
- I modelli di testo valutati e i tre dataset sono in inglese, mentre l'utente del VH parlerà italiano.
- Sul testo, l'unico confronto su dati mai visti dai modelli è DailyDialog, che è scritto e non parlato.

## Riferimenti

- S. Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*, ACL 2019.
- Y. Li et al., *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset*, IJCNLP 2017 (CC BY-NC-SA 4.0).
- E. Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP 2018 (Emotions dataset).
- J. Hartmann, *Emotion English DistilRoBERTa-base*, https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
- F. Catania, J. W. Wilke, F. Garzotto, *Emozionalmente: A Crowdsourced Corpus of Simulated Emotional Speech in Italian*, IEEE Transactions on Audio, Speech and Language Processing, vol. 33, 2025. DOI 10.1109/TASLPRO.2025.3540662. Dataset: DOI 10.5281/zenodo.12616095 (CC BY 4.0). La correzione della voce è addestrata su questo dataset.
