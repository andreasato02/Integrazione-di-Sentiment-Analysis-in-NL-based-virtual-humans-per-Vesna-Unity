import sys

from transformers import pipeline

if len(sys.argv) < 2:
    print("Uso: python test_audio.py /percorso/del/file.m4a")
    sys.exit(1)

clf = pipeline(
    "audio-classification",
    model="hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0",
    top_k=None,
)

for r in clf(sys.argv[1]):
    print(r["label"], round(r["score"], 3))