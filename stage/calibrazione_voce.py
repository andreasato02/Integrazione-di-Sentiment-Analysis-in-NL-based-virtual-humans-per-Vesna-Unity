"""Correzione delle probabilita' del modello vocale (vedi calibra_emozionalmente.py).

Il modello vocale tende a rispondere "joy" troppo spesso. Una regressione
logistica addestrata sul train di Emozionalmente rimette in equilibrio le sue
7 probabilita'. Sul test set (speaker mai visti) l'UAR sale da 46,8% a 56,2%.

Se calibrazione.npz manca, calibra() restituisce le probabilita' invariate.
"""
import os

import numpy as np

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrazione.npz")

try:
    _d = np.load(_FILE, allow_pickle=True)
    _CLASSI = [str(c) for c in _d["classi"]]
    _W, _B, _MEDIA, _STD = _d["W"], _d["b"], _d["media"], _d["std"]
    DISPONIBILE = True
except Exception:
    DISPONIBILE = False


def calibra(punteggi):
    """punteggi: dict {emozione: probabilita'} -> dict corretto che somma a 1."""
    if not DISPONIBILE:
        return dict(punteggi)
    x = np.array([punteggi.get(c, 0.0) for c in _CLASSI], dtype=float)
    f = (np.log(np.clip(x, 1e-6, 1.0)) - _MEDIA) / _STD
    z = f @ _W + _B
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return {c: float(v) for c, v in zip(_CLASSI, p)}
