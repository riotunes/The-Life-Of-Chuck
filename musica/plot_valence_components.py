"""Plot Valence Components

Legge la cache `audio/music_analysis_cache.json` e visualizza SOLO i componenti che
contribuiscono alla valence, oltre alla valence finale.

Output:
- plots/valence_components.png

Usage:
    python plot_valence_components.py
    python plot_valence_components.py --cache audio/music_analysis_cache.json --out plots/valence_components.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ammorbidisci_distribuzione(dati: np.ndarray, intensita: float = 0.5) -> np.ndarray:
    """
    Rende i dati più uniformi preservando parzialmente la forma originale.
    
    Args:
        dati: array numpy dei valori.
        intensita: float tra 0.0 (originale) e 1.0 (uniforme).
    
    Returns:
        Array con distribuzione smussata (ordinato).
    """
    # 1. Ordina i dati
    indici_ordinamento = np.argsort(dati)
    dati_ordinati = dati[indici_ordinamento]
    
    # 2. Crea il target uniforme (linea retta dal min al max)
    target_uniforme = np.linspace(dati_ordinati.min(), dati_ordinati.max(), len(dati))
    
    # 3. Fai la media pesata
    nuovi_dati_ordinati = (1 - intensita) * dati_ordinati + intensita * target_uniforme
    
    return nuovi_dati_ordinati


plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 10


def load_tracks(cache_path: Path) -> list[dict]:
    with cache_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    tracks = [t for t in data.get('tracks', []) if t.get('analyzed', False)]
    return tracks


def series(tracks: list[dict], key: str, *, clip01: bool = True) -> np.ndarray:
    vals = []
    for t in tracks:
        v = t.get(key, None)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if clip01:
            v = max(0.0, min(1.0, v))
        vals.append(v)
    return np.array(vals, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot dei componenti della valence')
    parser.add_argument('--cache', type=str, default='audio/music_analysis_cache.json')
    parser.add_argument('--out', type=str, default='plots/valence_components.png')
    parser.add_argument('--bins', type=int, default=30)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cache_path = (script_dir / args.cache).resolve()
    out_path = (script_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        raise FileNotFoundError(f'Cache non trovata: {cache_path}')

    tracks = load_tracks(cache_path)
    if not tracks:
        raise RuntimeError('Nessuna traccia analizzata trovata nella cache')

    # Componenti salvati da music_score.py
    mode = series(tracks, 'valence_mode')
    brightness = series(tracks, 'valence_brightness')
    dance = series(tracks, 'valence_dance')
    penalty = series(tracks, 'valence_penalty_factor')
    pre = series(tracks, 'valence_pre_penalty')
    final = series(tracks, 'valence')

    # Applica smoothing (bending) alla distribuzione del mode min/maj
    # intensita=0.5 -> 50% originale, 50% uniforme
    mode_smoothed = ammorbidisci_distribuzione(mode, intensita=0.5)

    missing_any = any(x.size == 0 for x in (mode, brightness, dance, penalty, pre, final))
    if missing_any:
        raise RuntimeError(
            'Componenti valence mancanti nella cache. Rigenera la cache con:\n'
            '  python music_score.py --analyze-only --clear-cache\n'
            'poi rilancia questo script.'
        )

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    plots = [
        ('Mode contribution (valence_mode) — SMOOTHED', mode_smoothed),
        ('Brightness contribution (valence_brightness)', brightness),
        ('Dance contribution (valence_dance)', dance),
        ('Dissonance penalty factor (valence_penalty_factor)', penalty),
        ('Pre-penalty valence (valence_pre_penalty)', pre),
        ('Final valence (valence)', final),
    ]

    for ax, (title, values) in zip(axes, plots):
        ax.hist(values, bins=args.bins, color='cyan', alpha=0.85)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)

    fig.suptitle('Valence Components — Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'✅ Salvato: {out_path}')


if __name__ == '__main__':
    main()
