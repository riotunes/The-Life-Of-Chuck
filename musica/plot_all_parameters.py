"""Plot All Parameters 1D

Legge la cache `audio/music_analysis_cache.json` e visualizza istogrammi 1D
per TUTTI i parametri calcolati da music_score.py.

Output:
- plots/all_parameters_1d.png

Usage:
    python plot_all_parameters.py
    python plot_all_parameters.py --cache audio/music_analysis_cache.json --out plots/all_parameters_1d.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['font.size'] = 8


def load_tracks(cache_path: Path) -> list[dict]:
    with cache_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    tracks = [t for t in data.get('tracks', []) if t.get('analyzed', False)]
    return tracks


def series(tracks: list[dict], key: str) -> np.ndarray:
    """Estrae i valori per un campo specifico."""
    vals = []
    for t in tracks:
        v = t.get(key, None)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        vals.append(v)
    return np.array(vals, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot 1D di tutti i parametri')
    parser.add_argument('--cache', type=str, default='audio/music_analysis_cache.json')
    parser.add_argument('--out', type=str, default='plots/all_parameters_1d.png')
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

    print(f"📊 Caricati {len(tracks)} brani dalla cache")

    # Lista di tutti i parametri da plottare (raggruppati per categoria)
    parameters = [
        # Valence components
        ('valence_mode', 'Valence Mode (min/maj)', (0, 1)),
        ('valence_brightness', 'Valence Brightness', (0, 1)),
        ('valence_dance', 'Valence Dance', (0, 1)),
        ('valence_penalty_factor', 'Valence Penalty Factor', None),
        ('valence_pre_penalty', 'Valence Pre-Penalty', (0, 1)),
        ('valence', 'Valence (final)', (0, 1)),
        
        # Mood
        ('mood_happy', 'Mood Happy', (0, 1)),
        ('mood_sad', 'Mood Sad', (0, 1)),
        ('mood_aggressive', 'Mood Aggressive', (0, 1)),
        ('mood_relaxed', 'Mood Relaxed', (0, 1)),
        
        # Energy & Arousal
        ('energy', 'Energy', (0, 1)),
        ('arousal', 'Arousal', (0, 1)),
        
        # Audio characteristics
        ('danceability', 'Danceability', None),
        ('instrumentalness', 'Instrumentalness', (0, 1)),
        ('acousticness', 'Acousticness', (0, 1)),
        ('electronicness', 'Electronicness', (0, 1)),
        
        # Rhythm
        ('bpm', 'BPM', None),
        
        # Key
        ('key_strength', 'Key Strength', (0, 1)),
    ]

    # Calcola layout griglia
    n_params = len(parameters)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.5 * n_rows))
    axes = axes.flatten()

    for i, (key, title, xlim) in enumerate(parameters):
        ax = axes[i]
        values = series(tracks, key)
        
        if values.size == 0:
            ax.set_title(f'{title}\n(no data)', color='red')
            ax.set_visible(True)
            continue
        
        # Calcola statistiche
        min_val, max_val = values.min(), values.max()
        mean_val, std_val = values.mean(), values.std()
        
        # Plot istogramma
        color = 'cyan' if 'valence' in key else ('lime' if 'mood' in key else ('orange' if key in ['energy', 'arousal'] else 'magenta'))
        ax.hist(values, bins=args.bins, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
        
        # Linea media
        ax.axvline(mean_val, color='yellow', linestyle='--', linewidth=1.5, label=f'mean={mean_val:.3f}')
        
        # Titolo con statistiche
        ax.set_title(f'{title}\nmin={min_val:.3f}, max={max_val:.3f}, μ={mean_val:.3f}, σ={std_val:.3f}')
        
        if xlim:
            ax.set_xlim(xlim)
        
        ax.grid(alpha=0.2)
        ax.set_ylabel('Count')

    # Nascondi assi vuoti
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Music Score Parameters — 1D Distributions ({len(tracks)} tracks)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'✅ Salvato: {out_path}')


if __name__ == '__main__':
    main()
