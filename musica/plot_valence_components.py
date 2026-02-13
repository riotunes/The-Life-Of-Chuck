"""Plot Valence Components

Reads the cache `audio/music_analysis_cache.json` and visualizes ONLY the components
contributing to valence, in addition to the final valence score.

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


def smooth_distribution(data: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Makes the data more uniform while partially preserving the original shape.
    
    Args:
        data: numpy array of values.
        intensity: float between 0.0 (original) and 1.0 (uniform).
    
    Returns:
        Array with smoothed distribution (sorted).
    """
    # 1. Sort the data
    sort_indices = np.argsort(data)
    sorted_data = data[sort_indices]
    
    # 2. Create the uniform target (straight line from min to max)
    uniform_target = np.linspace(sorted_data.min(), sorted_data.max(), len(data))
    
    # 3. Perform weighted average
    new_sorted_data = (1 - intensity) * sorted_data + intensity * uniform_target
    
    return new_sorted_data


plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 10


def load_tracks(cache_path: Path) -> list[dict]:
    """Loads track data from the JSON cache file."""
    with cache_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    # Only include tracks that have been successfully analyzed
    tracks = [t for t in data.get('tracks', []) if t.get('analyzed', False)]
    return tracks


def series(tracks: list[dict], key: str, *, clip01: bool = True) -> np.ndarray:
    """Extracts numeric values for a specific field/key."""
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
    parser = argparse.ArgumentParser(description='Plot valence components')
    parser.add_argument('--cache', type=str, default='audio/music_analysis_cache.json')
    parser.add_argument('--out', type=str, default='plots/valence_components.png')
    parser.add_argument('--bins', type=int, default=30)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    cache_path = (script_dir / args.cache).resolve()
    out_path = (script_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        raise FileNotFoundError(f'Cache not found: {cache_path}')

    tracks = load_tracks(cache_path)
    if not tracks:
        raise RuntimeError('No analyzed tracks found in the cache')

    # Components saved by music_score.py
    mode = series(tracks, 'valence_mode')
    brightness = series(tracks, 'valence_brightness')
    dance = series(tracks, 'valence_dance')
    penalty = series(tracks, 'valence_penalty_factor')
    pre = series(tracks, 'valence_pre_penalty')
    final = series(tracks, 'valence')

    # Apply smoothing (bending) to the min/maj mode distribution
    # intensity=0.5 -> 50% original, 50% uniform
    mode_smoothed = smooth_distribution(mode, intensity=0.5)

    missing_any = any(x.size == 0 for x in (mode, brightness, dance, penalty, pre, final))
    if missing_any:
        raise RuntimeError(
            'Valence components missing from cache. Regenerate the cache with:\n'
            '  python music_score.py --analyze-only --clear-cache\n'
            'then rerun this script.'
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

    print(f'✅ Saved: {out_path}')


if __name__ == '__main__':
    main()