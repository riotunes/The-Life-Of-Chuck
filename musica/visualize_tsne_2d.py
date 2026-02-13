"""musica/visualize_tsne_2d.py

t-SNE + Plotly 2D Visualization for Music Data
Visualizza i brani nello spazio delle features ridotto a 2D con t-SNE.
Grafico interattivo con Plotly.

Usage:
    python visualize_tsne_2d.py \
        [--cache path/to/cache.json] \
        [--output plot.html] \
        [--color-by valence|arousal|bpm|instrumentalness|electronicness] \
        [--drop-valence] [--valence-weight 1.0]
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn non installato. Installa con: pip install scikit-learn")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly non installato. Installa con: pip install plotly")


def load_music_data(cache_path: str) -> list:
    """Carica i dati delle tracce dal file cache JSON."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tracks = data.get('tracks', [])
    return [t for t in tracks if t.get('analyzed', False)]


def extract_features(tracks: list, *, include_valence: bool = True, valence_weight: float = 1.0) -> tuple:
    """
    Estrae le features dai brani e le normalizza.

    Returns:
        feature_matrix: np.ndarray
        feature_names: lista nomi features
        track_names: lista nomi file
        metadata: dict con info extra per hover
    """
    feature_names = ['arousal', 'valence', 'bpm', 'instrumentalness', 'electronicness']

    features = []
    track_names = []
    metadata = {
        'arousal': [],
        'valence': [],
        'bpm': [],
        'instrumentalness': [],
        'electronicness': [],
    }

    if valence_weight < 0:
        raise ValueError('valence_weight must be >= 0')

    for track in tracks:
        raw_arousal = track.get('arousal', 0.5)
        raw_valence = track.get('valence', 0.5)

        raw_instrumentalness = track.get('instrumentalness', None)
        if raw_instrumentalness is None:
            raw_instrumentalness = min(1.0, track.get('danceability', 0.0))

        raw_electronicness = track.get('electronicness', None)
        if raw_electronicness is None:
            raw_electronicness = track.get('mood_aggressive', 0.0)

        bpm_norm = track.get('bpm', 0.5)

        vec = [
            raw_arousal,
            raw_valence * valence_weight,
            bpm_norm,
            max(0.0, min(1.0, float(raw_instrumentalness))),
            max(0.0, min(1.0, float(raw_electronicness))),
        ]

        if not include_valence:
            vec.pop(1)

        features.append(vec)
        track_names.append(track.get('filename', 'Unknown'))
        metadata['arousal'].append(raw_arousal)
        metadata['valence'].append(raw_valence)
        metadata['bpm'].append(bpm_norm)
        metadata['instrumentalness'].append(max(0.0, min(1.0, float(raw_instrumentalness))))
        metadata['electronicness'].append(max(0.0, min(1.0, float(raw_electronicness))))

    if not include_valence:
        feature_names = ['arousal', 'bpm', 'instrumentalness', 'electronicness']

    return np.array(features), feature_names, track_names, metadata


def compute_tsne_2d(
    features: np.ndarray,
    *,
    perplexity: float = 10.0,
    metric: str = 'euclidean',
    learning_rate: str | float = 'auto',
    n_iter: int = 1500,
) -> np.ndarray:
    """Riduce le features a 2D con t-SNE."""
    if not SKLEARN_AVAILABLE:
        raise ImportError('scikit-learn non disponibile. Installa con: pip install scikit-learn')

    if features.ndim != 2:
        raise ValueError('features must be a 2D array')

    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError('Need at least 3 samples for t-SNE')

    max_perplexity = max(1.0, float(n_samples - 1))
    if perplexity >= n_samples:
        adjusted = min(30.0, max_perplexity)
        print(f"⚠️ perplexity={perplexity} troppo alta per {n_samples} campioni. Uso {adjusted:.2f}")
        perplexity = adjusted

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        learning_rate=learning_rate,
        max_iter=n_iter,
        init='pca',
        random_state=42,
    )

    return tsne.fit_transform(features)


def create_2d_plot(
    embedding: np.ndarray,
    track_names: list,
    metadata: dict,
    *,
    output_path: str = None,
    color_by: str = 'valence',
    colorscale: str = None,
    plot_title: str = '🎵 Music Feature Space (t-SNE 2D)',
) -> go.Figure:
    """Crea un grafico 2D interattivo con Plotly."""
    if not PLOTLY_AVAILABLE:
        raise ImportError('Plotly non disponibile. Installa con: pip install plotly')

    hover_text = []
    for i, name in enumerate(track_names):
        text = (
            f"<b>{name}</b><br>"
            f"Arousal: {metadata['arousal'][i]:.2f}<br>"
            f"Valence: {metadata['valence'][i]:.2f}<br>"
            f"BPM: {metadata['bpm'][i]:.1f}<br>"
            f"Instrumentalness: {metadata['instrumentalness'][i]:.2f}<br>"
            f"Electronicness: {metadata['electronicness'][i]:.2f}"
        )
        hover_text.append(text)

    allowed_color_by = {'valence', 'arousal', 'bpm', 'instrumentalness', 'electronicness'}
    if color_by not in allowed_color_by:
        raise ValueError(f"color_by must be one of: {sorted(allowed_color_by)}")

    marker_dict = dict(
        size=[8 + a * 12 for a in metadata['arousal']],
        color=metadata[color_by],
        colorscale=colorscale,
        colorbar=dict(title=color_by.title()),
        opacity=0.85,
        line=dict(width=0.5, color='white'),
    )

    if color_by == 'valence':
        marker_dict['colorscale'] = colorscale or 'RdYlGn'
        marker_dict['colorbar'] = dict(
            title='Valence',
            tickvals=[0, 0.5, 1],
            ticktext=['Sad', 'Neutral', 'Happy'],
        )
    else:
        marker_dict['colorscale'] = colorscale or 'Viridis'

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers+text',
                marker=marker_dict,
                text=None,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_text,
            )
        ]
    )

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=20)),
        xaxis=dict(title='t-SNE 1', gridcolor='gray', zerolinecolor='gray'),
        yaxis=dict(title='t-SNE 2', gridcolor='gray', zerolinecolor='gray'),
        paper_bgcolor='rgb(10, 10, 20)',
        plot_bgcolor='rgb(20, 20, 30)',
        font=dict(color='white'),
        margin=dict(l=40, r=20, b=40, t=60),
        hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_size=12, font_family='monospace'),
    )

    if output_path:
        fig.write_html(output_path)
        print(f"✅ Grafico salvato in: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualizza dati musicali con t-SNE 2D')
    parser.add_argument('--cache', type=str, default='audio/music_analysis_cache.json', help='Path al file cache JSON')
    parser.add_argument('--output', type=str, default='plots/tsne_2d_music.html', help='Path output HTML')
    parser.add_argument('--perplexity', type=float, default=10.0, help='Perplexity t-SNE (deve essere < numero tracce)')
    parser.add_argument('--metric', type=str, default='euclidean', help='Metrica distanza t-SNE')
    parser.add_argument('--learning-rate', type=str, default='auto', help='Learning rate t-SNE (es: auto, 50, 200)')
    parser.add_argument('--n-iter', type=int, default=1500, help='Numero iterazioni t-SNE')
    parser.add_argument('--drop-valence', action='store_true', help='Rimuove valence dalle feature usate da t-SNE')
    parser.add_argument('--valence-weight', type=float, default=1.0, help='Peso moltiplicativo valence nelle feature')
    parser.add_argument('--color-by', type=str, default='valence',
                        choices=['valence', 'arousal', 'bpm', 'instrumentalness', 'electronicness'],
                        help='Variabile usata per il colore dei punti')
    parser.add_argument('--colorscale', type=str, default=None,
                        help='Plotly colorscale (default: RdYlGn per valence, Viridis altrimenti)')
    parser.add_argument('--show', action='store_true', help='Apri grafico nel browser')
    args = parser.parse_args()

    if not PLOTLY_AVAILABLE:
        print('\n❌ Plotly mancante. Installa con:')
        print('   pip install plotly')
        return

    if not SKLEARN_AVAILABLE:
        print('\n❌ scikit-learn mancante. Installa con:')
        print('   pip install scikit-learn')
        return

    script_dir = Path(__file__).parent
    cache_path = script_dir / args.cache

    if not cache_path.exists():
        print(f'❌ File cache non trovato: {cache_path}')
        print('   Esegui prima l\'analisi musicale con music_score.py')
        return

    print(f'📂 Caricamento dati da: {cache_path}')
    tracks = load_music_data(str(cache_path))
    print(f'   Trovate {len(tracks)} tracce analizzate')

    if len(tracks) < 3:
        print('⚠️ Troppo poche tracce per t-SNE (minimo 3)')
        return

    print('🔢 Estrazione features...')
    features, feature_names, track_names, metadata = extract_features(
        tracks,
        include_valence=(not args.drop_valence and args.valence_weight != 0),
        valence_weight=args.valence_weight,
    )
    print(f'   Features: {feature_names}')
    print(f'   Shape: {features.shape}')

    try:
        learning_rate_value: str | float = float(args.learning_rate)
    except ValueError:
        learning_rate_value = args.learning_rate

    print(
        '🌀 Applicando t-SNE '
        f"(perplexity={args.perplexity}, metric={args.metric}, learning_rate={learning_rate_value}, n_iter={args.n_iter})..."
    )

    embedding = compute_tsne_2d(
        features,
        perplexity=args.perplexity,
        metric=args.metric,
        learning_rate=learning_rate_value,
        n_iter=args.n_iter,
    )
    print(f'   Embedding shape: {embedding.shape}')

    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print('📊 Creazione grafico 2D...')
    fig = create_2d_plot(
        embedding,
        track_names,
        metadata,
        output_path=str(output_path),
        color_by=args.color_by,
        colorscale=args.colorscale,
        plot_title='🎵 Music Feature Space (t-SNE 2D)',
    )

    if args.show:
        fig.show()

    print('\n✅ Completato!')
    print(f'   Apri {output_path} nel browser per visualizzare il grafico interattivo')


if __name__ == '__main__':
    main()
