"""musica/visualize_umap_3d.py

UMAP + Plotly 3D Visualization for Music Data
Visualizza i brani nello spazio delle features ridotto a 3D con UMAP.
Grafico interattivo con Plotly.

Usage:
    python visualize_umap_3d.py \
        [--cache path/to/cache.json] \
        [--output plot.html] \
        [--color-by valence|arousal|bpm|danceability|aggressive] \
        [--drop-valence] [--valence-weight 1.0]
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP non installato. Installa con: pip install umap-learn")

try:
    import plotly.express as px
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
    # Filtra solo tracce analizzate
    return [t for t in tracks if t.get('analyzed', False)]


def extract_features(tracks: list, *, include_valence: bool = True, valence_weight: float = 1.0) -> tuple:
    """
    Estrae le features dai brani e le normalizza.
    
    Returns:
        feature_matrix: np.ndarray di shape (n_tracks, 5)
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
        'electronicness': []
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

        # Usa direttamente il bpm normalizzato (ora in 'bpm')
        bpm_norm = track.get('bpm', 0.5)

        vec = [
            raw_arousal,
            raw_valence * valence_weight,
            bpm_norm,
            max(0.0, min(1.0, float(raw_instrumentalness))),
            max(0.0, min(1.0, float(raw_electronicness)))
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


def compute_umap_3d(features: np.ndarray, n_neighbors: int = 15, 
                    min_dist: float = 0.1, metric: str = 'euclidean') -> np.ndarray:
    """
    Applica UMAP per ridurre le features a 3 dimensioni.
    
    Args:
        features: matrice (n_samples, n_features)
        n_neighbors: numero di vicini per UMAP
        min_dist: distanza minima tra punti
        metric: metrica di distanza
    
    Returns:
        embedding: np.ndarray di shape (n_samples, 3)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP non disponibile. Installa con: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    
    embedding = reducer.fit_transform(features)
    return embedding


def compute_pca_3d(features: np.ndarray) -> np.ndarray:
    """Riduce le features a 3D con PCA (SVD), come baseline lineare."""
    if features.ndim != 2:
        raise ValueError('features must be a 2D array')
    if features.shape[0] < 3:
        raise ValueError('Need at least 3 samples for PCA')

    if not np.isfinite(features).all():
        raise ValueError('features contain NaN/Inf; cannot compute PCA safely')

    # Centering
    X = features.astype(float)
    X = X - X.mean(axis=0, keepdims=True)

    # SVD: X = U S Vt  -> components are rows of Vt
    # Project onto first 3 PCs: X @ Vt[:3].T
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:3].T

    # In alcuni ambienti BLAS/Numpy può emettere warning spurie su matmul
    # anche quando i dati sono finiti. Silenziamo localmente per non sporcare l'output.
    with np.errstate(all='ignore'):
        emb = X @ components
    return emb


def compute_embedding_3d(
    features: np.ndarray,
    *,
    method: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    """Compute a 3D embedding with the requested method."""
    if method == 'umap':
        return compute_umap_3d(features, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    if method == 'pca':
        return compute_pca_3d(features)
    raise ValueError(f'Unknown method: {method}')


def create_3d_plot(
    embedding: np.ndarray,
    track_names: list,
    metadata: dict,
    *,
    output_path: str = None,
    color_by: str = 'valence',
    colorscale: str = None,
    plot_title: str = '🎵 Music Feature Space (UMAP 3D)',
    axis_titles: tuple[str, str, str] = ('UMAP 1', 'UMAP 2', 'UMAP 3'),
) -> go.Figure:
    """
    Crea un grafico 3D interattivo con Plotly.
    
    Args:
        embedding: coordinate UMAP 3D
        track_names: nomi dei brani
        metadata: info per hover
        output_path: path per salvare HTML (opzionale)
    
    Returns:
        fig: Plotly Figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly non disponibile. Installa con: pip install plotly")
    
    # Crea dataframe-like per plotly
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
    
    # Colore: di default valence (positivo = felice)
    allowed_color_by = {'valence', 'arousal', 'bpm', 'instrumentalness', 'electronicness'}
    if color_by not in allowed_color_by:
        raise ValueError(f"color_by must be one of: {sorted(allowed_color_by)}")

    colors = metadata[color_by]
    
    # Usa arousal per dimensione punti
    sizes = [8 + a * 12 for a in metadata['arousal']]
    
    marker_dict = dict(
        size=sizes,
        color=colors,
        colorscale=colorscale,
        colorbar=dict(title=color_by.title()),
        opacity=0.8,
        line=dict(width=0.5, color='white'),
    )

    if color_by == 'valence':
        marker_dict['colorscale'] = colorscale or 'RdYlGn'  # Rosso (triste) -> Verde (felice)
        marker_dict['colorbar'] = dict(
            title='Valence',
            tickvals=[0, 0.5, 1],
            ticktext=['Sad', 'Neutral', 'Happy'],
        )
    else:
        marker_dict['colorscale'] = colorscale or 'Viridis'

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=marker_dict,
        text=track_names,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    )])
    
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
            bgcolor='rgb(20, 20, 30)',
            xaxis=dict(gridcolor='gray', zerolinecolor='gray'),
            yaxis=dict(gridcolor='gray', zerolinecolor='gray'),
            zaxis=dict(gridcolor='gray', zerolinecolor='gray'),
        ),
        paper_bgcolor='rgb(10, 10, 20)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=50),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=12,
            font_family='monospace'
        )
    )
    
    # Salva HTML se richiesto
    if output_path:
        fig.write_html(output_path)
        print(f"✅ Grafico salvato in: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualizza dati musicali con UMAP 3D')
    parser.add_argument('--cache', type=str, 
                        default='audio/music_analysis_cache.json',
                        help='Path al file cache JSON')
    parser.add_argument('--output', type=str, 
                        default=None,
                        help='Path output HTML (default dipende da --method)')
    parser.add_argument('--method', type=str, default='umap',
                        choices=['umap', 'pca'],
                        help='Metodo di riduzione a 3D (umap = non lineare, pca = baseline lineare)')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='Numero di vicini per UMAP')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='Distanza minima UMAP')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Metrica distanza UMAP (ignorata per PCA)')
    parser.add_argument('--drop-valence', action='store_true',
                        help='Rimuove la valence dalle feature usate da UMAP (hover/colore possono restare)')
    parser.add_argument('--valence-weight', type=float, default=1.0,
                        help='Peso moltiplicativo della valence nelle feature UMAP (0 = ignora valence)')
    parser.add_argument('--color-by', type=str, default='valence',
                        choices=['valence', 'arousal', 'bpm', 'instrumentalness', 'electronicness'],
                        help='Variabile usata per il colore dei punti')
    parser.add_argument('--colorscale', type=str, default=None,
                        help='Plotly colorscale (default: RdYlGn per valence, Viridis altrimenti)')
    parser.add_argument('--show', action='store_true',
                        help='Apri grafico nel browser')
    
    args = parser.parse_args()
    
    # Verifica dipendenze
    if not PLOTLY_AVAILABLE:
        print("\n❌ Plotly mancante. Installa con:")
        print("   pip install plotly")
        return
    if args.method == 'umap' and not UMAP_AVAILABLE:
        print("\n❌ UMAP non disponibile. Installa con:")
        print("   pip install umap-learn")
        return
    
    # Trova path cache
    script_dir = Path(__file__).parent
    cache_path = script_dir / args.cache
    
    if not cache_path.exists():
        print(f"❌ File cache non trovato: {cache_path}")
        print("   Esegui prima l'analisi musicale con music_score.py")
        return
    
    print(f"📂 Caricamento dati da: {cache_path}")
    tracks = load_music_data(cache_path)
    print(f"   Trovate {len(tracks)} tracce analizzate")
    
    if len(tracks) < 5:
        print("⚠️ Troppo poche tracce per UMAP (minimo 5)")
        return
    
    # Estrai features
    print("🔢 Estrazione features...")
    features, feature_names, track_names, metadata = extract_features(
        tracks,
        include_valence=(not args.drop_valence and args.valence_weight != 0),
        valence_weight=args.valence_weight,
    )
    print(f"   Features: {feature_names}")
    print(f"   Shape: {features.shape}")
    
    # UMAP
    if args.method == 'umap':
        print(f"🗺️ Applicando UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric})...")
    else:
        print("📐 Applicando PCA (baseline lineare)...")

    embedding = compute_embedding_3d(
        features,
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
    )
    print(f"   Embedding shape: {embedding.shape}")
    
    # Plot
    if args.output is None:
        default_output = 'plots/umap_3d_music.html' if args.method == 'umap' else 'plots/pca_3d_music.html'
        output_path = script_dir / default_output
    else:
        output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("📊 Creazione grafico 3D...")
    axis_titles = ('UMAP 1', 'UMAP 2', 'UMAP 3') if args.method == 'umap' else ('PC1', 'PC2', 'PC3')
    plot_title = '🎵 Music Feature Space (UMAP 3D)' if args.method == 'umap' else '🎵 Music Feature Space (PCA 3D)'

    fig = create_3d_plot(
        embedding,
        track_names,
        metadata,
        output_path=str(output_path),
        color_by=args.color_by,
        colorscale=args.colorscale,
        plot_title=plot_title,
        axis_titles=axis_titles,
    )
    
    if args.show:
        fig.show()
    
    print("\n✅ Completato!")
    print(f"   Apri {output_path} nel browser per visualizzare il grafico interattivo")


if __name__ == '__main__':
    main()
