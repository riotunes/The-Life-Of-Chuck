"""
Visualizzazione dei dati di analisi musicale.
Legge il file music_analysis_cache.json e crea grafici interattivi.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import necessario per grafici 3D

# Configurazione stile
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_analysis_data(cache_path: str) -> dict:
    """Carica i dati dal file JSON."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def shorten_name(filename: str, max_len: int = 25) -> str:
    """Accorcia il nome del file per la visualizzazione."""
    name = Path(filename).stem
    if len(name) > max_len:
        return name[:max_len-3] + "..."
    return name


def plot_mood_comparison(tracks: list, save_path: str = None):
    """Grafico a barre comparativo dei mood per ogni brano."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [shorten_name(t['filename']) for t in tracks]
    x = np.arange(len(names))
    width = 0.2
    
    happy = [t.get('mood_happy', 0) for t in tracks]
    sad = [t.get('mood_sad', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    relaxed = [t.get('mood_relaxed', 0) for t in tracks]
    
    bars1 = ax.bar(x - 1.5*width, happy, width, label='😊 Happy', color='#FFD700')
    bars2 = ax.bar(x - 0.5*width, sad, width, label='😢 Sad', color='#4169E1')
    bars3 = ax.bar(x + 0.5*width, aggressive, width, label='😤 Aggressive', color='#DC143C')
    bars4 = ax.bar(x + 1.5*width, relaxed, width, label='😌 Relaxed', color='#32CD32')
    
    ax.set_ylabel('Score (0-1)')
    ax.set_title('🎭 Mood Comparison per Track')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_arousal_valence(tracks: list, save_path: str = None):
    """Scatter plot Arousal vs Valence (modello circomplesso delle emozioni)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    arousal = [t.get('arousal', 0.5) for t in tracks]
    valence = [t.get('valence', 0.5) for t in tracks]
    energy = [t.get('energy', 0.5) for t in tracks]
    names = [shorten_name(t['filename'], 20) for t in tracks]
    
    # Scatter con dimensione basata su energia
    sizes = [e * 300 + 50 for e in energy]
    scatter = ax.scatter(valence, arousal, s=sizes, c=energy, 
                        cmap='plasma', alpha=0.7, edgecolors='white')
    
    # Etichette per ogni punto
    for i, name in enumerate(names):
        ax.annotate(name, (valence[i], arousal[i]), 
                   fontsize=8, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')
    
    # Quadranti emotivi
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.3)
    
    # Etichette quadranti
    ax.text(0.25, 0.85, '😤 Teso/Arrabbiato', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.75, 0.85, '😊 Felice/Eccitato', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.25, 0.15, '😢 Triste/Depresso', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.75, 0.15, '😌 Calmo/Rilassato', ha='center', fontsize=11, alpha=0.7)
    
    ax.set_xlabel('Valence (Negativo ← → Positivo)')
    ax.set_ylabel('Arousal (Calmo ← → Eccitato)')
    ax.set_title('🎯 Mappa Emotiva: Arousal vs Valence')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Colorbar per energia
    cbar = plt.colorbar(scatter)
    cbar.set_label('Energia')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bpm_energy(tracks: list, save_path: str = None):
    """Scatter plot BPM vs Energy con mood dominante come colore."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bpm = [t.get('bpm', 0) for t in tracks]  # ora normalizzato
    energy = [t.get('energy', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Determina mood dominante per ogni track
    def get_dominant_mood(t):
        moods = {
            'happy': t.get('mood_happy', 0),
            'sad': t.get('mood_sad', 0),
            'aggressive': t.get('mood_aggressive', 0),
            'relaxed': t.get('mood_relaxed', 0)
        }
        return max(moods, key=moods.get)
    
    mood_colors = {
        'happy': '#FFD700',
        'sad': '#4169E1',
        'aggressive': '#DC143C',
        'relaxed': '#32CD32'
    }
    
    colors = [mood_colors[get_dominant_mood(t)] for t in tracks]
    
    # Scatter
    scatter = ax.scatter(bpm, energy, s=200, c=colors, alpha=0.8, edgecolors='white')
    
    # Etichette
    for i, name in enumerate(names):
        ax.annotate(name, (bpm[i], energy[i]), 
                   fontsize=8, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')
    
    ax.set_xlabel('BPM')
    ax.set_ylabel('Energy')
    ax.set_title('⚡ BPM vs Energy (colore = mood dominante)')
    ax.grid(alpha=0.3)
    
    # Legenda custom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFD700', label='😊 Happy'),
        Patch(facecolor='#4169E1', label='😢 Sad'),
        Patch(facecolor='#DC143C', label='😤 Aggressive'),
        Patch(facecolor='#32CD32', label='😌 Relaxed')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_radar_chart(tracks: list, track_indices: list = None, save_path: str = None):
    """Radar chart per confrontare le features di più brani."""
    if track_indices is None:
        # Prendi i primi 4 brani se non specificato
        track_indices = list(range(min(4, len(tracks))))
    
    categories = ['Energy', 'Arousal', 'Valence', 'Danceability', 
                  'Happy', 'Sad', 'Aggressive', 'Relaxed']
    N = len(categories)
    
    # Angoli per il radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Chiudi il cerchio
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(track_indices)))
    
    for idx, track_idx in enumerate(track_indices):
        t = tracks[track_idx]
        values = [
            t.get('energy', 0),
            t.get('arousal', 0),
            t.get('valence', 0),
            min(1.0, t.get('danceability', 0)),  # Normalizza danceability
            t.get('mood_happy', 0),
            t.get('mood_sad', 0),
            t.get('mood_aggressive', 0),
            t.get('mood_relaxed', 0)
        ]
        values += values[:1]  # Chiudi il cerchio
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=shorten_name(t['filename'], 20), color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('🎵 Audio Features Radar')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_key_distribution(tracks: list, save_path: str = None):
    """Distribuzione delle tonalità (major vs minor)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Conta major vs minor
    scales = [t.get('scale', 'unknown') for t in tracks]
    scale_counts = {'major': scales.count('major'), 'minor': scales.count('minor')}
    
    # Pie chart per major/minor
    colors_pie = ['#32CD32', '#4169E1']
    ax1.pie(scale_counts.values(), labels=['Major', 'Minor'], 
           autopct='%1.0f%%', colors=colors_pie, startangle=90,
           explode=(0.05, 0.05), shadow=True)
    ax1.set_title('🎹 Distribuzione Major vs Minor')
    
    # Bar chart per tonalità specifiche
    keys = [f"{t.get('key', '?')} {t.get('scale', '?')}" for t in tracks]
    key_counts = {}
    for k in keys:
        key_counts[k] = key_counts.get(k, 0) + 1
    
    sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
    key_names = [k[0] for k in sorted_keys]
    key_values = [k[1] for k in sorted_keys]
    
    colors_bar = ['#32CD32' if 'major' in k.lower() else '#4169E1' for k in key_names]
    ax2.barh(key_names, key_values, color=colors_bar)
    ax2.set_xlabel('Numero di brani')
    ax2.set_title('🎼 Tonalità specifiche')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_features_heatmap(tracks: list, save_path: str = None):
    """Heatmap di tutte le features per tutti i brani."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(tracks) * 0.5)))
    
    features = ['energy', 'arousal', 'valence', 'danceability', 
                'mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed']
    
    data = []
    names = []
    for t in tracks:
        row = [min(1.0, t.get(f, 0)) for f in features]  # Normalizza a 1
        data.append(row)
        names.append(shorten_name(t['filename'], 30))
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels([f.replace('mood_', '').title() for f in features])
    ax.set_yticklabels(names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Aggiungi valori nelle celle
    for i in range(len(names)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_title('🔥 Features Heatmap')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_histograms(tracks: list, save_path: str = None):
    """Istogrammi delle principali feature (incluse le nuove metriche).

    Include: bpm, energy, arousal, valence, danceability (legacy),
    instrumentalness, acousticness, electronicness.
    """
    # Helper to safely extract numeric values
    def get_series(key: str, *, clip01: bool = False, fallback_key: str | None = None):
        vals = []
        for t in tracks:
            v = t.get(key, None)
            if v is None and fallback_key is not None:
                v = t.get(fallback_key, None)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if clip01:
                v = max(0.0, min(1.0, v))
            vals.append(v)
        return np.array(vals, dtype=float)

    series = [
        ('BPM', get_series('bpm')),
        ('Energy', get_series('energy', clip01=True)),
        ('Arousal', get_series('arousal', clip01=True)),
        ('Valence', get_series('valence', clip01=True)),
        ('Danceability (legacy)', get_series('danceability', clip01=True)),
        ('Instrumentalness', get_series('instrumentalness', clip01=True, fallback_key='danceability')),
        ('Acousticness', get_series('acousticness', clip01=True)),
        ('Electronicness', get_series('electronicness', clip01=True, fallback_key='mood_aggressive')),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (title, values) in zip(axes, series):
        if values.size == 0:
            ax.set_title(f'{title} (no data)')
            ax.axis('off')
            continue

        ax.hist(values, bins=30, color='cyan', alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    # Hide any extra axes if we ever change series length
    for ax in axes[len(series):]:
        ax.axis('off')

    fig.suptitle('📊 Feature Distributions (Histograms)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_valence_arousal_bpm(tracks: list, save_path: str = None):
    """Grafico 3D: Valence (X), Arousal (Y), BPM (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]  # ora normalizzato
    energy = [t.get('energy', 0.5) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Scatter plot 3D
    # Colore basato su Energy, Dimensione fissa o basata su altro
    scatter = ax.scatter(valence, arousal, bpm, c=energy, cmap='plasma', s=100, alpha=0.8, edgecolors='white')
    
    # Etichette
    for i, name in enumerate(names):
        ax.text(valence[i], arousal[i], bpm[i], name, fontsize=8)
    
    ax.set_xlabel('Valence (Positività)')
    ax.set_ylabel('Arousal (Attivazione)')
    ax.set_zlabel('BPM (Tempo)')
    ax.set_title('🧊 3D: Valence vs Arousal vs BPM')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Energy')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_energy_danceability_bpm(tracks: list, save_path: str = None):
    """Grafico 3D: Energy (X), Danceability (Y), BPM (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    energy = [t.get('energy', 0.5) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]  # ora normalizzato
    
    # Colore basato su Happy Mood
    happy = [t.get('mood_happy', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    scatter = ax.scatter(energy, danceability, bpm, c=happy, cmap='viridis', s=100, alpha=0.8, edgecolors='white')
    
    for i, name in enumerate(names):
        ax.text(energy[i], danceability[i], bpm[i], name, fontsize=8)
    
    ax.set_xlabel('Energy')
    ax.set_ylabel('Danceability')
    ax.set_zlabel('BPM')
    ax.set_title('💃 3D: Energy vs Danceability vs BPM')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Happy Score')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_mood_space(tracks: list, save_path: str = None):
    """Grafico 3D nello spazio dei mood: Happy (X), Sad (Y), Aggressive (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    happy = [t.get('mood_happy', 0) for t in tracks]
    sad = [t.get('mood_sad', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Colore basato su Relaxed (il quarto mood)
    relaxed = [t.get('mood_relaxed', 0) for t in tracks]
    
    scatter = ax.scatter(happy, sad, aggressive, c=relaxed, cmap='cool', s=100, alpha=0.8, edgecolors='white')
    
    for i, name in enumerate(names):
        ax.text(happy[i], sad[i], aggressive[i], name, fontsize=8)
    
    ax.set_xlabel('Happy')
    ax.set_ylabel('Sad')
    ax.set_zlabel('Aggressive')
    ax.set_title('🎭 3D Mood Space (Colore = Relaxed)')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Relaxed Score')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_5d_multidimensional(tracks: list, save_path: str = None):
    """
    Grafico 5D Multidimensionale:
    X: Valence
    Y: Arousal
    Z: BPM
    Colore: Aggressive
    Dimensione: Danceability
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]  # ora normalizzato
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Normalizza danceability per la dimensione dei punti (min 50, max 500)
    sizes = [d * 450 + 50 for d in danceability]
    
    # Scatter plot
    scatter = ax.scatter(valence, arousal, bpm, 
                        c=aggressive, cmap='Reds', 
                        s=sizes, alpha=0.8, edgecolors='white', linewidth=1)
    
    # Linee di proiezione verso il piano di base (per aiutare la percezione 3D)
    min_z = min(bpm) - 10
    for i in range(len(tracks)):
        ax.plot([valence[i], valence[i]], [arousal[i], arousal[i]], [min_z, bpm[i]], 
               color='gray', linestyle=':', alpha=0.3)
        ax.text(valence[i], arousal[i], bpm[i], names[i], fontsize=9)
    
    ax.set_xlabel('Valence (Positività)')
    ax.set_ylabel('Arousal (Attivazione)')
    ax.set_zlabel('BPM (Tempo)')
    ax.set_title('🌌 5D Analysis: Valence/Arousal/BPM\n(Colore=Aggressive, Dimensione=Danceability)')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Aggressive Score (Colore)')
    
    # Legenda per la dimensione
    # Creiamo punti fittizi per la legenda
    l1 = ax.scatter([], [], [], s=50, c='gray', edgecolors='none', label='Low Danceability')
    l2 = ax.scatter([], [], [], s=275, c='gray', edgecolors='none', label='Med Danceability')
    l3 = ax.scatter([], [], [], s=500, c='gray', edgecolors='none', label='High Danceability')
    
    ax.legend(handles=[l1, l2, l3], loc='upper left', title="Dimensione", frameon=True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_5d_planes(tracks: list, save_path: str = None):
    """
    Visualizza le 5 dimensioni su 5 piani (grafici) diversi per analizzare le correlazioni.
    Dimensioni: Valence, Arousal, BPM, Danceability, Aggressive
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('🌌 5D Analysis: Multi-Plane View', fontsize=16)
    
    # Dati
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Helper per scatter plot
    def create_plane(ax_pos, x_data, y_data, x_label, y_label, title, color_data=None, cmap='viridis'):
        ax = fig.add_subplot(2, 3, ax_pos)
        c = color_data if color_data is not None else 'cyan'
        scatter = ax.scatter(x_data, y_data, c=c, cmap=cmap if color_data is not None else None, 
                           s=100, alpha=0.7, edgecolors='white')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Annotazioni
        for i, name in enumerate(names):
            ax.annotate(name, (x_data[i], y_data[i]), fontsize=7, alpha=0.8)
        return scatter

    # Piano 1: Emotional (Valence vs Arousal) - Colore: Aggressive
    create_plane(1, valence, arousal, 'Valence', 'Arousal', '1. Emotional Plane', aggressive, 'Reds')
    
    # Piano 2: Rhythmic (BPM vs Danceability) - Colore: Arousal
    create_plane(2, bpm, danceability, 'BPM', 'Danceability', '2. Rhythmic Plane', arousal, 'plasma')
    
    # Piano 3: Intensity (Aggressive vs Arousal) - Colore: BPM
    create_plane(3, aggressive, arousal, 'Aggressive', 'Arousal', '3. Intensity Plane', bpm, 'cool')
    
    # Piano 4: Mood-Tempo (Valence vs BPM) - Colore: Danceability
    create_plane(4, valence, bpm, 'Valence', 'BPM', '4. Mood-Tempo Plane', danceability, 'viridis')
    
    # Piano 5: Action (Aggressive vs Danceability) - Colore: Valence
    create_plane(5, aggressive, danceability, 'Aggressive', 'Danceability', '5. Action Plane', valence, 'RdYlGn')
    
    # Slot 6: Spiegazione
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.5, 
             "Legenda Piani:\n\n"
             "1. Emotional: Emozioni base (Colore: Aggressive)\n"
             "2. Rhythmic: Struttura ritmica (Colore: Arousal)\n"
             "3. Intensity: Forza e attivazione (Colore: BPM)\n"
             "4. Mood-Tempo: Felicità/Velocità (Colore: Dance)\n"
             "5. Action: Aggressività ballabile (Colore: Valence)\n",
             fontsize=11, va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Funzione principale per generare tutti i grafici."""
    # Percorso cache / output (relativi al repo)
    script_dir = Path(__file__).parent
    CACHE_PATH = str(script_dir / 'audio' / 'music_analysis_cache.json')
    OUTPUT_DIR = str(script_dir / 'plots')
    
    # Crea cartella output se non esiste
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Carica dati
    try:
        data = load_analysis_data(CACHE_PATH)
        tracks = [t for t in data.get('tracks', []) if t.get('analyzed', False)]
        
        if not tracks:
            print("❌ Nessun brano analizzato trovato nella cache!")
            return
        
        print(f"📊 Trovati {len(tracks)} brani analizzati\n")
        
        # Genera grafici
        print("1️⃣  Generando Mood Comparison...")
        plot_mood_comparison(tracks, f"{OUTPUT_DIR}/mood_comparison.png")
        
        print("2️⃣  Generando Arousal vs Valence...")
        plot_arousal_valence(tracks, f"{OUTPUT_DIR}/arousal_valence.png")
        
        print("3️⃣  Generando BPM vs Energy...")
        plot_bpm_energy(tracks, f"{OUTPUT_DIR}/bpm_energy.png")
        
        print("4️⃣  Generando Radar Chart...")
        plot_radar_chart(tracks, save_path=f"{OUTPUT_DIR}/radar_chart.png")

        print("4️⃣b Generando Feature Histograms...")
        plot_feature_histograms(tracks, save_path=f"{OUTPUT_DIR}/feature_histograms.png")
        
        print("5️⃣  Generando Key Distribution...")
        plot_key_distribution(tracks, f"{OUTPUT_DIR}/key_distribution.png")
        
        print("6️⃣  Generando Features Heatmap...")
        plot_all_features_heatmap(tracks, f"{OUTPUT_DIR}/features_heatmap.png")
        
        print("7️⃣  Generando 3D Valence vs Arousal vs BPM...")
        plot_3d_valence_arousal_bpm(tracks, f"{OUTPUT_DIR}/3d_valence_arousal_bpm.png")
        
        print("8️⃣  Generando 3D Energy vs Danceability vs BPM...")
        plot_3d_energy_danceability_bpm(tracks, f"{OUTPUT_DIR}/3d_energy_danceability_bpm.png")
        
        print("9️⃣  Generando 3D Mood Space...")
        plot_3d_mood_space(tracks, f"{OUTPUT_DIR}/3d_mood_space.png")
        
        print("🔟 Generando 5D Multidimensional Plot...")
        plot_5d_multidimensional(tracks, f"{OUTPUT_DIR}/5d_multidimensional.png")
        
        print("1️⃣1️⃣ Generando 5D Multi-Plane View...")
        plot_5d_planes(tracks, f"{OUTPUT_DIR}/5d_planes.png")
        
        print(f"\n✅ Grafici salvati in: {OUTPUT_DIR}")
        
    except FileNotFoundError:
        print(f"❌ Cache non trovata: {CACHE_PATH}")
        print("   Esegui prima music_score.py per generare l'analisi.")


if __name__ == "__main__":
    main()
