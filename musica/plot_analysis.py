"""
Music analysis data visualization.
Reads music_analysis_cache.json and creates interactive plots.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necessary import for 3D plotting

# Style Configuration
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_analysis_data(cache_path: str) -> dict:
    """Loads data from the JSON file."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def shorten_name(filename: str, max_len: int = 25) -> str:
    """Shortens the filename for display purposes."""
    name = Path(filename).stem
    if len(name) > max_len:
        return name[:max_len-3] + "..."
    return name


def plot_mood_comparison(tracks: list, save_path: str = None):
    """Comparative bar chart of moods for each track."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [shorten_name(t['filename']) for t in tracks]
    x = np.arange(len(names))
    
    
    
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
    """Scatter plot: Arousal vs Valence (Circumplex Model of Affect)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    arousal = [t.get('arousal', 0.5) for t in tracks]
    valence = [t.get('valence', 0.5) for t in tracks]
    energy = [t.get('energy', 0.5) for t in tracks]
    names = [shorten_name(t['filename'], 20) for t in tracks]
    
    # Scatter plot with size based on energy
    sizes = [e * 300 + 50 for e in energy]
    scatter = ax.scatter(valence, arousal, s=sizes, c=energy, 
                        cmap='plasma', alpha=0.7, edgecolors='white')
    
    # Labels for each point
    for i, name in enumerate(names):
        ax.annotate(name, (valence[i], arousal[i]), 
                   fontsize=8, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')
    
    # Emotional quadrants
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.3)
    
    # Quadrant Labels
    ax.text(0.25, 0.85, '😤 Tense/Angry', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.75, 0.85, '😊 Happy/Excited', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.25, 0.15, '😢 Sad/Depressed', ha='center', fontsize=11, alpha=0.7)
    ax.text(0.75, 0.15, '😌 Calm/Relaxed', ha='center', fontsize=11, alpha=0.7)
    
    ax.set_xlabel('Valence (Negative ← → Positive)')
    ax.set_ylabel('Arousal (Calm ← → Excited)')
    ax.set_title('🎯 Emotional Map: Arousal vs Valence')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Colorbar for energy
    cbar = plt.colorbar(scatter)
    cbar.set_label('Energy')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_bpm_energy(tracks: list, save_path: str = None):
    """Scatter plot: BPM vs Energy with dominant mood as color."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bpm = [t.get('bpm', 0) for t in tracks]  # now normalized
    energy = [t.get('energy', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Determine dominant mood for each track
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
    
    # Scatter plot
    scatter = ax.scatter(bpm, energy, s=200, c=colors, alpha=0.8, edgecolors='white')
    
    # Labels
    for i, name in enumerate(names):
        ax.annotate(name, (bpm[i], energy[i]), 
                   fontsize=8, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')
    
    ax.set_xlabel('BPM')
    ax.set_ylabel('Energy')
    ax.set_title('⚡ BPM vs Energy (Color = Dominant Mood)')
    ax.grid(alpha=0.3)
    
    # Custom Legend
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
    """Radar chart to compare features across multiple tracks."""
    if track_indices is None:
        # Take the first 4 tracks if not specified
        track_indices = list(range(min(4, len(tracks))))
    
    categories = ['Energy', 'Arousal', 'Valence', 'Danceability', 
                  'Happy', 'Sad', 'Aggressive', 'Relaxed']
    N = len(categories)
    
    # Angles for the radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(track_indices)))
    
    for idx, track_idx in enumerate(track_indices):
        t = tracks[track_idx]
        values = [
            t.get('energy', 0),
            t.get('arousal', 0),
            t.get('valence', 0),
            min(1.0, t.get('danceability', 0)),  # Normalize danceability
            t.get('mood_happy', 0),
            t.get('mood_sad', 0),
            t.get('mood_aggressive', 0),
            t.get('mood_relaxed', 0)
        ]
        values += values[:1]  # Close the circle
        
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
    """Distribution of keys (major vs minor)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count major vs minor
    scales = [t.get('scale', 'unknown') for t in tracks]
    scale_counts = {'major': scales.count('major'), 'minor': scales.count('minor')}
    
    # Pie chart for major/minor
    colors_pie = ['#32CD32', '#4169E1']
    ax1.pie(scale_counts.values(), labels=['Major', 'Minor'], 
            autopct='%1.0f%%', colors=colors_pie, startangle=90,
            explode=(0.05, 0.05), shadow=True)
    ax1.set_title('🎹 Major vs Minor Distribution')
    
    # Bar chart for specific keys
    keys = [f"{t.get('key', '?')} {t.get('scale', '?')}" for t in tracks]
    key_counts = {}
    for k in keys:
        key_counts[k] = key_counts.get(k, 0) + 1
    
    sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
    key_names = [k[0] for k in sorted_keys]
    key_values = [k[1] for k in sorted_keys]
    
    colors_bar = ['#32CD32' if 'major' in k.lower() else '#4169E1' for k in key_names]
    ax2.barh(key_names, key_values, color=colors_bar)
    ax2.set_xlabel('Number of Tracks')
    ax2.set_title('🎼 Specific Keys')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_features_heatmap(tracks: list, save_path: str = None):
    """Heatmap of all features for all tracks."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(tracks) * 0.5)))
    
    features = ['energy', 'arousal', 'valence', 'danceability', 
                'mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed']
    
    data = []
    names = []
    for t in tracks:
        row = [min(1.0, t.get(f, 0)) for f in features]  # Normalize to 1
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
    
    # Add values in cells
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
    """Histograms of main features (including new metrics).

    Includes: bpm, energy, arousal, valence, danceability (legacy),
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

    series_data = [
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

    for ax, (title, values) in zip(axes, series_data):
        if values.size == 0:
            ax.set_title(f'{title} (no data)')
            ax.axis('off')
            continue

        ax.hist(values, bins=30, color='cyan', alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    # Hide any extra axes
    for ax in axes[len(series_data):]:
        ax.axis('off')

    fig.suptitle('📊 Feature Distributions (Histograms)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_valence_arousal_bpm(tracks: list, save_path: str = None):
    """3D Plot: Valence (X), Arousal (Y), BPM (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]  # now normalized
    energy = [t.get('energy', 0.5) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # 3D Scatter plot
    # Color based on Energy
    scatter = ax.scatter(valence, arousal, bpm, c=energy, cmap='plasma', s=100, alpha=0.8, edgecolors='white')
    
    # Labels
    for i, name in enumerate(names):
        ax.text(valence[i], arousal[i], bpm[i], name, fontsize=8)
    
    ax.set_xlabel('Valence (Positivity)')
    ax.set_ylabel('Arousal (Activation)')
    ax.set_zlabel('BPM (Tempo)')
    ax.set_title('🧊 3D: Valence vs Arousal vs BPM')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Energy')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_energy_danceability_bpm(tracks: list, save_path: str = None):
    """3D Plot: Energy (X), Danceability (Y), BPM (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    energy = [t.get('energy', 0.5) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]
    
    # Color based on Happy Mood
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
    """3D Plot in mood space: Happy (X), Sad (Y), Aggressive (Z)."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    happy = [t.get('mood_happy', 0) for t in tracks]
    sad = [t.get('mood_sad', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Color based on Relaxed (the fourth mood)
    relaxed = [t.get('mood_relaxed', 0) for t in tracks]
    
    scatter = ax.scatter(happy, sad, aggressive, c=relaxed, cmap='cool', s=100, alpha=0.8, edgecolors='white')
    
    for i, name in enumerate(names):
        ax.text(happy[i], sad[i], aggressive[i], name, fontsize=8)
    
    ax.set_xlabel('Happy')
    ax.set_ylabel('Sad')
    ax.set_zlabel('Aggressive')
    ax.set_title('🎭 3D Mood Space (Color = Relaxed)')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Relaxed Score')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_5d_multidimensional(tracks: list, save_path: str = None):
    """
    Multidimensional 5D Plot:
    X: Valence, Y: Arousal, Z: BPM
    Color: Aggressive, Size: Danceability
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Normalize danceability for point size (min 50, max 500)
    sizes = [d * 450 + 50 for d in danceability]
    
    # Scatter plot
    scatter = ax.scatter(valence, arousal, bpm, 
                        c=aggressive, cmap='Reds', 
                        s=sizes, alpha=0.8, edgecolors='white', linewidth=1)
    
    # Projection lines toward the base plane (to aid 3D perception)
    min_z = min(bpm) - 10
    for i in range(len(tracks)):
        ax.plot([valence[i], valence[i]], [arousal[i], arousal[i]], [min_z, bpm[i]], 
               color='gray', linestyle=':', alpha=0.3)
        ax.text(valence[i], arousal[i], bpm[i], names[i], fontsize=9)
    
    ax.set_xlabel('Valence (Positivity)')
    ax.set_ylabel('Arousal (Activation)')
    ax.set_zlabel('BPM (Tempo)')
    ax.set_title('🌌 5D Analysis: Valence/Arousal/BPM\n(Color=Aggressive, Size=Danceability)')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Aggressive Score (Color)')
    
    # Size Legend
    l1 = ax.scatter([], [], [], s=50, c='gray', edgecolors='none', label='Low Danceability')
    l2 = ax.scatter([], [], [], s=275, c='gray', edgecolors='none', label='Med Danceability')
    l3 = ax.scatter([], [], [], s=500, c='gray', edgecolors='none', label='High Danceability')
    
    ax.legend(handles=[l1, l2, l3], loc='upper left', title="Size", frameon=True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_5d_planes(tracks: list, save_path: str = None):
    """
    Visualizes 5 dimensions on 5 different planes (plots) to analyze correlations.
    Dimensions: Valence, Arousal, BPM, Danceability, Aggressive
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('🌌 5D Analysis: Multi-Plane View', fontsize=16)
    
    # Data
    valence = [t.get('valence', 0.5) for t in tracks]
    arousal = [t.get('arousal', 0.5) for t in tracks]
    bpm = [t.get('bpm', 0) for t in tracks]
    aggressive = [t.get('mood_aggressive', 0) for t in tracks]
    danceability = [t.get('danceability', 0) for t in tracks]
    names = [shorten_name(t['filename'], 15) for t in tracks]
    
    # Scatter plot helper
    def create_plane(ax_pos, x_data, y_data, x_label, y_label, title, color_data=None, cmap='viridis'):
        ax = fig.add_subplot(2, 3, ax_pos)
        c = color_data if color_data is not None else 'cyan'
        scatter = ax.scatter(x_data, y_data, c=c, cmap=cmap if color_data is not None else None, 
                           s=100, alpha=0.7, edgecolors='white')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Annotations
        for i, name in enumerate(names):
            ax.annotate(name, (x_data[i], y_data[i]), fontsize=7, alpha=0.8)
        return scatter

    # Plane 1: Emotional (Valence vs Arousal) - Color: Aggressive
    create_plane(1, valence, arousal, 'Valence', 'Arousal', '1. Emotional Plane', aggressive, 'Reds')
    
    # Plane 2: Rhythmic (BPM vs Danceability) - Color: Arousal
    create_plane(2, bpm, danceability, 'BPM', 'Danceability', '2. Rhythmic Plane', arousal, 'plasma')
    
    # Plane 3: Intensity (Aggressive vs Arousal) - Color: BPM
    create_plane(3, aggressive, arousal, 'Aggressive', 'Arousal', '3. Intensity Plane', bpm, 'cool')
    
    # Plane 4: Mood-Tempo (Valence vs BPM) - Color: Danceability
    create_plane(4, valence, bpm, 'Valence', 'BPM', '4. Mood-Tempo Plane', danceability, 'viridis')
    
    # Plane 5: Action (Aggressive vs Danceability) - Color: Valence
    create_plane(5, aggressive, danceability, 'Aggressive', 'Danceability', '5. Action Plane', valence, 'RdYlGn')
    
    # Slot 6: Explanation
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.5, 
             "Plane Legend:\n\n"
             "1. Emotional: Base emotions (Color: Aggressive)\n"
             "2. Rhythmic: Rhythmic structure (Color: Arousal)\n"
             "3. Intensity: Strength and activation (Color: BPM)\n"
             "4. Mood-Tempo: Happiness/Speed (Color: Dance)\n"
             "5. Action: Danceable aggression (Color: Valence)\n",
             fontsize=11, va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function to generate all plots."""
    # Cache / output paths (relative to repo)
    script_dir = Path(__file__).parent
    CACHE_PATH = str(script_dir / 'audio' / 'music_analysis_cache.json')
    OUTPUT_DIR = str(script_dir / 'plots')
    
    # Create output folder if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        data = load_analysis_data(CACHE_PATH)
        tracks = [t for t in data.get('tracks', []) if t.get('analyzed', False)]
        
        if not tracks:
            print("❌ No analyzed tracks found in cache!")
            return
        
        print(f"📊 Found {len(tracks)} analyzed tracks\n")
        
        # Generate Plots
        print("1️⃣  Generating Mood Comparison...")
        plot_mood_comparison(tracks, f"{OUTPUT_DIR}/mood_comparison.png")
        
        print("2️⃣  Generating Arousal vs Valence...")
        plot_arousal_valence(tracks, f"{OUTPUT_DIR}/arousal_valence.png")
        
        print("3️⃣  Generating BPM vs Energy...")
        plot_bpm_energy(tracks, f"{OUTPUT_DIR}/bpm_energy.png")
        
        print("4️⃣  Generating Radar Chart...")
        plot_radar_chart(tracks, save_path=f"{OUTPUT_DIR}/radar_chart.png")

        print("4️⃣b Generating Feature Histograms...")
        plot_feature_histograms(tracks, save_path=f"{OUTPUT_DIR}/feature_histograms.png")
        
        print("5️⃣  Generating Key Distribution...")
        plot_key_distribution(tracks, f"{OUTPUT_DIR}/key_distribution.png")
        
        print("6️⃣  Generating Features Heatmap...")
        plot_all_features_heatmap(tracks, f"{OUTPUT_DIR}/features_heatmap.png")
        
        print("7️⃣  Generating 3D Valence vs Arousal vs BPM...")
        plot_3d_valence_arousal_bpm(tracks, f"{OUTPUT_DIR}/3d_valence_arousal_bpm.png")
        
        print("8️⃣  Generating 3D Energy vs Danceability vs BPM...")
        plot_3d_energy_danceability_bpm(tracks, f"{OUTPUT_DIR}/3d_energy_danceability_bpm.png")
        
        print("9️⃣  Generating 3D Mood Space...")
        plot_3d_mood_space(tracks, f"{OUTPUT_DIR}/3d_mood_space.png")
        
        print("🔟 Generating 5D Multidimensional Plot...")
        plot_5d_multidimensional(tracks, f"{OUTPUT_DIR}/5d_multidimensional.png")
        
        print("1️⃣1️⃣ Generating 5D Multi-Plane View...")
        plot_5d_planes(tracks, f"{OUTPUT_DIR}/5d_planes.png")
        
        print(f"\n✅ Plots saved in: {OUTPUT_DIR}")
        
    except FileNotFoundError:
        print(f"❌ Cache not found: {CACHE_PATH}")
        print("   Run music_score.py first to generate the analysis.")


if __name__ == "__main__":
    main()