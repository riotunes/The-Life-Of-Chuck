#!/usr/bin/env python3
"""Test per vedere quali brani vengono selezionati per ogni età."""

from musica.music_score import MusicAnalyzer, MusicPlayer
from collections import Counter

# Inizializza
analyzer = MusicAnalyzer('/Users/riccardotocci/Desktop/nuovo_Branch/Senza_nome/musica/audio')
results = analyzer.analyze()

player = MusicPlayer(analyzer=analyzer, osc_ip='0.0.0.0', osc_port=9001, playback_duration=1.0)

# Parametri dai file music_params
params_list = [
    {'age': '24', 'arousal': 0.85, 'valence': 0.75, 'bpm': 145, 'instr': 0.2, 'elec': 0.4},
    {'age': '35', 'arousal': 0.7, 'valence': 0.65, 'bpm': 110, 'instr': 0.4, 'elec': 0.5},
    {'age': '46', 'arousal': 0.95, 'valence': 0.9, 'bpm': 128, 'instr': 0.1, 'elec': 0.8},
    {'age': '57', 'arousal': 0.5, 'valence': 0.6, 'bpm': 95, 'instr': 0.6, 'elec': 0.9},
    {'age': '68', 'arousal': 0.4, 'valence': 0.8, 'bpm': 80, 'instr': 0.8, 'elec': 1.0},
    {'age': '79', 'arousal': 0.3, 'valence': 0.5, 'bpm': 70, 'instr': 0.9, 'elec': 0.5},
    {'age': 'death', 'arousal': 0.1, 'valence': 0.4, 'bpm': 60, 'instr': 0.9, 'elec': 1.0}
]

# Trova il brano più vicino per ogni set di parametri (CON anti-ripetizione)
brani_scelti = []
used_tracks = []  # Traccia i brani già usati

print('🎵 Brani selezionati per ogni età (con vincolo anti-ripetizione):')
print('='*80)
for p in params_list:
    closest = player.find_closest_track(
        arousal=p['arousal'],
        valence=p['valence'],
        bpm=p['bpm'],
        instrumentalness=p['instr'],
        electronicness=p['elec'],
        exclude_filenames=used_tracks  # Escludi brani già usati
    )
    filename = closest['filename']
    used_tracks.append(filename)  # Aggiungi ai brani usati
    brani_scelti.append(filename)
    print(f"Age {p['age']:>5}: {filename}")
    print(f"         Target:      arousal={p['arousal']:.2f}, valence={p['valence']:.2f}, bpm={p['bpm']:.0f}, instr={p['instr']:.2f}, elec={p['elec']:.2f}")
    print(f"         Track vals:  arousal={closest['arousal']:.2f}, valence={closest['valence']:.2f}, bpm={closest['bpm']:.0f}, instr={closest['instrumentalness']:.2f}, elec={closest['electronicness']:.2f}")
    print()

# Mostra conteggio dei brani
print('='*80)
print('📊 Conteggio brani:')
for brano, count in Counter(brani_scelti).most_common():
    print(f"  {count}x - {brano}")
