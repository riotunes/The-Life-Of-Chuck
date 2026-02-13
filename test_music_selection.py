#!/usr/bin/env python3
"""Test to verify that the music selection logic is working as expected. This will print out the selected tracks for each set of parameters and show how many times each track was selected."""

from musica.music_score import MusicAnalyzer, MusicPlayer
from collections import Counter

# inizializza MusicAnalyzer e MusicPlayer
analyzer = MusicAnalyzer('/Users/riccardotocci/Desktop/nuovo_Branch/Senza_nome/musica/audio')
results = analyzer.analyze()

player = MusicPlayer(analyzer=analyzer, osc_ip='0.0.0.0', osc_port=9001, playback_duration=1.0)

# parameters to test (age, arousal, valence, bpm, instrumentalness, electronicness)
params_list = [
    {'age': '24', 'arousal': 0.85, 'valence': 0.75, 'bpm': 145, 'instr': 0.2, 'elec': 0.4},
    {'age': '35', 'arousal': 0.7, 'valence': 0.65, 'bpm': 110, 'instr': 0.4, 'elec': 0.5},
    {'age': '46', 'arousal': 0.95, 'valence': 0.9, 'bpm': 128, 'instr': 0.1, 'elec': 0.8},
    {'age': '57', 'arousal': 0.5, 'valence': 0.6, 'bpm': 95, 'instr': 0.6, 'elec': 0.9},
    {'age': '68', 'arousal': 0.4, 'valence': 0.8, 'bpm': 80, 'instr': 0.8, 'elec': 1.0},
    {'age': '79', 'arousal': 0.3, 'valence': 0.5, 'bpm': 70, 'instr': 0.9, 'elec': 0.5},
    {'age': 'death', 'arousal': 0.1, 'valence': 0.4, 'bpm': 60, 'instr': 0.9, 'elec': 1.0}
]

# find closest track for each set of parameters and print the results, ensuring that the same track is not selected more than once across all ages
chosen_tracks = []
used_tracks = []  # keep track of most selected tracks to avoid repetition

print('🎵 most selected tracks:')
print('='*80)
for p in params_list:
    closest = player.find_closest_track(
        arousal=p['arousal'],
        valence=p['valence'],
        bpm=p['bpm'],
        instrumentalness=p['instr'],
        electronicness=p['elec'],
        exclude_filenames=used_tracks  # esclude previously selected tracks to ensure variety
    )
    filename = closest['filename']
    used_tracks.append(filename)  # add selected track to exclusion list for next iterations
    chosen_tracks.append(filename)
    print(f"Age {p['age']:>5}: {filename}")
    print(f"         Target:      arousal={p['arousal']:.2f}, valence={p['valence']:.2f}, bpm={p['bpm']:.0f}, instr={p['instr']:.2f}, elec={p['elec']:.2f}")
    print(f"         Track vals:  arousal={closest['arousal']:.2f}, valence={closest['valence']:.2f}, bpm={closest['bpm']:.0f}, instr={closest['instrumentalness']:.2f}, elec={closest['electronicness']:.2f}")
    print()

# show summary of how many times each track was selected
print('='*80)
print('📊 Count of selected tracks:')
for brano, count in Counter(chosen_tracks).most_common():
    print(f"  {count}x - {brano}")
