"""
Music Analysis Module using Essentia
Analyzes audio files for mood, energy, and BPM.
Saves results in JSON and reloads them if the folder has not changed.
Includes an OSC server to receive parameters and play the closest matching track.
"""

import json
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Essentia imports
try:
    import essentia
    import essentia.standard as es
    from essentia.standard import MonoLoader, RhythmExtractor2013
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("⚠️ Essentia not installed. Install with: pip install essentia-tensorflow")

# OSC imports
try:
    from pythonosc import dispatcher, osc_server
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("⚠️ python-osc not installed. Install with: pip install python-osc")

# NumPy import
import numpy as np


def smooth_distribution(data: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Makes data more uniform while partially preserving the original shape.
    
    Args:
        data: numpy array of values.
        intensity: float between 0.0 (original) and 1.0 (uniform).
    
    Returns:
        Array with smoothed distribution (maintains original order).
    """
    if len(data) == 0:
        return data
    
    # 1. Store original order
    sort_indices = np.argsort(data)
    sorted_data = data[sort_indices]
    
    # 2. Create uniform target (straight line from min to max)
    uniform_target = np.linspace(sorted_data.min(), sorted_data.max(), len(data))
    
    # 3. Perform weighted average
    new_sorted_data = (1 - intensity) * sorted_data + intensity * uniform_target
    
    # 4. Map values back to original order
    result = np.empty_like(data)
    result[sort_indices] = new_sorted_data
    
    return result


# Audio playback imports
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("⚠️ sounddevice not installed. Install with: pip install sounddevice")

class MusicAnalyzer:
    """Music analyzer with intelligent caching."""
    
    # Supported audio extensions
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    
    def __init__(
        self,
        audio_folder: str,
        cache_file: str = "music_analysis_cache.json",
        models_folder: Optional[str] = None
    ):
        """
        Initializes the analyzer.
        
        Args:
            audio_folder: Path to the folder containing audio files
            cache_file: Name of the JSON file to save results
            models_folder: Path to the folder with Essentia models (optional)
        """
        self.audio_folder = Path(audio_folder)
        self.cache_file = self.audio_folder / cache_file
        self.models_folder = Path(models_folder) if models_folder else None
        self.analysis_results: Dict[str, Any] = {}
        
        # Verify that the folder exists
        if not self.audio_folder.exists():
            raise FileNotFoundError(f"Audio folder not found: {self.audio_folder}")
    
    def _get_audio_files(self) -> List[Path]:
        """Gets the list of all audio files in the folder."""
        audio_files = []
        for ext in self.AUDIO_EXTENSIONS:
            audio_files.extend(self.audio_folder.glob(f"*{ext}"))
            audio_files.extend(self.audio_folder.glob(f"*{ext.upper()}"))
        return sorted(audio_files)
    
    def _compute_folder_hash(self) -> str:
        """
        Computes a hash of the folder based on:
        - Number of files
        - File names
        - File sizes
        """
        audio_files = self._get_audio_files()
        hash_data = []
        
        for f in audio_files:
            stat = f.stat()
            hash_data.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
        
        hash_string = "|".join(hash_data)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _should_reanalyze(self) -> bool:
        """
        Checks if it is necessary to re-analyze the folder.
        Returns True if:
        - The cache file does not exist
        - The folder hash has changed (files added/removed/modified)
        """
        if not self.cache_file.exists():
            print("📁 Cache not found, starting analysis...")
            return True
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            cached_hash = cached_data.get('folder_hash', '')
            current_hash = self._compute_folder_hash()
            
            if cached_hash != current_hash:
                print("🔄 Folder modified, starting re-analysis...")
                return True
            
            print("✅ Cache valid, loading existing data...")
            return False
            
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Cache corrupted, starting re-analysis...")
            return True
    
    def _analyze_bpm(self, audio: essentia.array) -> Dict[str, Any]:
        """Analyzes the BPM of an audio file."""
        try:
            rhythm_extractor = RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            
            return {
                'bpm': round(bpm, 2),
                'beats_confidence': round(float(beats_confidence), 3),
                'num_beats': len(beats)
            }
        except Exception as e:
            print(f"  ⚠️ BPM analysis error: {e}")
            return {'bpm': 0, 'beats_confidence': 0, 'num_beats': 0}
    
    def _analyze_mood_energy(self, audio: essentia.array, sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Analyzes mood and energy using Essentia algorithms.
        Extracts audio features and calculates mood based on spectral, rhythmic, and harmonic characteristics.
        """
        results = {
            'energy': 0.0,
            'mood_happy': 0.0,
            'mood_sad': 0.0,
            'mood_aggressive': 0.0,
            'mood_relaxed': 0.0,
            'danceability': 0.0,
            'instrumentalness': 0.0,
            'acousticness': 0.0,
            'electronicness': 0.0,
            'valence': 0.5,
            # Valence components (for debug/plot)
            'valence_mode': 0.5,
            'valence_brightness': 0.5,
            'valence_dance': 0.0,
            'valence_penalty_factor': 1.0,
            'valence_pre_penalty': 0.5,
            'arousal': 0.5,
            'key': '',
            'scale': '',  # 'major' or 'minor'
            'key_strength': 0.0
        }
        
        try:
            # === ENERGY ===
            rms = es.RMS()
            frame_size = 2048
            hop_size = 1024
            
            energies = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                energies.append(rms(frame))
            
            if energies:
                avg_energy = sum(energies) / len(energies)
                # Normalize energy (typically RMS is between 0 and 0.3 for normalized audio)
                results['energy'] = round(min(1.0, avg_energy * 5), 3)
            
            # === KEY ANALYSIS (FUNDAMENTAL FOR MOOD) ===
            # Major/minor mode is the most important factor for happy/sad
            # We represent the mode as a CONTINUOUS, linear value in [0..1]:
            # - mode_prob_major = 1.0 -> definitely major
            # - mode_prob_major = 0.0 -> definitely minor
            # - mode_prob_major = 0.5 -> neutral / uncertain
            mode_prob_major = 0.5
            is_minor = 0.5  # compatibility: used below as minor probability (0..1)
            try:
                key_extractor = es.KeyExtractor()
                key, scale, key_strength = key_extractor(audio)
                results['key'] = key
                results['scale'] = scale
                results['key_strength'] = round(float(key_strength), 3)

                # Minor mode = sad, Major mode = happy.
                # Make the mode effect LINEAR with confidence (key_strength):
                # - major: 0.5 + 0.5*strength
                # - minor: 0.5 - 0.5*strength
                key_strength01 = max(0.0, min(1.0, float(key_strength)))
                # "Gamma" weight (power-law) in [0,1] to compress low confidence values
                # and make the mode effect more neutral until key_strength is high.
                # w(s) = s^gamma
                gamma = 2.5
                strength_weight = float(max(0.0, min(1.0, key_strength01 ** gamma)))
                if scale == 'major':
                    mode_prob_major = 0.5 + 0.5 * strength_weight
                elif scale == 'minor':
                    mode_prob_major = 0.5 - 0.5 * strength_weight
                else:
                    mode_prob_major = 0.5
                is_minor = 1.0 - mode_prob_major
            except Exception as e:
                print(f"    ⚠️ KeyExtractor not available: {e}")
            
            # === SPECTRAL FLATNESS (flat = dark/sad) ===
            try:
                flatness_extractor = es.Flatness()
                spectrum_calc = es.Spectrum(size=2048)
                windowing = es.Windowing(type='hann')
                
                flatnesses = []
                for i in range(0, len(audio) - 2048, 4096):
                    frame = audio[i:i + 2048]
                    windowed = windowing(frame)
                    spec = spectrum_calc(windowed)
                    flatnesses.append(flatness_extractor(spec))
                
                avg_flatness = sum(flatnesses) / len(flatnesses) if flatnesses else 0.5
                # High flatness (near 1) = noise/dark, low = tonal/bright
            except:
                avg_flatness = 0.5
            
            # === DISSONANCE ===
            try:
                dissonance_calc = es.Dissonance()
                spectral_peaks = es.SpectralPeaks()
                spectrum_calc = es.Spectrum(size=4096)
                windowing = es.Windowing(type='hann')
                
                dissonances = []
                for i in range(0, len(audio) - 4096, 8192):
                    frame = audio[i:i + 4096]
                    windowed = windowing(frame)
                    spec = spectrum_calc(windowed)
                    freqs, mags = spectral_peaks(spec)
                    if len(freqs) > 1:
                        diss = dissonance_calc(freqs, mags)
                        dissonances.append(diss)
                
                avg_dissonance = sum(dissonances) / len(dissonances) if dissonances else 0.3
            except:
                avg_dissonance = 0.3
            
            # === SPECTRAL FEATURES ===
            spectrum = es.Spectrum(size=2048)
            spectral_centroid = es.Centroid(range=sample_rate / 2)
            spectral_rolloff = es.RollOff()
            spectral_flux = es.Flux()
            spectral_complexity = es.SpectralComplexity()
            
            centroids = []
            rolloffs = []
            fluxes = []
            complexities = []
            
            windowing = es.Windowing(type='hann')
            
            for i in range(0, len(audio) - 2048, 1024):
                frame = audio[i:i + 2048]
                windowed = windowing(frame)
                spec = spectrum(windowed)
                
                centroids.append(spectral_centroid(spec))
                rolloffs.append(spectral_rolloff(spec))
                fluxes.append(spectral_flux(spec))
                complexities.append(spectral_complexity(spec))
            
            avg_centroid = sum(centroids) / len(centroids) if centroids else 0
            avg_rolloff = sum(rolloffs) / len(rolloffs) if rolloffs else 0
            avg_flux = sum(fluxes) / len(fluxes) if fluxes else 0
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0
            
            # Normalize features (typical values)
            norm_centroid = min(1.0, avg_centroid / 5000)   # Centroid: 0-10000 Hz typical
            norm_flux = min(1.0, avg_flux * 2)              # Flux: 0-0.5 typical
            norm_complexity = min(1.0, avg_complexity / 50)  # Complexity: 0-100
            
            # === RHYTHMIC FEATURES ===
            onset_rate = es.OnsetRate()
            onsets, onset_rate_value = onset_rate(audio)
            norm_onset = min(1.0, onset_rate_value / 4)  # 0-4 onset/sec typical
            
            # Danceability
            try:
                danceability_extractor = es.Danceability()
                danceability_val, _ = danceability_extractor(audio)
                results['danceability'] = round(float(danceability_val), 3)
            except:
                results['danceability'] = round(norm_onset * 0.7 + results['energy'] * 0.3, 3)
            
            # === DYNAMIC COMPLEXITY ===
            try:
                dynamic_complexity = es.DynamicComplexity()
                dyn_complexity, loudness = dynamic_complexity(audio)
                norm_dynamics = min(1.0, dyn_complexity / 10)
            except:
                norm_dynamics = 0.5

            # === ACOUSTICNESS / ELECTRONICNESS (heuristic proxy) ===
            # Goal: a continuous dimension from "acoustic" -> "electronic".
            # Use only pre-calculated features (no external models):
            # - high flatness and high flux/complexity tend to be more "electronic/noisy"
            # - consonance (low dissonance) and tonality (low flatness) tend to be more "acoustic"
            acousticness = (
                (1 - avg_flatness) * 0.40 +
                (1 - avg_dissonance) * 0.20 +
                (1 - norm_flux) * 0.20 +
                (1 - norm_complexity) * 0.20
            )
            acousticness = max(0.0, min(1.0, float(acousticness)))
            electronicness = 1.0 - acousticness

            results['acousticness'] = round(acousticness, 3)
            results['electronicness'] = round(electronicness, 3)

            # === INSTRUMENTALNESS (heuristic proxy) ===
            # Rough estimate of "absence of voice" probability based on:
            # - stability/tonality (key_strength)
            # - low rhythmic activity (onset)
            # - low spectral variation (flux) and lower dissonance
            key_strength01 = max(0.0, min(1.0, float(results.get('key_strength', 0.0))))
            instrumentalness = (
                key_strength01 * 0.35 +
                (1 - norm_onset) * 0.25 +
                (1 - norm_flux) * 0.25 +
                (1 - avg_dissonance) * 0.15
            )
            instrumentalness = max(0.0, min(1.0, float(instrumentalness)))
            results['instrumentalness'] = round(instrumentalness, 3)
            
            # === MOOD CALCULATION (IMPROVED) ===
            # MUSICAL MODE is the most important factor for happy/sad!
            # - Minor mode → sad (even with high energy, e.g., Skyfall)
            # - Major mode → happy
            # - High dissonance → tension, drama
            # - High spectral flatness → dark
            
            # Arousal (activation): energy + brightness + onset rate (increased weight)
            arousal = (results['energy'] * 0.30 + norm_centroid * 0.25 + norm_onset * 0.45)
            results['arousal'] = round(arousal, 3)
            
            # Valence (positivity): mode contribution in continuous linear form.
            # mode_prob_major is already in [0..1] and incorporates confidence (key_strength).
            mode_contribution = max(0.0, min(1.0, float(mode_prob_major)))

            brightness_contribution = norm_centroid * 0.5 + (1 - avg_flatness) * 0.5

            # Valence: rebalanced weights to avoid overly binary separation.
            # (previous: mode 50%, brightness 30%, dance 20%)
            dance = min(1.0, results['danceability'])
            valence_pre_penalty = (
                mode_contribution * 0.35 +
                brightness_contribution * 0.40 +
                dance * 0.25
            )
            
            # Removes dissonance penalty: valence equals pre-penalty
            penalty_factor = 1.0
            valence = valence_pre_penalty

            # Save components (0..1) for visualization/debug
            results['valence_mode'] = round(max(0.0, min(1.0, float(mode_contribution))), 3)
            results['valence_brightness'] = round(max(0.0, min(1.0, float(brightness_contribution))), 3)
            results['valence_dance'] = round(max(0.0, min(1.0, float(dance))), 3)
            results['valence_penalty_factor'] = round(max(0.0, min(1.0, float(penalty_factor))), 3)
            results['valence_pre_penalty'] = round(max(0.0, min(1.0, float(valence_pre_penalty))), 3)
            results['valence'] = round(max(0.0, min(1.0, valence)), 3)
            
            # Mood Happy: high valence (requires major mode!)
            happy = valence * 0.7 + arousal * 0.3
            # Penalize if it's in minor even with high energy
            if is_minor > 0.5:
                happy = happy * 0.5
            results['mood_happy'] = round(min(1.0, max(0, happy)), 3)
            
            # Mood Sad: low valence, minor mode
            # Can be sad even with high energy (dramatic ballads)
            sad_base = (1 - valence) * 0.5 + is_minor * 0.5
            # Dissonance increases drama/sadness
            sad = sad_base + avg_dissonance * 0.2
            results['mood_sad'] = round(min(1.0, max(0, sad)), 3)
            
            # Mood Aggressive: high arousal + high flux + high dissonance
            aggressive = (arousal * 0.35 + norm_flux * 0.25 + 
                         norm_complexity * 0.20 + avg_dissonance * 0.20)
            results['mood_aggressive'] = round(min(1.0, aggressive), 3)
            
            # Mood Relaxed: low arousal + consonance + low complexity
            relaxed = ((1 - arousal) * 0.4 + 
                      (1 - norm_flux) * 0.25 + 
                      (1 - results['energy']) * 0.20 +
                      (1 - avg_dissonance) * 0.15)
            results['mood_relaxed'] = round(min(1.0, relaxed), 3)
            
        except Exception as e:
            print(f"  ⚠️ Mood/energy analysis error: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _analyze_single_file(self, audio_path: Path) -> Dict[str, Any]:
        """Analyzes a single audio file."""
        print(f"  🎵 Analyzing: {audio_path.name}")
        
        try:
            # Load audio (mono, 44100 Hz)
            loader_44k = MonoLoader(filename=str(audio_path), sampleRate=44100)
            audio_44k = loader_44k()
            
            # Duration
            duration = len(audio_44k) / 44100
            
            # BPM Analysis
            bpm_results = self._analyze_bpm(audio_44k)
            
            # Mood/Energy Analysis (uses 44100 Hz)
            mood_results = self._analyze_mood_energy(audio_44k, sample_rate=44100)
            
            return {
                'filename': audio_path.name,
                'path': str(audio_path),
                'duration_seconds': round(duration, 2),
                **bpm_results,
                **mood_results,
                'analyzed': True
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                'filename': audio_path.name,
                'path': str(audio_path),
                'error': str(e),
                'analyzed': False
            }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Performs the analysis of the folder.
        If the folder hasn't changed, loads results from cache.
        """
        if not ESSENTIA_AVAILABLE:
            raise ImportError("Essentia is not available. Install with: pip install essentia-tensorflow")
        
        # Check if we need to re-analyze
        if not self._should_reanalyze():
            return self.load_from_cache()
        
        # Perform analysis
        audio_files = self._get_audio_files()
        
        if not audio_files:
            print("⚠️ No audio files found in the folder!")
            return {'tracks': [], 'folder_hash': '', 'total_tracks': 0}
        
        print(f"\n🎼 Starting analysis of {len(audio_files)} audio files...\n")
        
        tracks = []
        for audio_file in audio_files:
            track_data = self._analyze_single_file(audio_file)
            tracks.append(track_data)
        
        # Apply smoothing (bending) to the valence_mode distribution
        # to get a smoother graph
        tracks = self._apply_mode_smoothing(tracks, intensity=1.0)
        
        # Normalize brightness to range [0, 1] based on global min/max
        tracks = self._normalize_brightness(tracks)
        
        # Normalize dance to range [0, 1] based on global min/max
        tracks = self._normalize_dance(tracks)
        
        # Normalize energy to range [0, 1] based on global min/max
        tracks = self._normalize_field(tracks, 'energy')

        # Normalize valence_pre_penalty to range [0, 1]
        tracks = self._normalize_field(tracks, 'valence_pre_penalty')
        
        # Normalize valence to range [0, 1]
        tracks = self._normalize_field(tracks, 'valence')

        # Normalize arousal to range [0, 1]
        tracks = self._normalize_field(tracks, 'arousal')
        
        # Normalize instrumentalness to range [0, 1]
        tracks = self._normalize_field(tracks, 'instrumentalness')
        
        # Normalize acousticness to range [0, 1]
        tracks = self._normalize_field(tracks, 'acousticness')
        
        # Normalize electronicness to range [0, 1]
        tracks = self._normalize_field(tracks, 'electronicness')

        # Normalize key_strength to range [0, 1]
        tracks = self._normalize_field(tracks, 'key_strength')

        # Normalize BPM creating `bpm_norm` field (keeps original `bpm`)
        tracks = self._normalize_bpm_field(tracks)
        
        # Prepare results
        self.analysis_results = {
            'folder_hash': self._compute_folder_hash(),
            'folder_path': str(self.audio_folder),
            'total_tracks': len(tracks),
            'tracks': tracks,
            'summary': self._compute_summary(tracks)
        }
        
        # Save cache
        self._save_cache()
        
        print(f"\n✅ Analysis complete! {len(tracks)} tracks analyzed.")
        return self.analysis_results
    
    def _apply_mode_smoothing(self, tracks: List[Dict], intensity: float = 0.5) -> List[Dict]:
        """
        Applies smoothing (bending) to the valence_mode distribution.
        Makes the distribution more uniform for a smoother graph.
        
        Args:
            tracks: List of dictionaries with track data
            intensity: float between 0.0 (original) and 1.0 (uniform)
        
        Returns:
            List of tracks with smoothed valence_mode
        """
        # Filter only analyzed tracks with valid valence_mode
        analyzed_indices = []
        mode_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_mode' in t:
                analyzed_indices.append(i)
                mode_values.append(t['valence_mode'])
        
        if len(mode_values) < 2:
            return tracks  # Not enough data for smoothing
        
        # Apply smoothing function
        mode_array = np.array(mode_values, dtype=float)
        smoothed_values = smooth_distribution(mode_array, intensity)
        
        # Update values in tracks and recalculate valence_pre_penalty and valence
        for idx, smoothed_val in zip(analyzed_indices, smoothed_values):
            t = tracks[idx]
            t['valence_mode_original'] = t['valence_mode']
            t['valence_mode'] = round(float(smoothed_val), 3)
            
            # Recalculate valence_pre_penalty with new smoothed valence_mode
            if 'valence_brightness' in t and 'valence_dance' in t:
                new_pre_penalty = (
                    t['valence_mode'] * 0.35 +
                    t['valence_brightness'] * 0.40 +
                    t['valence_dance'] * 0.25
                )
                t['valence_pre_penalty_original'] = t.get('valence_pre_penalty', new_pre_penalty)
                t['valence_pre_penalty'] = round(max(0.0, min(1.0, new_pre_penalty)), 3)
                
                # Recalculate valence (since penalty_factor = 1.0, it equals pre_penalty)
                t['valence_original'] = t.get('valence', new_pre_penalty)
                t['valence'] = t['valence_pre_penalty']
        
        print(f"  🔄 Smoothing applied to valence_mode (intensity: {intensity})")
        print(f"  🔄 Recalculated valence_pre_penalty and valence for {len(analyzed_indices)} tracks")
        return tracks
    
    def _normalize_brightness(self, tracks: List[Dict]) -> List[Dict]:
        """
        Normalizes valence_brightness to range [0, 1] using min-max scaling.
        Expands the distribution to use the full available range.
        
        Args:
            tracks: List of dictionaries with track data
        
        Returns:
            List of tracks with normalized valence_brightness
        """
        # Collect brightness values from analyzed tracks
        analyzed_indices = []
        brightness_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_brightness' in t:
                analyzed_indices.append(i)
                brightness_values.append(t['valence_brightness'])
        
        if len(brightness_values) < 2:
            return tracks  # Not enough data
        
        brightness_array = np.array(brightness_values, dtype=float)
        min_val = brightness_array.min()
        max_val = brightness_array.max()
        
        if max_val - min_val < 1e-6:
            return tracks  # Range too small
        
        # Normalize to range [0, 1]
        normalized = (brightness_array - min_val) / (max_val - min_val)
        
        # Update values in tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx]['valence_brightness_original'] = tracks[idx]['valence_brightness']
            tracks[idx]['valence_brightness'] = round(float(norm_val), 3)
        
        print(f"  📊 Normalized Brightness: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _normalize_dance(self, tracks: List[Dict]) -> List[Dict]:
        """
        Normalizes valence_dance to range [0, 1] using min-max scaling.
        Expands the distribution to use the full available range.
        
        Args:
            tracks: List of dictionaries with track data
        
        Returns:
            List of tracks with normalized valence_dance
        """
        # Collect dance values from analyzed tracks
        analyzed_indices = []
        dance_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_dance' in t:
                analyzed_indices.append(i)
                dance_values.append(t['valence_dance'])
        
        if len(dance_values) < 2:
            return tracks  # Not enough data
        
        dance_array = np.array(dance_values, dtype=float)
        min_val = dance_array.min()
        max_val = dance_array.max()
        
        if max_val - min_val < 1e-6:
            return tracks  # Range too small
        
        # Normalize to range [0, 1]
        normalized = (dance_array - min_val) / (max_val - min_val)
        
        # Update values in tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx]['valence_dance_original'] = tracks[idx]['valence_dance']
            tracks[idx]['valence_dance'] = round(float(norm_val), 3)
        
        print(f"  📊 Normalized Dance: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks

    def _normalize_bpm_field(self, tracks: List[Dict]) -> List[Dict]:
        """
        Normalizes the BPM field creating a new `bpm_norm` field in the range [0,1].
        Keeps original value in `bpm_original`.
        """
        analyzed_indices = []
        bpm_values = []

        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'bpm' in t and t.get('bpm', 0) > 0:
                analyzed_indices.append(i)
                bpm_values.append(float(t['bpm']))

        if len(bpm_values) < 2:
            return tracks

        bpm_array = np.array(bpm_values, dtype=float)
        min_val = bpm_array.min()
        max_val = bpm_array.max()

        if max_val - min_val < 1e-6:
            return tracks

        normalized = (bpm_array - min_val) / (max_val - min_val)

        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx]['bpm'] = round(float(norm_val), 3)

        print(f"  📊 Normalized BPM: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _normalize_field(self, tracks: List[Dict], field_name: str) -> List[Dict]:
        """
        Normalizes a generic field to range [0, 1] using min-max scaling.
        
        Args:
            tracks: List of dictionaries with track data
            field_name: Name of the field to normalize
        
        Returns:
            List of tracks with normalized field
        """
        # Collect values from specified field
        analyzed_indices = []
        field_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and field_name in t:
                analyzed_indices.append(i)
                field_values.append(t[field_name])
        
        if len(field_values) < 2:
            return tracks  # Not enough data
        
        field_array = np.array(field_values, dtype=float)
        min_val = field_array.min()
        max_val = field_array.max()
        
        if max_val - min_val < 1e-6:
            print(f"  ⚠️ {field_name}: range too small, skipping normalization")
            return tracks  # Range too small
        
        # Normalize to range [0, 1]
        normalized = (field_array - min_val) / (max_val - min_val)
        
        # Update values in tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx][f'{field_name}_original'] = tracks[idx][field_name]
            tracks[idx][field_name] = round(float(norm_val), 3)
        
        print(f"  📊 Normalized {field_name}: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _compute_summary(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Calculates summary statistics."""
        analyzed_tracks = [t for t in tracks if t.get('analyzed', False)]
        
        if not analyzed_tracks:
            return {}
        
        bpms = [t['bpm'] for t in analyzed_tracks if t.get('bpm', 0) > 0]
        energies = [t['energy'] for t in analyzed_tracks]
        return {
            'avg_bpm': round(sum(bpms) / len(bpms), 3) if bpms else 0,
            'min_bpm': min(bpms) if bpms else 0,
            'max_bpm': max(bpms) if bpms else 0,
            'avg_energy': round(sum(energies) / len(energies), 3) if energies else 0,
            'total_duration_minutes': round(sum(t['duration_seconds'] for t in analyzed_tracks) / 60, 2)
        }
    
    def _save_cache(self):
        """Saves results to JSON file."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Cache saved to: {self.cache_file}")
    
    def load_from_cache(self) -> Dict[str, Any]:
        """Loads results from cache."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            self.analysis_results = json.load(f)
        print(f"📂 Loaded {self.analysis_results.get('total_tracks', 0)} tracks from cache.")
        return self.analysis_results
    
    def get_tracks_by_bpm_range(self, min_bpm: float, max_bpm: float) -> List[Dict]:
        """Filter tracks by BPM range."""
        if not self.analysis_results:
            self.analyze()
        
        return [
            t for t in self.analysis_results.get('tracks', [])
            if min_bpm <= t.get('bpm', 0) <= max_bpm
        ]
    
    def get_tracks_by_energy(self, min_energy: float = 0.0, max_energy: float = 1.0) -> List[Dict]:
        """Filter tracks by energy level."""
        if not self.analysis_results:
            self.analyze()
        
        return [
            t for t in self.analysis_results.get('tracks', [])
            if min_energy <= t.get('energy', 0) <= max_energy
        ]
    
    def get_sorted_by(self, key: str, reverse: bool = False) -> List[Dict]:
        """Sort tracks by a specific key."""
        if not self.analysis_results:
            self.analyze()
        
        tracks = self.analysis_results.get('tracks', [])
        return sorted(tracks, key=lambda x: x.get(key, 0), reverse=reverse)


class MusicPlayer:
    """
    OSC Server that receives musical parameters and plays the closest matching track.
    
    Receives via OSC:
    - arousal, valence, bpm, instrumentalness, electronicness
    
    Calculates normalized Euclidean distance and plays the closest track
    for 30 seconds starting from the first onset.
    """
    
    def __init__(
        self, 
        analyzer: MusicAnalyzer,
        osc_ip: str = "0.0.0.0",
        osc_port: int = 9000,
        playback_duration: float = 30.0
    ):
        """
        Initializes the player.
        
        Args:
            analyzer: MusicAnalyzer instance with pre-analyzed data
            osc_ip: IP address for the OSC server
            osc_port: Port for the OSC server
            playback_duration: Playback duration in seconds
        """
        self.analyzer = analyzer
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        self.playback_duration = playback_duration
        
        # Ensure data is loaded
        if not self.analyzer.analysis_results:
            self.analyzer.analyze()
        
        self.tracks = [t for t in self.analyzer.analysis_results.get('tracks', []) 
                      if t.get('analyzed', False)]
        
        # Playback status
        self.is_playing = False
        self.current_stream = None
        self.stop_event = threading.Event()
        
        # BPM Normalization (for distance calculation)
        bpms = [t.get('bpm', 100) for t in self.tracks]
        self.bpm_min = min(bpms) if bpms else 60
        self.bpm_max = max(bpms) if bpms else 180
        
        # Pre-calculate feature matrix for fast searching
        self._feature_matrix = None
        self._build_feature_matrix()
        
        # OSC Server
        self.server = None
        self.server_thread = None
        
    def _normalize_bpm(self, bpm: float) -> float:
        """Normalizes BPM between 0 and 1."""
        if self.bpm_max == self.bpm_min:
            return 0.5
        return (bpm - self.bpm_min) / (self.bpm_max - self.bpm_min)
    
    def _track_to_feature_vector(self, track: Dict) -> np.ndarray:
        """Converts a track into a normalized feature vector."""
        instrumentalness = track.get('instrumentalness', None)
        if instrumentalness is None:
            # Fallback: if missing, use danceability as proxy
            instrumentalness = min(1.0, track.get('danceability', 0.0))

        electronicness = track.get('electronicness', None)
        if electronicness is None:
            # Fallback: if missing, use mood_aggressive as proxy
            electronicness = track.get('mood_aggressive', 0.0)

        # Prefer `bpm_norm` if present (generated by analyzer), otherwise normalize here
        bpm_val = track.get('bpm_norm', None)
        if bpm_val is None:
            bpm_feature = self._normalize_bpm(track.get('bpm', 100))
        else:
            bpm_feature = max(0.0, min(1.0, float(bpm_val)))

        return np.array([
            max(0.0, min(1.0, float(track.get('arousal', 0.5)))),
            max(0.0, min(1.0, float(track.get('valence', 0.5)))),
            bpm_feature,
            max(0.0, min(1.0, float(instrumentalness))),
            max(0.0, min(1.0, float(electronicness)))
        ])
    
    def _build_feature_matrix(self):
        """Pre-calculates the feature matrix for all tracks."""
        if not self.tracks:
            self._feature_matrix = None
            return
        
        self._feature_matrix = np.array([self._track_to_feature_vector(t) for t in self.tracks])
    
    def _calculate_distance_squared(self, track: Dict, target: Dict) -> float:
        """
        Calculates the SQUARED Euclidean distance between a track and target parameters.
        Sufficient for minimum distance comparisons.
        """
        track_vec = self._track_to_feature_vector(track)
        target_vec = np.array([
            target.get('arousal', 0.5),
            target.get('valence', 0.5),
            self._normalize_bpm(target.get('bpm', 100)),
            target.get('instrumentalness', 0.5),
            target.get('electronicness', 0.5)
        ])
        
        diff = track_vec - target_vec
        return np.dot(diff, diff)  # Squared distance, no root
    
    def _calculate_distance(self, track: Dict, target: Dict) -> float:
        """Wrapper for compatibility: returns distance (with root)."""
        return np.sqrt(self._calculate_distance_squared(track, target))
    
    def find_closest_track(self, arousal: float, valence: float, bpm: float,
                          instrumentalness: float, electronicness: float,
                          exclude_filenames: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Finds track with minimum distance using vectorized numpy.
        Uses squared Euclidean distance for efficiency.
        
        Args:
            exclude_filenames: List of filenames to exclude (to avoid repetition)
        """
        if not self.tracks:
            return None
        
        # Build exclusion set for fast lookup
        excluded = set(exclude_filenames) if exclude_filenames else set()
        
        # Build normalized target vector
        target_vec = np.array([
            arousal,
            valence,
            self._normalize_bpm(bpm),
            instrumentalness,
            electronicness
        ])
        
        # Vectorized calculation: squared distance for all tracks
        if self._feature_matrix is not None:
            diff = self._feature_matrix - target_vec
            distances_sq = np.sum(diff ** 2, axis=1)  # No root
            
            # Apply exclusion: set infinite distance for excluded tracks
            for i, track in enumerate(self.tracks):
                if track.get('filename', '') in excluded:
                    distances_sq[i] = float('inf')
            
            # Check if all tracks are excluded
            if np.all(np.isinf(distances_sq)):
                print("  ⚠️ All tracks excluded! Removing anti-repetition constraint.")
                diff = self._feature_matrix - target_vec
                distances_sq = np.sum(diff ** 2, axis=1)
            
            closest_idx = np.argmin(distances_sq)
            return self.tracks[closest_idx]
        
        # Fallback if matrix unavailable
        min_dist_sq = float('inf')
        closest = None
        
        for track in self.tracks:
            # Skip excluded tracks
            if track.get('filename', '') in excluded:
                continue
            
            track_vec = self._track_to_feature_vector(track)
            diff = track_vec - target_vec
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest = track
        
        # If all excluded, search without constraint
        if closest is None and excluded:
            print("  ⚠️ All tracks excluded! Removing anti-repetition constraint.")
            return self.find_closest_track(arousal, valence, bpm, instrumentalness, electronicness, None)
        
        return closest
    
    def _find_first_onset(self, audio_path: str) -> float:
        """Finds the timestamp of the first onset in the audio file."""
        try:
            loader = MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            
            onset_rate = es.OnsetRate()
            onsets, _ = onset_rate(audio)
            
            if len(onsets) > 0:
                return float(onsets[0])
            return 0.0
        except Exception as e:
            print(f"  ⚠️ Error finding onset: {e}")
            return 0.0
    
    def _play_audio(self, audio_path: str, start_time: float, duration: float):
        """Plays audio from specified time for indicated duration."""
        if not AUDIO_PLAYBACK_AVAILABLE:
            print("❌ Playback unavailable. Install sounddevice.")
            return
        
        try:
            # Load audio
            loader = MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            
            sample_rate = 44100
            start_sample = int(start_time * sample_rate)
            end_sample = start_sample + int(duration * sample_rate)
            
            # Ensure we don't exceed audio length
            end_sample = min(end_sample, len(audio))
            
            if start_sample >= len(audio):
                start_sample = 0
            
            audio_segment = audio[start_sample:end_sample]
            
            # Fade in/out to prevent clicks
            fade_samples = int(0.05 * sample_rate)  # 50ms fade
            if len(audio_segment) > fade_samples * 2:
                # Fade in
                fade_in = np.linspace(0, 1, fade_samples)
                audio_segment[:fade_samples] *= fade_in
                # Fade out
                fade_out = np.linspace(1, 0, fade_samples)
                audio_segment[-fade_samples:] *= fade_out
            
            print(f"  ▶️ Playing: {duration:.1f}s from {start_time:.2f}s")
            
            self.is_playing = True
            self.stop_event.clear()
            
            # Playback
            sd.play(audio_segment, sample_rate)
            
            # Wait for end of playback or stop
            elapsed = 0
            while elapsed < duration and not self.stop_event.is_set():
                time.sleep(0.1)
                elapsed += 0.1
            
            sd.stop()
            self.is_playing = False
            print("  ⏹️ Playback ended")
            
        except Exception as e:
            print(f"  ❌ Playback error: {e}")
            self.is_playing = False
    
    def stop_playback(self):
        """Stops current playback."""
        self.stop_event.set()
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        self.is_playing = False
    
    def play_closest(self, arousal: float, valence: float, bpm: float,
                    instrumentalness: float, electronicness: float):
        """Finds and plays the track closest to parameters."""
        # Stop any ongoing playback
        self.stop_playback()
        
        print(f"\n🎯 Received OSC parameters:")
        print(f"   Arousal={arousal:.2f}, Valence={valence:.2f}, BPM={bpm:.1f}")
        print(f"   Instrumentalness={instrumentalness:.2f}, Electronicness={electronicness:.2f}")
        
        # Find closest track
        closest = self.find_closest_track(arousal, valence, bpm, instrumentalness, electronicness)
        
        if closest is None:
            print("❌ No track found in dataset!")
            return
        
        print(f"\n🎵 Closest track: {closest['filename']}")
        print(f"   Distance: {self._calculate_distance(closest, {'arousal': arousal, 'valence': valence, 'bpm': bpm, 'instrumentalness': instrumentalness, 'electronicness': electronicness}):.3f}")
        print(f"   Track: arousal={closest.get('arousal', 0):.2f}, valence={closest.get('valence', 0):.2f}, bpm={closest.get('bpm', 0):.1f}")
        
        # Find first onset
        audio_path = closest.get('path', '')
        first_onset = self._find_first_onset(audio_path)
        print(f"   First onset: {first_onset:.2f}s")
        
        # Play in a separate thread
        play_thread = threading.Thread(
            target=self._play_audio,
            args=(audio_path, first_onset, self.playback_duration)
        )
        play_thread.daemon = True
        play_thread.start()
    
    def _osc_handler(self, address: str, *args):
        """Handler for OSC messages."""
        print(f"\n📨 OSC message received: {address}")
        
        if len(args) >= 5:
            arousal = float(args[0])
            valence = float(args[1])
            bpm = float(args[2])
            instrumentalness = float(args[3])
            electronicness = float(args[4])

            self.play_closest(arousal, valence, bpm, instrumentalness, electronicness)
        else:
            print(f"⚠️ Insufficient parameters. Received: {args}")
            print("   Expected format: arousal valence bpm instrumentalness electronicness")
    
    def start_server(self, osc_address: str = "/music/play"):
        """Starts the OSC server."""
        if not OSC_AVAILABLE:
            print("❌ python-osc unavailable!")
            return
        
        # Setup dispatcher
        disp = dispatcher.Dispatcher()
        disp.map(osc_address, self._osc_handler)
        disp.set_default_handler(self._osc_handler)
        
        # Create server
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.osc_ip, self.osc_port), disp
        )
        
        print("\n" + "="*60)
        print("🎧 MUSIC PLAYER OSC SERVER")
        print("="*60)
        print(f"   Address: {self.osc_ip}:{self.osc_port}")
        print(f"   OSC Route: {osc_address}")
        print(f"   Playback duration: {self.playback_duration}s")
        print(f"   Available tracks: {len(self.tracks)}")
        print("\n   OSC Message Format:")
        print(f"   {osc_address} [arousal] [valence] [bpm] [instrumentalness] [electronicness]")
        print("\n   Example (0-1 values except BPM):")
        print(f"   {osc_address} 0.7 0.3 120 0.8 0.5")
        print("\n   Press Ctrl+C to stop server")
        print("="*60 + "\n")
        
        # Start server
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped.")
            self.stop_playback()
    
    def start_server_async(self, osc_address: str = "/music/play"):
        """Starts the OSC server in a separate thread."""
        self.server_thread = threading.Thread(
            target=self.start_server,
            args=(osc_address,)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        return self.server_thread


# Usage Example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Analyzer & OSC Player")
    parser.add_argument('--server', action='store_true', 
                        help='Starts OSC server to receive parameters and play music')
    parser.add_argument('--port', type=int, default=9000,
                        help='OSC Port (default: 9000)')
    parser.add_argument('--address', type=str, default='/music/play',
                        help='OSC Address (default: /music/play)')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Playback duration in seconds (default: 30)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Perform analysis only, without starting server')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Removes cache file before analysis')
    parser.add_argument('--audio-folder', type=str,
                        help='Path to audio folder to analyze')
    
    args = parser.parse_args()
    
    # Configure audio folder path (dynamic relative to project)
    script_dir = Path(__file__).parent
    AUDIO_FOLDER = str((script_dir / "audio").resolve())
    
    # Override from CLI if provided
    if args.audio_folder:
        AUDIO_FOLDER = args.audio_folder
    
    # Create analyzer
    try:
        analyzer = MusicAnalyzer(
            audio_folder=AUDIO_FOLDER,
            cache_file="music_analysis_cache.json"
        )
        
        # Remove cache if requested
        if args.clear_cache:
            try:
                if analyzer.cache_file.exists():
                    analyzer.cache_file.unlink()
                    print(f"🗑️ Cache removed: {analyzer.cache_file}")
                else:
                    print("ℹ️ No cache to remove.")
            except Exception as e:
                print(f"⚠️ Error removing cache: {e}")
        
        # Run analysis (or load from cache if nothing changed)
        results = analyzer.analyze()
        
        # Show results
        print("\n" + "="*70)
        print("📊 ANALYSIS RESULTS")
        print("="*70)
        
        for track in results.get('tracks', []):
            if track.get('analyzed'):
                print(f"\n🎵 {track['filename']}")
                key_info = f"{track.get('key', '?')} {track.get('scale', '?')}" if track.get('key') else "N/A"
                print(f"   Key: {key_info} (strength: {track.get('key_strength', 0):.2f})")
                print(f"   BPM: {track['bpm']} (confidence: {track['beats_confidence']})")
                print(f"   Energy: {track['energy']:.2f} | Arousal: {track['arousal']:.2f} | Valence: {track['valence']:.2f}")
                print(f"   Danceability: {track['danceability']:.2f}")
                print(f"   Mood: 😊 Happy={track['mood_happy']:.2f} | 😢 Sad={track['mood_sad']:.2f} | 😤 Aggr={track['mood_aggressive']:.2f} | 😌 Relax={track['mood_relaxed']:.2f}")
                print(f"   Duration: {track['duration_seconds']}s")
        
        # Show summary
        if 'summary' in results:
            print("\n" + "="*70)
            print("📈 GENERAL STATISTICS")
            print("="*70)
            summary = results['summary']
            print(f"   Avg BPM: {summary.get('avg_bpm', 0)}")
            print(f"   BPM Range: {summary.get('min_bpm', 0)} - {summary.get('max_bpm', 0)}")
            print(f"   Avg Energy: {summary.get('avg_energy', 0):.2f}")
            print(f"   Total Duration: {summary.get('total_duration_minutes', 0):.1f} minutes")
        
        # Start OSC server if requested
        if args.server and not args.analyze_only:
            player = MusicPlayer(
                analyzer=analyzer,
                osc_port=args.port,
                playback_duration=args.duration
            )
            player.start_server(osc_address=args.address)
        elif not args.analyze_only:
            print("\n💡 To start the OSC server use: python music_score.py --server")
            print("   Options: --port 9000 --address /music/play --duration 30")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Create 'audio' folder in: {AUDIO_FOLDER}")