"""
Music Analysis Module using Essentia
Analizza file audio per mood, energia e BPM.
Salva i risultati in JSON e li ricarica se la cartella non è cambiata.
Include un server OSC per ricevere parametri e riprodurre il brano più vicino.
"""

import os
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
    from essentia.standard import MonoLoader, RhythmExtractor2013, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("⚠️ Essentia non installato. Installa con: pip install essentia-tensorflow")

# OSC imports
try:
    from pythonosc import dispatcher, osc_server
    from pythonosc.udp_client import SimpleUDPClient
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("⚠️ python-osc non installato. Installa con: pip install python-osc")

# NumPy import
import numpy as np


def ammorbidisci_distribuzione(dati: np.ndarray, intensita: float = 0.5) -> np.ndarray:
    """
    Rende i dati più uniformi preservando parzialmente la forma originale.
    
    Args:
        dati: array numpy dei valori.
        intensita: float tra 0.0 (originale) e 1.0 (uniforme).
    
    Returns:
        Array con distribuzione smussata (mantiene l'ordine originale).
    """
    if len(dati) == 0:
        return dati
    
    # 1. Memorizza l'ordine originale
    indici_ordinamento = np.argsort(dati)
    dati_ordinati = dati[indici_ordinamento]
    
    # 2. Crea il target uniforme (linea retta dal min al max)
    target_uniforme = np.linspace(dati_ordinati.min(), dati_ordinati.max(), len(dati))
    
    # 3. Fai la media pesata
    nuovi_dati_ordinati = (1 - intensita) * dati_ordinati + intensita * target_uniforme
    
    # 4. Riporta i valori all'ordine originale
    risultato = np.empty_like(dati)
    risultato[indici_ordinamento] = nuovi_dati_ordinati
    
    return risultato


# Audio playback imports
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("⚠️ sounddevice non installato. Installa con: pip install sounddevice")


class MusicAnalyzer:
    """Analizzatore musicale con caching intelligente."""
    
    # Estensioni audio supportate
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}
    
    def __init__(
        self,
        audio_folder: str,
        cache_file: str = "music_analysis_cache.json",
        models_folder: Optional[str] = None
    ):
        """
        Inizializza l'analizzatore.
        
        Args:
            audio_folder: Percorso alla cartella con i file audio
            cache_file: Nome del file JSON per salvare i risultati
            models_folder: Percorso alla cartella con i modelli Essentia (opzionale)
        """
        self.audio_folder = Path(audio_folder)
        self.cache_file = self.audio_folder / cache_file
        self.models_folder = Path(models_folder) if models_folder else None
        self.analysis_results: Dict[str, Any] = {}
        
        # Verifica che la cartella esista
        if not self.audio_folder.exists():
            raise FileNotFoundError(f"Cartella audio non trovata: {self.audio_folder}")
    
    def _get_audio_files(self) -> List[Path]:
        """Ottiene la lista di tutti i file audio nella cartella."""
        audio_files = []
        for ext in self.AUDIO_EXTENSIONS:
            audio_files.extend(self.audio_folder.glob(f"*{ext}"))
            audio_files.extend(self.audio_folder.glob(f"*{ext.upper()}"))
        return sorted(audio_files)
    
    def _compute_folder_hash(self) -> str:
        """
        Calcola un hash della cartella basato su:
        - Numero di file
        - Nomi dei file
        - Dimensioni dei file
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
        Controlla se è necessario rianalizzare la cartella.
        Ritorna True se:
        - Il file cache non esiste
        - L'hash della cartella è cambiato (file aggiunti/rimossi/modificati)
        """
        if not self.cache_file.exists():
            print("📁 Cache non trovata, avvio analisi...")
            return True
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            cached_hash = cached_data.get('folder_hash', '')
            current_hash = self._compute_folder_hash()
            
            if cached_hash != current_hash:
                print("🔄 Cartella modificata, avvio ri-analisi...")
                return True
            
            print("✅ Cache valida, caricamento dati esistenti...")
            return False
            
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Cache corrotta, avvio ri-analisi...")
            return True
    
    def _analyze_bpm(self, audio: essentia.array) -> Dict[str, Any]:
        """Analizza il BPM di un file audio."""
        try:
            rhythm_extractor = RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            
            return {
                'bpm': round(bpm, 2),
                'beats_confidence': round(float(beats_confidence), 3),
                'num_beats': len(beats)
            }
        except Exception as e:
            print(f"  ⚠️ Errore analisi BPM: {e}")
            return {'bpm': 0, 'beats_confidence': 0, 'num_beats': 0}
    
    def _analyze_mood_energy(self, audio: essentia.array, sample_rate: int = 44100) -> Dict[str, Any]:
        """
        Analizza mood ed energia usando gli algoritmi Essentia.
        Estrae features audio e calcola mood basandosi su caratteristiche spettrali, ritmiche e armoniche.
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
            # Valence components (per debug/plot)
            'valence_mode': 0.5,
            'valence_brightness': 0.5,
            'valence_dance': 0.0,
            'valence_penalty_factor': 1.0,
            'valence_pre_penalty': 0.5,
            'arousal': 0.5,
            'key': '',
            'scale': '',  # 'major' o 'minor'
            'key_strength': 0.0
        }
        
        try:
            # === ENERGIA ===
            rms = es.RMS()
            frame_size = 2048
            hop_size = 1024
            
            energies = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                energies.append(rms(frame))
            
            if energies:
                avg_energy = sum(energies) / len(energies)
                max_energy = max(energies)
                # Normalizza energia (tipicamente RMS è tra 0 e 0.3 per audio normalizzato)
                results['energy'] = round(min(1.0, avg_energy * 5), 3)
            
            # === ANALISI TONALITÀ (FONDAMENTALE PER MOOD) ===
            # Il modo maggiore/minore è il fattore più importante per happy/sad
            # Rappresentiamo il modo come valore CONTINUO, lineare, in [0..1]:
            # - mode_prob_major = 1.0 -> sicuramente major
            # - mode_prob_major = 0.0 -> sicuramente minor
            # - mode_prob_major = 0.5 -> neutro / incerto
            mode_prob_major = 0.5
            is_minor = 0.5  # compatibilità: usato sotto come probabilità di minor (0..1)
            try:
                key_extractor = es.KeyExtractor()
                key, scale, key_strength = key_extractor(audio)
                results['key'] = key
                results['scale'] = scale
                results['key_strength'] = round(float(key_strength), 3)

                # Minor mode = triste, Major mode = felice.
                # Rendiamo l'effetto del modo LINEARE con la confidenza (key_strength):
                # - major: 0.5 + 0.5*strength
                # - minor: 0.5 - 0.5*strength
                key_strength01 = max(0.0, min(1.0, float(key_strength)))
                # Peso "gamma" (power-law) in [0,1] per comprimere valori bassi di confidenza
                # e rendere l'effetto del modo più neutro finché key_strength non è alto.
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
                print(f"    ⚠️ KeyExtractor non disponibile: {e}")
            
            # === SPECTRAL FLATNESS (piatto = cupo/triste) ===
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
                # Flatness alto (vicino a 1) = rumore/cupo, basso = tonale/brillante
            except:
                avg_flatness = 0.5
            
            # === DISSONANZA ===
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
            
            # === FEATURES SPETTRALI ===
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
            
            # Normalizza features (valori tipici)
            norm_centroid = min(1.0, avg_centroid / 5000)  # Centroide: 0-10000 Hz tipico
            norm_rolloff = min(1.0, avg_rolloff / 10000)   # Rolloff: 0-20000 Hz
            norm_flux = min(1.0, avg_flux * 2)             # Flux: 0-0.5 tipico
            norm_complexity = min(1.0, avg_complexity / 50) # Complexity: 0-100
            
            # === FEATURES RITMICHE ===
            onset_rate = es.OnsetRate()
            onsets, onset_rate_value = onset_rate(audio)
            norm_onset = min(1.0, onset_rate_value / 4)  # 0-4 onset/sec tipico
            
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

            # === ACOUSTICNESS / ELECTRONICNESS (proxy euristico) ===
            # Obiettivo: una dimensione continua che vada da "acustico" -> "elettronico".
            # Usiamo solo feature già calcolate (no modelli esterni):
            # - flatness alto e flux/complexity alti tendono ad essere più "elettronici/noisy"
            # - consonanza (bassa dissonanza) e tonality (bassa flatness) tendono ad essere più "acustici"
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

            # === INSTRUMENTALNESS (proxy euristico) ===
            # Stima grezza della probabilità di "assenza di voce" basata su:
            # - stabilità/tonalità (key_strength)
            # - bassa attività ritmica (onset)
            # - bassa variazione spettrale (flux) e minore dissonanza
            key_strength01 = max(0.0, min(1.0, float(results.get('key_strength', 0.0))))
            instrumentalness = (
                key_strength01 * 0.35 +
                (1 - norm_onset) * 0.25 +
                (1 - norm_flux) * 0.25 +
                (1 - avg_dissonance) * 0.15
            )
            instrumentalness = max(0.0, min(1.0, float(instrumentalness)))
            results['instrumentalness'] = round(instrumentalness, 3)
            
            # === CALCOLO MOOD (MIGLIORATO) ===
            # Il MODO MUSICALE è il fattore più importante per happy/sad!
            # - Minor mode → triste (anche con alta energia, es. Skyfall)
            # - Major mode → felice
            # - Alta dissonanza → tensione, drammaticità
            # - Spectral flatness alto → cupo
            
            # Arousal (attivazione): energia + brightness + onset rate (peso aumentato)
            arousal = (results['energy'] * 0.30 + norm_centroid * 0.25 + norm_onset * 0.45)
            results['arousal'] = round(arousal, 3)
            
            # Valence (positività): contributo del modo in forma lineare continua.
            # mode_prob_major è già in [0..1] ed incorpora la confidenza (key_strength).
            mode_contribution = max(0.0, min(1.0, float(mode_prob_major)))

            brightness_contribution = norm_centroid * 0.5 + (1 - avg_flatness) * 0.5

            # Valence: pesi ribalanciati per evitare una separazione troppo binaria.
            # (prima: modo 50%, brightness 30%, dance 20%)
            dance = min(1.0, results['danceability'])
            valence_pre_penalty = (
                mode_contribution * 0.35 +
                brightness_contribution * 0.40 +
                dance * 0.25
            )
            
            # Rimuove la penalità di dissonanza: valence uguale al pre-penalty
            penalty_factor = 1.0
            valence = valence_pre_penalty

            # Salva componenti (0..1) per visualizzazioni/debug
            results['valence_mode'] = round(max(0.0, min(1.0, float(mode_contribution))), 3)
            results['valence_brightness'] = round(max(0.0, min(1.0, float(brightness_contribution))), 3)
            results['valence_dance'] = round(max(0.0, min(1.0, float(dance))), 3)
            results['valence_penalty_factor'] = round(max(0.0, min(1.0, float(penalty_factor))), 3)
            results['valence_pre_penalty'] = round(max(0.0, min(1.0, float(valence_pre_penalty))), 3)
            results['valence'] = round(max(0.0, min(1.0, valence)), 3)
            
            # Mood Happy: alta valence (richiede modo maggiore!)
            happy = valence * 0.7 + arousal * 0.3
            # Penalizza se è in minore anche con alta energia
            if is_minor > 0.5:
                happy = happy * 0.5
            results['mood_happy'] = round(min(1.0, max(0, happy)), 3)
            
            # Mood Sad: bassa valence, modo minore
            # Può essere triste anche con alta energia (ballate drammatiche)
            sad_base = (1 - valence) * 0.5 + is_minor * 0.5
            # La dissonanza aumenta la drammaticità/tristezza
            sad = sad_base + avg_dissonance * 0.2
            results['mood_sad'] = round(min(1.0, max(0, sad)), 3)
            
            # Mood Aggressive: alto arousal + alto flux + alta dissonanza
            aggressive = (arousal * 0.35 + norm_flux * 0.25 + 
                         norm_complexity * 0.20 + avg_dissonance * 0.20)
            results['mood_aggressive'] = round(min(1.0, aggressive), 3)
            
            # Mood Relaxed: basso arousal + consonanza + bassa complessità
            relaxed = ((1 - arousal) * 0.4 + 
                      (1 - norm_flux) * 0.25 + 
                      (1 - results['energy']) * 0.20 +
                      (1 - avg_dissonance) * 0.15)
            results['mood_relaxed'] = round(min(1.0, relaxed), 3)
            
        except Exception as e:
            print(f"  ⚠️ Errore analisi mood/energy: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _analyze_single_file(self, audio_path: Path) -> Dict[str, Any]:
        """Analizza un singolo file audio."""
        print(f"  🎵 Analizzando: {audio_path.name}")
        
        try:
            # Carica audio (mono, 44100 Hz)
            loader_44k = MonoLoader(filename=str(audio_path), sampleRate=44100)
            audio_44k = loader_44k()
            
            # Durata
            duration = len(audio_44k) / 44100
            
            # Analisi BPM
            bpm_results = self._analyze_bpm(audio_44k)
            
            # Analisi Mood/Energy (usa 44100 Hz)
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
            print(f"  ❌ Errore: {e}")
            return {
                'filename': audio_path.name,
                'path': str(audio_path),
                'error': str(e),
                'analyzed': False
            }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Esegue l'analisi della cartella.
        Se la cartella non è cambiata, carica i risultati dalla cache.
        """
        if not ESSENTIA_AVAILABLE:
            raise ImportError("Essentia non è disponibile. Installalo con: pip install essentia-tensorflow")
        
        # Check se dobbiamo rianalizzare
        if not self._should_reanalyze():
            return self.load_from_cache()
        
        # Esegui analisi
        audio_files = self._get_audio_files()
        
        if not audio_files:
            print("⚠️ Nessun file audio trovato nella cartella!")
            return {'tracks': [], 'folder_hash': '', 'total_tracks': 0}
        
        print(f"\n🎼 Avvio analisi di {len(audio_files)} file audio...\n")
        
        tracks = []
        for audio_file in audio_files:
            track_data = self._analyze_single_file(audio_file)
            tracks.append(track_data)
        
        # Applica smoothing (bending) alla distribuzione di valence_mode
        # per ottenere un grafico più smussato
        tracks = self._apply_mode_smoothing(tracks, intensita=0.7)
        
        # Normalizza brightness al range [0, 1] basandosi su min/max globale
        tracks = self._normalize_brightness(tracks)
        
        # Normalizza dance al range [0, 1] basandosi su min/max globale
        tracks = self._normalize_dance(tracks)
        
        # Normalizza energy al range [0, 1] basandosi su min/max globale
        tracks = self._normalize_field(tracks, 'energy')

        # Normalizza valence_pre_penalty al range [0, 1]
        tracks = self._normalize_field(tracks, 'valence_pre_penalty')
        
        # Normalizza valence al range [0, 1]
        tracks = self._normalize_field(tracks, 'valence')
        
        # Normalizza instrumentalness al range [0, 1]
        tracks = self._normalize_field(tracks, 'instrumentalness')
        
        # Normalizza acousticness al range [0, 1]
        tracks = self._normalize_field(tracks, 'acousticness')
        
        # Normalizza electronicness al range [0, 1]
        tracks = self._normalize_field(tracks, 'electronicness')

        # Normalizza key_strength al range [0, 1]
        tracks = self._normalize_field(tracks, 'key_strength')
        
        # Prepara risultati
        self.analysis_results = {
            'folder_hash': self._compute_folder_hash(),
            'folder_path': str(self.audio_folder),
            'total_tracks': len(tracks),
            'tracks': tracks,
            'summary': self._compute_summary(tracks)
        }
        
        # Salva cache
        self._save_cache()
        
        print(f"\n✅ Analisi completata! {len(tracks)} brani analizzati.")
        return self.analysis_results
    
    def _apply_mode_smoothing(self, tracks: List[Dict], intensita: float = 0.5) -> List[Dict]:
        """
        Applica smoothing (bending) alla distribuzione di valence_mode.
        Rende la distribuzione più uniforme per un grafico più smussato.
        
        Args:
            tracks: Lista di dizionari con i dati dei brani
            intensita: float tra 0.0 (originale) e 1.0 (uniforme)
        
        Returns:
            Lista di tracks con valence_mode smussato
        """
        # Filtra solo i brani analizzati con valence_mode valido
        analyzed_indices = []
        mode_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_mode' in t:
                analyzed_indices.append(i)
                mode_values.append(t['valence_mode'])
        
        if len(mode_values) < 2:
            return tracks  # Non abbastanza dati per lo smoothing
        
        # Applica la funzione di smoothing
        mode_array = np.array(mode_values, dtype=float)
        smoothed_values = ammorbidisci_distribuzione(mode_array, intensita)
        
        # Aggiorna i valori nei tracks
        for idx, smoothed_val in zip(analyzed_indices, smoothed_values):
            tracks[idx]['valence_mode_original'] = tracks[idx]['valence_mode']
            tracks[idx]['valence_mode'] = round(float(smoothed_val), 3)
        
        print(f"  🔄 Smoothing applicato a valence_mode (intensità: {intensita})")
        return tracks
    
    def _normalize_brightness(self, tracks: List[Dict]) -> List[Dict]:
        """
        Normalizza valence_brightness al range [0, 1] usando min-max scaling.
        Questo espande la distribuzione per usare tutto il range disponibile.
        
        Args:
            tracks: Lista di dizionari con i dati dei brani
        
        Returns:
            Lista di tracks con valence_brightness normalizzato
        """
        # Raccoglie i valori di brightness dai brani analizzati
        analyzed_indices = []
        brightness_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_brightness' in t:
                analyzed_indices.append(i)
                brightness_values.append(t['valence_brightness'])
        
        if len(brightness_values) < 2:
            return tracks  # Non abbastanza dati
        
        brightness_array = np.array(brightness_values, dtype=float)
        min_val = brightness_array.min()
        max_val = brightness_array.max()
        
        if max_val - min_val < 1e-6:
            return tracks  # Range troppo piccolo
        
        # Normalizza al range [0, 1]
        normalized = (brightness_array - min_val) / (max_val - min_val)
        
        # Aggiorna i valori nei tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx]['valence_brightness_original'] = tracks[idx]['valence_brightness']
            tracks[idx]['valence_brightness'] = round(float(norm_val), 3)
        
        print(f"  📊 Brightness normalizzato: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _normalize_dance(self, tracks: List[Dict]) -> List[Dict]:
        """
        Normalizza valence_dance al range [0, 1] usando min-max scaling.
        Questo espande la distribuzione per usare tutto il range disponibile.
        
        Args:
            tracks: Lista di dizionari con i dati dei brani
        
        Returns:
            Lista di tracks con valence_dance normalizzato
        """
        # Raccoglie i valori di dance dai brani analizzati
        analyzed_indices = []
        dance_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and 'valence_dance' in t:
                analyzed_indices.append(i)
                dance_values.append(t['valence_dance'])
        
        if len(dance_values) < 2:
            return tracks  # Non abbastanza dati
        
        dance_array = np.array(dance_values, dtype=float)
        min_val = dance_array.min()
        max_val = dance_array.max()
        
        if max_val - min_val < 1e-6:
            return tracks  # Range troppo piccolo
        
        # Normalizza al range [0, 1]
        normalized = (dance_array - min_val) / (max_val - min_val)
        
        # Aggiorna i valori nei tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx]['valence_dance_original'] = tracks[idx]['valence_dance']
            tracks[idx]['valence_dance'] = round(float(norm_val), 3)
        
        print(f"  📊 Dance normalizzato: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _normalize_field(self, tracks: List[Dict], field_name: str) -> List[Dict]:
        """
        Normalizza un campo generico al range [0, 1] usando min-max scaling.
        
        Args:
            tracks: Lista di dizionari con i dati dei brani
            field_name: Nome del campo da normalizzare
        
        Returns:
            Lista di tracks con il campo normalizzato
        """
        # Raccoglie i valori dal campo specificato
        analyzed_indices = []
        field_values = []
        
        for i, t in enumerate(tracks):
            if t.get('analyzed', False) and field_name in t:
                analyzed_indices.append(i)
                field_values.append(t[field_name])
        
        if len(field_values) < 2:
            return tracks  # Non abbastanza dati
        
        field_array = np.array(field_values, dtype=float)
        min_val = field_array.min()
        max_val = field_array.max()
        
        if max_val - min_val < 1e-6:
            print(f"  ⚠️ {field_name}: range troppo piccolo, skip normalizzazione")
            return tracks  # Range troppo piccolo
        
        # Normalizza al range [0, 1]
        normalized = (field_array - min_val) / (max_val - min_val)
        
        # Aggiorna i valori nei tracks
        for idx, norm_val in zip(analyzed_indices, normalized):
            tracks[idx][f'{field_name}_original'] = tracks[idx][field_name]
            tracks[idx][field_name] = round(float(norm_val), 3)
        
        print(f"  📊 {field_name} normalizzato: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        return tracks
    
    def _compute_summary(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Calcola statistiche riassuntive."""
        analyzed_tracks = [t for t in tracks if t.get('analyzed', False)]
        
        if not analyzed_tracks:
            return {}
        
        bpms = [t['bpm'] for t in analyzed_tracks if t.get('bpm', 0) > 0]
        energies = [t['energy'] for t in analyzed_tracks]
        
        return {
            'avg_bpm': round(sum(bpms) / len(bpms), 2) if bpms else 0,
            'min_bpm': min(bpms) if bpms else 0,
            'max_bpm': max(bpms) if bpms else 0,
            'avg_energy': round(sum(energies) / len(energies), 3) if energies else 0,
            'total_duration_minutes': round(sum(t['duration_seconds'] for t in analyzed_tracks) / 60, 2)
        }
    
    def _save_cache(self):
        """Salva i risultati nel file JSON."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Cache salvata in: {self.cache_file}")
    
    def load_from_cache(self) -> Dict[str, Any]:
        """Carica i risultati dalla cache."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            self.analysis_results = json.load(f)
        print(f"📂 Caricati {self.analysis_results.get('total_tracks', 0)} brani dalla cache.")
        return self.analysis_results
    
    def get_tracks_by_bpm_range(self, min_bpm: float, max_bpm: float) -> List[Dict]:
        """Filtra brani per range BPM."""
        if not self.analysis_results:
            self.analyze()
        
        return [
            t for t in self.analysis_results.get('tracks', [])
            if min_bpm <= t.get('bpm', 0) <= max_bpm
        ]
    
    def get_tracks_by_energy(self, min_energy: float = 0.0, max_energy: float = 1.0) -> List[Dict]:
        """Filtra brani per livello di energia."""
        if not self.analysis_results:
            self.analyze()
        
        return [
            t for t in self.analysis_results.get('tracks', [])
            if min_energy <= t.get('energy', 0) <= max_energy
        ]
    
    def get_sorted_by(self, key: str, reverse: bool = False) -> List[Dict]:
        """Ordina i brani per una chiave specifica."""
        if not self.analysis_results:
            self.analyze()
        
        tracks = self.analysis_results.get('tracks', [])
        return sorted(tracks, key=lambda x: x.get(key, 0), reverse=reverse)


class MusicPlayer:
    """
    Server OSC che riceve parametri musicali e riproduce il brano più vicino.
    
    Riceve via OSC:
    - arousal, valence, bpm, instrumentalness, electronicness
    
    Calcola la distanza euclidea normalizzata e riproduce il brano più vicino
    per 30 secondi a partire dal primo onset.
    """
    
    def __init__(
        self, 
        analyzer: MusicAnalyzer,
        osc_ip: str = "0.0.0.0",
        osc_port: int = 9000,
        playback_duration: float = 30.0
    ):
        """
        Inizializza il player.
        
        Args:
            analyzer: Istanza di MusicAnalyzer con dati già analizzati
            osc_ip: Indirizzo IP per il server OSC
            osc_port: Porta per il server OSC
            playback_duration: Durata della riproduzione in secondi
        """
        self.analyzer = analyzer
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        self.playback_duration = playback_duration
        
        # Assicurati che i dati siano caricati
        if not self.analyzer.analysis_results:
            self.analyzer.analyze()
        
        self.tracks = [t for t in self.analyzer.analysis_results.get('tracks', []) 
                      if t.get('analyzed', False)]
        
        # Stato playback
        self.is_playing = False
        self.current_stream = None
        self.stop_event = threading.Event()
        
        # Normalizzazione BPM (per calcolo distanza)
        bpms = [t.get('bpm', 100) for t in self.tracks]
        self.bpm_min = min(bpms) if bpms else 60
        self.bpm_max = max(bpms) if bpms else 180
        
        # Pre-calcola matrice features per ricerca veloce
        self._feature_matrix = None
        self._build_feature_matrix()
        
        # Server OSC
        self.server = None
        self.server_thread = None
        
    def _normalize_bpm(self, bpm: float) -> float:
        """Normalizza BPM tra 0 e 1."""
        if self.bpm_max == self.bpm_min:
            return 0.5
        return (bpm - self.bpm_min) / (self.bpm_max - self.bpm_min)
    
    def _track_to_feature_vector(self, track: Dict) -> np.ndarray:
        """Converte una traccia in un vettore di features normalizzato."""
        instrumentalness = track.get('instrumentalness', None)
        if instrumentalness is None:
            # Fallback: se manca, usa danceability come proxy (meno ideale ma evita rotture)
            instrumentalness = min(1.0, track.get('danceability', 0.0))

        electronicness = track.get('electronicness', None)
        if electronicness is None:
            # Fallback: se manca, usa mood_aggressive come proxy (meno ideale)
            electronicness = track.get('mood_aggressive', 0.0)

        return np.array([
            track.get('arousal', 0.5),
            track.get('valence', 0.5),
            self._normalize_bpm(track.get('bpm', 100)),
            max(0.0, min(1.0, float(instrumentalness))),
            max(0.0, min(1.0, float(electronicness)))
        ])
    
    def _build_feature_matrix(self):
        """Pre-calcola la matrice di features per tutte le tracce."""
        if not self.tracks:
            self._feature_matrix = None
            return
        
        self._feature_matrix = np.array([self._track_to_feature_vector(t) for t in self.tracks])
    
    def _calculate_distance_squared(self, track: Dict, target: Dict) -> float:
        """
        Calcola la distanza euclidea AL QUADRATO (senza radice) tra un brano e i parametri target.
        Sufficiente per confronti di distanza minima.
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
        return np.dot(diff, diff)  # Distanza al quadrato, senza radice
    
    def _calculate_distance(self, track: Dict, target: Dict) -> float:
        """Wrapper per compatibilità: restituisce distanza (con radice)."""
        return np.sqrt(self._calculate_distance_squared(track, target))
    
    def find_closest_track(self, arousal: float, valence: float, bpm: float,
                          instrumentalness: float, electronicness: float) -> Optional[Dict]:
        """
        Trova il brano con distanza minima usando numpy vectorized.
        Usa distanza euclidea al quadrato (senza radice) per efficienza.
        """
        if not self.tracks:
            return None
        
        # Costruisci vettore target normalizzato
        target_vec = np.array([
            arousal,
            valence,
            self._normalize_bpm(bpm),
            instrumentalness,
            electronicness
        ])
        
        # Calcolo vectorized: distanza al quadrato per tutte le tracce
        if self._feature_matrix is not None:
            diff = self._feature_matrix - target_vec
            distances_sq = np.sum(diff ** 2, axis=1)  # Senza radice
            closest_idx = np.argmin(distances_sq)
            return self.tracks[closest_idx]
        
        # Fallback se matrice non disponibile
        min_dist_sq = float('inf')
        closest = None
        
        for track in self.tracks:
            track_vec = self._track_to_feature_vector(track)
            diff = track_vec - target_vec
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest = track
        
        return closest
    
    def _find_first_onset(self, audio_path: str) -> float:
        """Trova il tempo del primo onset nel file audio."""
        try:
            loader = MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            
            onset_rate = es.OnsetRate()
            onsets, _ = onset_rate(audio)
            
            if len(onsets) > 0:
                return float(onsets[0])
            return 0.0
        except Exception as e:
            print(f"  ⚠️ Errore nel trovare onset: {e}")
            return 0.0
    
    def _play_audio(self, audio_path: str, start_time: float, duration: float):
        """Riproduce l'audio dal tempo specificato per la durata indicata."""
        if not AUDIO_PLAYBACK_AVAILABLE:
            print("❌ Playback non disponibile. Installa sounddevice.")
            return
        
        try:
            # Carica audio
            loader = MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            
            sample_rate = 44100
            start_sample = int(start_time * sample_rate)
            end_sample = start_sample + int(duration * sample_rate)
            
            # Assicurati di non superare la lunghezza dell'audio
            end_sample = min(end_sample, len(audio))
            
            if start_sample >= len(audio):
                start_sample = 0
            
            audio_segment = audio[start_sample:end_sample]
            
            # Fade in/out per evitare click
            fade_samples = int(0.05 * sample_rate)  # 50ms fade
            if len(audio_segment) > fade_samples * 2:
                # Fade in
                fade_in = np.linspace(0, 1, fade_samples)
                audio_segment[:fade_samples] *= fade_in
                # Fade out
                fade_out = np.linspace(1, 0, fade_samples)
                audio_segment[-fade_samples:] *= fade_out
            
            print(f"  ▶️ Riproduzione: {duration:.1f}s da {start_time:.2f}s")
            
            self.is_playing = True
            self.stop_event.clear()
            
            # Riproduzione
            sd.play(audio_segment, sample_rate)
            
            # Attendi fine riproduzione o stop
            elapsed = 0
            while elapsed < duration and not self.stop_event.is_set():
                time.sleep(0.1)
                elapsed += 0.1
            
            sd.stop()
            self.is_playing = False
            print("  ⏹️ Riproduzione terminata")
            
        except Exception as e:
            print(f"  ❌ Errore riproduzione: {e}")
            self.is_playing = False
    
    def stop_playback(self):
        """Ferma la riproduzione corrente."""
        self.stop_event.set()
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        self.is_playing = False
    
    def play_closest(self, arousal: float, valence: float, bpm: float,
                    instrumentalness: float, electronicness: float):
        """Trova e riproduce il brano più vicino ai parametri."""
        # Ferma eventuale riproduzione in corso
        self.stop_playback()
        
        print(f"\n🎯 Ricevuti parametri OSC:")
        print(f"   Arousal={arousal:.2f}, Valence={valence:.2f}, BPM={bpm:.1f}")
        print(f"   Instrumentalness={instrumentalness:.2f}, Electronicness={electronicness:.2f}")
        
        # Trova brano più vicino
        closest = self.find_closest_track(arousal, valence, bpm, instrumentalness, electronicness)
        
        if closest is None:
            print("❌ Nessun brano trovato nel dataset!")
            return
        
        print(f"\n🎵 Brano più vicino: {closest['filename']}")
        print(f"   Distance: {self._calculate_distance(closest, {'arousal': arousal, 'valence': valence, 'bpm': bpm, 'instrumentalness': instrumentalness, 'electronicness': electronicness}):.3f}")
        print(f"   Track: arousal={closest.get('arousal', 0):.2f}, valence={closest.get('valence', 0):.2f}, bpm={closest.get('bpm', 0):.1f}")
        
        # Trova primo onset
        audio_path = closest.get('path', '')
        first_onset = self._find_first_onset(audio_path)
        print(f"   Primo onset: {first_onset:.2f}s")
        
        # Riproduci in un thread separato
        play_thread = threading.Thread(
            target=self._play_audio,
            args=(audio_path, first_onset, self.playback_duration)
        )
        play_thread.daemon = True
        play_thread.start()
    
    def _osc_handler(self, address: str, *args):
        """Handler per messaggi OSC."""
        print(f"\n📨 Messaggio OSC ricevuto: {address}")
        
        if len(args) >= 5:
            arousal = float(args[0])
            valence = float(args[1])
            bpm = float(args[2])
            instrumentalness = float(args[3])
            electronicness = float(args[4])

            self.play_closest(arousal, valence, bpm, instrumentalness, electronicness)
        else:
            print(f"⚠️ Parametri insufficienti. Ricevuti: {args}")
            print("   Formato atteso: arousal valence bpm instrumentalness electronicness")
    
    def start_server(self, osc_address: str = "/music/play"):
        """Avvia il server OSC."""
        if not OSC_AVAILABLE:
            print("❌ python-osc non disponibile!")
            return
        
        # Setup dispatcher
        disp = dispatcher.Dispatcher()
        disp.map(osc_address, self._osc_handler)
        disp.set_default_handler(self._osc_handler)
        
        # Crea server
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.osc_ip, self.osc_port), disp
        )
        
        print("\n" + "="*60)
        print("🎧 MUSIC PLAYER OSC SERVER")
        print("="*60)
        print(f"   Indirizzo: {self.osc_ip}:{self.osc_port}")
        print(f"   OSC Address: {osc_address}")
        print(f"   Durata playback: {self.playback_duration}s")
        print(f"   Brani disponibili: {len(self.tracks)}")
        print("\n   Formato messaggio OSC:")
        print(f"   {osc_address} [arousal] [valence] [bpm] [instrumentalness] [electronicness]")
        print("\n   Esempio (valori 0-1 tranne BPM):")
        print(f"   {osc_address} 0.7 0.3 120 0.8 0.5")
        print("\n   Premi Ctrl+C per fermare il server")
        print("="*60 + "\n")
        
        # Avvia server
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server fermato.")
            self.stop_playback()
    
    def start_server_async(self, osc_address: str = "/music/play"):
        """Avvia il server OSC in un thread separato."""
        self.server_thread = threading.Thread(
            target=self.start_server,
            args=(osc_address,)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        return self.server_thread


# Esempio di utilizzo
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Analyzer & OSC Player")
    parser.add_argument('--server', action='store_true', 
                       help='Avvia il server OSC per ricevere parametri e riprodurre musica')
    parser.add_argument('--port', type=int, default=9000,
                       help='Porta OSC (default: 9000)')
    parser.add_argument('--address', type=str, default='/music/play',
                       help='Indirizzo OSC (default: /music/play)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Durata riproduzione in secondi (default: 30)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Solo analisi, senza avviare il server')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Rimuove il file di cache prima dell\'analisi')
    parser.add_argument('--audio-folder', type=str,
                       help='Percorso della cartella audio da analizzare')
    
    args = parser.parse_args()
    
    # Configura il percorso della cartella audio (dinamico rispetto al progetto)
    script_dir = Path(__file__).parent
    AUDIO_FOLDER = str((script_dir / "audio").resolve())
    
    # Override da CLI se fornito
    if args.audio_folder:
        AUDIO_FOLDER = args.audio_folder
    
    # Crea l'analizzatore
    try:
        analyzer = MusicAnalyzer(
            audio_folder=AUDIO_FOLDER,
            cache_file="music_analysis_cache.json"
        )
        
        # Rimuovi cache se richiesto
        if args.clear_cache:
            try:
                if analyzer.cache_file.exists():
                    analyzer.cache_file.unlink()
                    print(f"🗑️ Cache rimossa: {analyzer.cache_file}")
                else:
                    print("ℹ️ Nessuna cache da rimuovere.")
            except Exception as e:
                print(f"⚠️ Errore nella rimozione della cache: {e}")
        
        # Esegui analisi (o carica dalla cache se non è cambiato nulla)
        results = analyzer.analyze()
        
        # Mostra risultati
        print("\n" + "="*70)
        print("📊 RISULTATI ANALISI")
        print("="*70)
        
        for track in results.get('tracks', []):
            if track.get('analyzed'):
                print(f"\n🎵 {track['filename']}")
                key_info = f"{track.get('key', '?')} {track.get('scale', '?')}" if track.get('key') else "N/A"
                print(f"   Tonalità: {key_info} (strength: {track.get('key_strength', 0):.2f})")
                print(f"   BPM: {track['bpm']} (confidence: {track['beats_confidence']})")
                print(f"   Energia: {track['energy']:.2f} | Arousal: {track['arousal']:.2f} | Valence: {track['valence']:.2f}")
                print(f"   Danceability: {track['danceability']:.2f}")
                print(f"   Mood: 😊 Happy={track['mood_happy']:.2f} | 😢 Sad={track['mood_sad']:.2f} | 😤 Aggr={track['mood_aggressive']:.2f} | 😌 Relax={track['mood_relaxed']:.2f}")
                print(f"   Durata: {track['duration_seconds']}s")
        
        # Mostra summary
        if 'summary' in results:
            print("\n" + "="*70)
            print("📈 STATISTICHE GENERALI")
            print("="*70)
            summary = results['summary']
            print(f"   BPM medio: {summary.get('avg_bpm', 0)}")
            print(f"   Range BPM: {summary.get('min_bpm', 0)} - {summary.get('max_bpm', 0)}")
            print(f"   Energia media: {summary.get('avg_energy', 0):.2f}")
            print(f"   Durata totale: {summary.get('total_duration_minutes', 0):.1f} minuti")
        
        # Avvia server OSC se richiesto
        if args.server and not args.analyze_only:
            player = MusicPlayer(
                analyzer=analyzer,
                osc_port=args.port,
                playback_duration=args.duration
            )
            player.start_server(osc_address=args.address)
        elif not args.analyze_only:
            print("\n💡 Per avviare il server OSC usa: python music_score.py --server")
            print("   Opzioni: --port 9000 --address /music/play --duration 30")
        
    except FileNotFoundError as e:
        print(f"❌ Errore: {e}")
        print(f"   Crea la cartella 'audio' in: {AUDIO_FOLDER}")
