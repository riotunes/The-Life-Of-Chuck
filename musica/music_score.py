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

# Audio playback imports
try:
    import sounddevice as sd
    import numpy as np
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
            'valence': 0.5,
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
            is_minor = 0.5  # Default neutro
            try:
                key_extractor = es.KeyExtractor()
                key, scale, key_strength = key_extractor(audio)
                results['key'] = key
                results['scale'] = scale
                results['key_strength'] = round(float(key_strength), 3)
                
                # Minor mode = triste, Major mode = felice
                if scale == 'minor':
                    is_minor = 1.0
                elif scale == 'major':
                    is_minor = 0.0
                else:
                    is_minor = 0.5
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
            
            # === CALCOLO MOOD (MIGLIORATO) ===
            # Il MODO MUSICALE è il fattore più importante per happy/sad!
            # - Minor mode → triste (anche con alta energia, es. Skyfall)
            # - Major mode → felice
            # - Alta dissonanza → tensione, drammaticità
            # - Spectral flatness alto → cupo
            
            # Arousal (attivazione): energia + brightness + onset rate (peso aumentato)
            arousal = (results['energy'] * 0.30 + norm_centroid * 0.25 + norm_onset * 0.45)
            results['arousal'] = round(arousal, 3)
            
            # Valence (positività): MODO è fondamentale!
            # is_minor: 1.0 = minore (triste), 0.0 = maggiore (felice)
            mode_contribution = (1 - is_minor)  # 1 se maggiore, 0 se minore
            brightness_contribution = norm_centroid * 0.5 + (1 - avg_flatness) * 0.5
            
            # Valence: modo pesa 50%, brightness 30%, danceability 20%
            valence = (mode_contribution * 0.50 + 
                      brightness_contribution * 0.30 + 
                      min(1.0, results['danceability']) * 0.20)
            
            # Penalizza valence per alta dissonanza (tensione = meno felice)
            valence = valence * (1 - avg_dissonance * 0.3)
            results['valence'] = round(max(0, min(1.0, valence)), 3)
            
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
    - arousal, valence, bpm, danceability, aggressive
    
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
        
        # Server OSC
        self.server = None
        self.server_thread = None
        
    def _normalize_bpm(self, bpm: float) -> float:
        """Normalizza BPM tra 0 e 1."""
        if self.bpm_max == self.bpm_min:
            return 0.5
        return (bpm - self.bpm_min) / (self.bpm_max - self.bpm_min)
    
    def _calculate_distance(self, track: Dict, target: Dict) -> float:
        """
        Calcola la distanza euclidea normalizzata tra un brano e i parametri target.
        Tutte le dimensioni sono normalizzate tra 0 e 1.
        """
        # Normalizza BPM del track
        track_bpm_norm = self._normalize_bpm(track.get('bpm', 100))
        target_bpm_norm = self._normalize_bpm(target.get('bpm', 100))
        
        # Calcola distanza euclidea
        distance = (
            (track.get('arousal', 0.5) - target.get('arousal', 0.5)) ** 2 +
            (track.get('valence', 0.5) - target.get('valence', 0.5)) ** 2 +
            (track_bpm_norm - target_bpm_norm) ** 2 +
            (min(1.0, track.get('danceability', 0)) - target.get('danceability', 0.5)) ** 2 +
            (track.get('mood_aggressive', 0) - target.get('aggressive', 0.5)) ** 2
        ) ** 0.5
        
        return distance
    
    def find_closest_track(self, arousal: float, valence: float, bpm: float, 
                          danceability: float, aggressive: float) -> Optional[Dict]:
        """Trova il brano con distanza minima dai parametri target."""
        if not self.tracks:
            return None
        
        target = {
            'arousal': arousal,
            'valence': valence,
            'bpm': bpm,
            'danceability': danceability,
            'aggressive': aggressive
        }
        
        closest = None
        min_distance = float('inf')
        
        for track in self.tracks:
            dist = self._calculate_distance(track, target)
            if dist < min_distance:
                min_distance = dist
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
                    danceability: float, aggressive: float):
        """Trova e riproduce il brano più vicino ai parametri."""
        # Ferma eventuale riproduzione in corso
        self.stop_playback()
        
        print(f"\n🎯 Ricevuti parametri OSC:")
        print(f"   Arousal={arousal:.2f}, Valence={valence:.2f}, BPM={bpm:.1f}")
        print(f"   Danceability={danceability:.2f}, Aggressive={aggressive:.2f}")
        
        # Trova brano più vicino
        closest = self.find_closest_track(arousal, valence, bpm, danceability, aggressive)
        
        if closest is None:
            print("❌ Nessun brano trovato nel dataset!")
            return
        
        print(f"\n🎵 Brano più vicino: {closest['filename']}")
        print(f"   Distance: {self._calculate_distance(closest, {'arousal': arousal, 'valence': valence, 'bpm': bpm, 'danceability': danceability, 'aggressive': aggressive}):.3f}")
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
            danceability = float(args[3])
            aggressive = float(args[4])
            
            self.play_closest(arousal, valence, bpm, danceability, aggressive)
        else:
            print(f"⚠️ Parametri insufficienti. Ricevuti: {args}")
            print("   Formato atteso: arousal valence bpm danceability aggressive")
    
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
        print(f"   {osc_address} [arousal] [valence] [bpm] [danceability] [aggressive]")
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
    
    args = parser.parse_args()
    
    # Configura il percorso della cartella audio
    AUDIO_FOLDER = "/Users/riccardotocci/Desktop/prototype_the_life_of_chuck/musica/audio"
    
    # Crea l'analizzatore
    try:
        analyzer = MusicAnalyzer(
            audio_folder=AUDIO_FOLDER,
            cache_file="music_analysis_cache.json"
        )
        
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
