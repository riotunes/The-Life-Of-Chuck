"""
Music Analysis Module using Essentia
Analizza file audio per mood, energia e BPM.
Salva i risultati in JSON e li ricarica se la cartella non è cambiata.
"""

import os
import json
import hashlib
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


# Esempio di utilizzo
if __name__ == "__main__":
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
        
    except FileNotFoundError as e:
        print(f"❌ Errore: {e}")
        print(f"   Crea la cartella 'audio' in: {AUDIO_FOLDER}")
