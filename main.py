# main.py

import os
import shutil
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify API key is set
if not os.environ.get("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found!")
    print()
    print("Please either:")
    print("  1. Create a .env file with: GEMINI_API_KEY=your_key_here")
    print("  2. Or set it in terminal: export GEMINI_API_KEY=your_key_here")
    print()
    sys.exit(1)

# Import our modules (after loading env)
from camera_capture import capture_face
from face_aging import generate_age_progression
from camera_to_uv import convert_images_to_uv
from osc_sender import send_pipeline_complete, send_with_metadata, send_num_stages
from process_coordinator import signal_pipeline_complete
from shared_config import calculate_num_stages, get_age_increment
import re
import time
import threading
from pathlib import Path

# Music integration
MUSIC_SCORE_AVAILABLE = False
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "musica"))
    from musica.music_score import MusicAnalyzer, MusicPlayer
    MUSIC_SCORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: music_score not available: {e}")
    MUSIC_SCORE_AVAILABLE = False

# OSC client for SuperCollider
try:
    from pythonosc import udp_client
    OSC_CLIENT_AVAILABLE = True
except ImportError:
    OSC_CLIENT_AVAILABLE = False
    print("⚠️ python-osc non installato. Installa con: pip install python-osc")

# Output directories
CAPTURES_DIR = "captures"
AGED_DIR = "aged_outputs"
UV_DIR = "uv_outputs"
FUTURES_DIR = "futures_texts"


def clear_directory(directory):
    """Remove all files in a directory, create it if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"Cleared: {directory}/")


def parse_music_params_from_gemini():
    """Parse music parameters from Gemini-generated text files in music_params folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    music_params_dir = os.path.join(script_dir, "music_params")
    
    music_params = []
    
    try:
        if os.path.exists(music_params_dir):
            # Get all music param files sorted
            music_files = sorted([f for f in os.listdir(music_params_dir) if f.startswith("music_") and f.endswith(".txt")])
            
            for music_file in music_files:
                filepath = os.path.join(music_params_dir, music_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Parse MUSIC: line
                # Format: MUSIC: arousal,valence,bpm,instrumentalness,electronicness
                match = re.search(r'MUSIC:\s*([0-9.,\s]+)', content)
                if match:
                    values = [float(x.strip()) for x in match.group(1).strip().split(',')]
                    if len(values) == 5:
                        params = {
                            'arousal': values[0],
                            'valence': values[1],
                            'bpm': values[2],
                            'instrumentalness': values[3],
                            'electronicness': values[4]
                        }
                        music_params.append(params)
        
        if not music_params:
            # Fallback to default values if no params found
            print("⚠️ No music parameters found, using defaults")
            music_params = [
                {'arousal': 0.5, 'valence': 0.5, 'bpm': 100, 'instrumentalness': 0.5, 'electronicness': 0.3},
                {'arousal': 0.6, 'valence': 0.4, 'bpm': 110, 'instrumentalness': 0.6, 'electronicness': 0.4},
                {'arousal': 0.7, 'valence': 0.3, 'bpm': 120, 'instrumentalness': 0.5, 'electronicness': 0.5},
                {'arousal': 0.5, 'valence': 0.2, 'bpm': 90, 'instrumentalness': 0.3, 'electronicness': 0.6},
                {'arousal': 0.3, 'valence': 0.1, 'bpm': 70, 'instrumentalness': 0.2, 'electronicness': 0.7}
            ]
    
    except Exception as e:
        print(f"Error parsing music params: {e}")
        # Use default values
        music_params = [
            {'arousal': 0.5, 'valence': 0.5, 'bpm': 100, 'instrumentalness': 0.5, 'electronicness': 0.3}
        ]
    
    return music_params


def _build_playlist(analyzer, music_params):
    """
    Usa MusicPlayer per scegliere i brani più vicini ai parametri.
    Ritorna una lista di dict con path e filename.
    """
    player = MusicPlayer(
        analyzer=analyzer,
        osc_ip="0.0.0.0",
        osc_port=9001,
        playback_duration=1.0
    )

    playlist = []
    for idx, params in enumerate(music_params):
        closest = player.find_closest_track(
            arousal=params['arousal'],
            valence=params['valence'],
            bpm=params['bpm'],
            instrumentalness=params['instrumentalness'],
            electronicness=params['electronicness']
        )
        if closest:
            playlist.append({
                'path': closest.get('path', ''),
                'filename': closest.get('filename', ''),
                'params': params,
                'scene': idx + 1
            })
            print(f"  Scene {idx + 1}: {closest.get('filename', 'Unknown')[:60]}")
        else:
            print(f"  Scene {idx + 1}: No track found!")
    return playlist


def _send_playlist_sequenced(playlist, ip="127.0.0.1", port=57120,
                             segment_dur=2.0, lead=1.0):
    """
    Invia i path a SuperCollider in sequenza:
    - primo subito
    - poi uno ogni (segment_dur - lead) secondi (default 29s)
    Processo bloccante: garantisce che tutti i messaggi vengano inviati prima di uscire.
    """
    if not OSC_CLIENT_AVAILABLE:
        print("❌ python-osc non disponibile, impossibile inviare a SuperCollider.")
        return False
    if not playlist:
        print("⚠️ Playlist vuota, niente da inviare.")
        return False

    client = udp_client.SimpleUDPClient(ip, port)

    try:
        for idx, item in enumerate(playlist):
            path = item.get('path', '')
            if not path:
                continue
            client.send_message("/addFile", path)
            print(f"➡️  [SC] /addFile {path} (scene {idx+1}/{len(playlist)})")
            if idx < len(playlist) - 1:
                wait_time = max(1.0, segment_dur - lead)  # es. 29s
                time.sleep(wait_time)
        print("✅ Playlist inviata a SuperCollider (sequenziale).")
        return True
    except KeyboardInterrupt:
        print("⏹️ Interrotto dall’utente.")
        return False


def start_music_integration():
    """
    Seleziona i brani (in base ai parametri Gemini) e invia la playlist a SuperCollider.
    Non riproduce audio in Python.
    """
    if not MUSIC_SCORE_AVAILABLE:
        print("\n⚠️ Music analyzer not available, skipping music integration")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "musica", "audio")
    
    if not os.path.exists(audio_folder):
        print(f"⚠️ Audio folder not found: {audio_folder}")
        return
    
    # 1) Leggi parametri musicali generati
    music_params = parse_music_params_from_gemini()
    
    # 2) Analizza (o carica cache) prima di scegliere i brani
    print("\n[MUSIC] Analisi/caching brani...")
    analyzer = MusicAnalyzer(audio_folder)
    analyzer.analyze()
    
    # 3) Costruisci playlist migliore per i parametri
    print("[MUSIC] Selezione brani per le scene...")
    playlist = _build_playlist(analyzer, music_params)
    if not playlist:
        print("❌ Nessun brano selezionato, interruzione.")
        return
    
    print("[MUSIC] Invio playlist a SuperCollider (scaglionata)...")
    _send_playlist_sequenced(
        playlist,
        ip="127.0.0.1",
        port=57120,
        segment_dur=2.0,
        lead=1.0
    )
    print("🎵 Playlist pronta su SuperCollider (30s a traccia, crossfade gestito in SC).")


def pre_analyze_music():
    """Pre-analyze music files at startup to build cache."""
    if not MUSIC_SCORE_AVAILABLE:
        print("\n⚠️ Music analyzer not available")
        return
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_folder = os.path.join(script_dir, "musica", "audio")
        
        if not os.path.exists(audio_folder):
            print(f"⚠️ Audio folder not found: {audio_folder}")
            return
        
        print("\n[MUSIC] Pre-analyzing audio files...")
        print("-" * 40)
        analyzer = MusicAnalyzer(audio_folder)
        analyzer.analyze()
        print("✅ Music analysis complete and cached!\n")
    
    except Exception as e:
        print(f"⚠️ Error during pre-analysis: {e}")


def run_pipeline():
    """Run the complete face aging pipeline."""
    
    print("=" * 50)
    print("   LIFE OF CHUCK - AGING PIPELINE")
    print("=" * 50)
    print()
    
    # Step 0: Clear previous outputs
    print("[SETUP] Clearing previous outputs...")
    clear_directory(CAPTURES_DIR)
    clear_directory(AGED_DIR)
    clear_directory(UV_DIR)
    print()
    
    # Step 1: Capture face
    '''print("[STEP 1/3] CAMERA CAPTURE")
    print("-" * 40)
    captured_image = capture_face(output_dir=CAPTURES_DIR)
    
    if captured_image is None:
        print("\nError: No image captured. Exiting.")
        sys.exit(1)
    
    print(f"\nCaptured: {captured_image}")
    print()'''
    
    # Step 2: Get age and generate progression
    print("[STEP 2/3] AGE PROGRESSION")
    print("-" * 40)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_path = os.path.join(script_dir, 'user_data.txt')

    try:
        with open(user_data_path, 'r') as f:
            content = f.read().strip()
        # Extract age from the AGE: line
        for line in content.split('\n'):
            if line.startswith('AGE:'):
                current_age = int(line.split(':')[1].strip())
                break
        else:
            raise ValueError("AGE line not found")
    except (ValueError, FileNotFoundError) as e:
        print(f"Invalid age or file not found: {e}")
        sys.exit(1)
    
    # Configuration - dynamically calculated based on user's age
    age_increment = get_age_increment()
    num_stages = calculate_num_stages(current_age)

    print(f"\n🎯 Age-based generation: {num_stages} stages for age {current_age}")
    print(f"\nGenerating ages: ", end="")
    print(", ".join([str(current_age + (i * age_increment)) for i in range(1, num_stages + 1)]))
    print()
    
    aged_images = generate_age_progression(
        input_image_path="chuck_origin.jpg",
        current_age=current_age,
        output_dir=AGED_DIR,
        age_increment=age_increment,
        num_stages=num_stages
    )
    
    if len(aged_images) == 0:
        print("\nError: No aged images generated. Exiting.")
        sys.exit(1)
    
    print(f"\nGenerated {len(aged_images)} aged images")
    print()
    
    # Step 3: Convert to UV textures
    print("[STEP 3/3] UV TEXTURE CONVERSION")
    print("-" * 40)
    
    uv_textures = convert_images_to_uv(
        input_dir=AGED_DIR,
        output_dir=UV_DIR,
        output_size=(1024, 1024)
    )
    
    if len(uv_textures) == 0:
        print("\nError: No UV textures generated. Exiting.")
        sys.exit(1)
    
    print()
    
    # Summary
    print("=" * 50)
    print("   PIPELINE COMPLETE")
    print("=" * 50)
    print()
    print(f"Captured image:  {CAPTURES_DIR}/")
    print(f"Aged images:     {AGED_DIR}/")
    print(f"UV textures:     {UV_DIR}/")
    print()
    print("Generated files:")
    for uv_path in uv_textures:
        print(f"  - {uv_path}")
    print()
    
    # Send OSC notification to TouchDesigner
    print("[OSC] Notifying TouchDesigner...")
    print("-" * 40)
    
    # Signal to coordinator that pipeline is complete
    signal_pipeline_complete()
    
    print("\n" + "=" * 50)
    print("   PIPELINE COMPLETE - Ready for Music")
    print("=" * 50)
    print("\nMusic will start when you press FINISH in the GUI\n")


def start_music_integration_cli():
    """Wrapper per CLI (compatibile con --music-only)."""
    start_music_integration()


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--music-only":
            start_music_integration_cli()
        elif sys.argv[1] == "--analyze-only":
            pre_analyze_music()
        else:
            # Any other argument means run pipeline (e.g., image path, age)
            run_pipeline()
    else:
        run_pipeline()