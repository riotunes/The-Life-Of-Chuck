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
from face_aging import generate_age_progression
from camera_to_uv import convert_images_to_uv
from process_coordinator import signal_pipeline_complete
from shared_config import calculate_num_stages, get_age_increment
import re
import time

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
    user_data_path = os.path.join(script_dir, "user_data.txt")
    
    # Read expected num_stages from user_data.txt for consistency
    expected_stages = None
    try:
        with open(user_data_path, 'r') as f:
            for line in f:
                if line.startswith('NUM_STAGES:'):
                    expected_stages = int(line.split(':')[1].strip())
                    break
    except:
        pass
    
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
    
    # Ensure music_params matches expected_stages
    if expected_stages is not None:
        if len(music_params) > expected_stages:
            print(f"⚠️ Trimming music params from {len(music_params)} to {expected_stages}")
            music_params = music_params[:expected_stages]
        elif len(music_params) < expected_stages:
            print(f"⚠️ Extending music params from {len(music_params)} to {expected_stages}")
            # Repeat last params to fill
            while len(music_params) < expected_stages:
                music_params.append(music_params[-1] if music_params else 
                    {'arousal': 0.5, 'valence': 0.5, 'bpm': 100, 'instrumentalness': 0.5, 'electronicness': 0.3})
    
    return music_params


def _build_playlist(analyzer, music_params):
    """
    Usa MusicPlayer per scegliere i brani più vicini ai parametri.
    Ritorna una lista di dict con path e filename.
    Usa vincolo anti-ripetizione: ogni brano può essere scelto solo una volta.
    """
    player = MusicPlayer(
        analyzer=analyzer,
        osc_ip="0.0.0.0",
        osc_port=9001,
        playback_duration=1.0
    )

    playlist = []
    used_tracks = []  # list of filenames already used to avoid repetition
    
    for idx, params in enumerate(music_params):
        closest = player.find_closest_track(
            arousal=params['arousal'],
            valence=params['valence'],
            bpm=params['bpm'],
            instrumentalness=params['instrumentalness'],
            electronicness=params['electronicness'],
            exclude_filenames=used_tracks  # restraint to avoid repeating tracks across scenes
        )
        if closest:
            filename = closest.get('filename', '')
            used_tracks.append(filename)  # add to used list to prevent future selection
            playlist.append({
                'path': closest.get('path', ''),
                'filename': filename,
                'params': params,
                'scene': idx + 1
            })
            print(f"  Scene {idx + 1}: {filename[:60]}")
        else:
            print(f"  Scene {idx + 1}: No track found!")
    return playlist


def _send_playlist_sequenced(playlist, ip="127.0.0.1", port=57120,
                             segment_dur=2.0, lead=1.0):
    """
    send playlist to SuperCollider one track at a time, with a delay between each to allow for crossfading.
    segment_dur is the total duration of each track segment (e.g., 30s), lead is how much before the end of the segment to send the next track (e.g., 1s before).
    This allows SuperCollider to handle crossfading between tracks.
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
        print("✅ Playlist sent to SuperCollider with sequencing.")
        return True
    except KeyboardInterrupt:
        print("⏹️ interrupted by user, stopping playlist sending.")
        return False


def start_music_integration():
    """
    selects tracks based on Gemini-generated parameters and sends them to SuperCollider.
    """
    if not MUSIC_SCORE_AVAILABLE:
        print("\n⚠️ Music analyzer not available, skipping music integration")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "musica", "audio")
    
    if not os.path.exists(audio_folder):
        print(f"⚠️ Audio folder not found: {audio_folder}")
        return
    
    # 1) Parse music parameters from Gemini-generated text files (or use defaults if not found)
    music_params = parse_music_params_from_gemini()
    
    # 2) Build playlist based on parameters, ensuring no track is repeated across scenes
    print("\n[MUSIC] analysis/ cathing track...")
    analyzer = MusicAnalyzer(audio_folder)
    analyzer.analyze()
    
    # 3) build playlist with anti-repetition constraint and print selected tracks
    print("[MUSIC] Selecting tracks based on parameters (with anti-repetition)...")
    playlist = _build_playlist(analyzer, music_params)
    if not playlist:
        print("❌ No tracks selected, skipping music sending.")
        return
    
    print("[MUSIC] Sending playlist to SuperCollider...")
    _send_playlist_sequenced(
        playlist,
        ip="127.0.0.1",
        port=57120,
        segment_dur=2.0,
        lead=1.0
    )
    print("🎵 Playlist sent to SuperCollider!")


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
    
    # Configuration - read num_stages from user_data.txt if available (for consistency)
    age_increment = get_age_increment()
    
    # Check if num_stages already saved by GUI
    num_stages = None
    for line in content.split('\n'):
        if line.startswith('NUM_STAGES:'):
            num_stages = int(line.split(':')[1].strip())
            break
    
    if num_stages is None:
        # Calculate and save if not present
        num_stages = calculate_num_stages(current_age)
        with open(user_data_path, 'a') as f:
            f.write(f"NUM_STAGES: {num_stages}\n")

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