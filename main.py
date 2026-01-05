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
from osc_sender import send_pipeline_complete, send_with_metadata
from process_coordinator import signal_pipeline_complete
import re
import time
import threading
from pathlib import Path


# Output directories
CAPTURES_DIR = "captures"
AGED_DIR = "aged_outputs"
UV_DIR = "uv_outputs"
FUTURES_DIR = "futures_texts"

# Music integration
MUSIC_SCORE_AVAILABLE = False
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "musica"))
    from musica.music_score import MusicAnalyzer, MusicPlayer
    MUSIC_SCORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: music_score not available: {e}")
    MUSIC_SCORE_AVAILABLE = False



def clear_directory(directory):
    """Remove all files in a directory, create it if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"Cleared: {directory}/")


def parse_music_params_from_gemini():
    """Parse music parameters from Gemini-generated text files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    futures_dir = os.path.join(script_dir, FUTURES_DIR)
    music_params_file = os.path.join(futures_dir, "music_params.txt")
    
    music_params = []
    
    try:
        if os.path.exists(music_params_file):
            with open(music_params_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse MUSIC: lines
            # Format: MUSIC: arousal,valence,bpm,danceability,aggressive
            music_lines = re.findall(r'MUSIC:\s*([0-9.,\s]+)', content)
            
            for line in music_lines:
                values = [float(x.strip()) for x in line.strip().split(',')]
                if len(values) == 5:
                    params = {
                        'arousal': values[0],
                        'valence': values[1],
                        'bpm': values[2],
                        'danceability': values[3],
                        'aggressive': values[4]
                    }
                    music_params.append(params)
        
        if not music_params:
            # Fallback to default values if no params found
            print("⚠️ No music parameters found, using defaults")
            music_params = [
                {'arousal': 0.5, 'valence': 0.5, 'bpm': 100, 'danceability': 0.5, 'aggressive': 0.3},
                {'arousal': 0.6, 'valence': 0.4, 'bpm': 110, 'danceability': 0.6, 'aggressive': 0.4},
                {'arousal': 0.7, 'valence': 0.3, 'bpm': 120, 'danceability': 0.5, 'aggressive': 0.5},
                {'arousal': 0.5, 'valence': 0.2, 'bpm': 90, 'danceability': 0.3, 'aggressive': 0.6},
                {'arousal': 0.3, 'valence': 0.1, 'bpm': 70, 'danceability': 0.2, 'aggressive': 0.7}
            ]
    
    except Exception as e:
        print(f"Error parsing music params: {e}")
        # Use default values
        music_params = [
            {'arousal': 0.5, 'valence': 0.5, 'bpm': 100, 'danceability': 0.5, 'aggressive': 0.3}
        ]
    
    return music_params


def start_music_player_thread(music_params):
    """Start music player in separate thread, sending params every 30 seconds."""
    if not MUSIC_SCORE_AVAILABLE:
        print("\n⚠️ Music player not available, skipping music integration")
        return None
    
    def music_loop():
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            audio_folder = os.path.join(script_dir, "musica", "audio")
            
            if not os.path.exists(audio_folder):
                print(f"⚠️ Audio folder not found: {audio_folder}")
                return
            
            # Initialize music analyzer and player
            print("\n[MUSIC] Initializing music analyzer...")
            analyzer = MusicAnalyzer(audio_folder)
            analyzer.analyze()
            
            print("[MUSIC] Starting music player...")
            player = MusicPlayer(
                analyzer=analyzer,
                osc_ip="0.0.0.0",
                osc_port=9001,  # Different port from TouchDesigner
                playback_duration=30.0
            )
            
            # Send params in sequence, every 30 seconds
            print(f"\n[MUSIC] Will play {len(music_params)} tracks (one per scene)")
            print("=" * 60)
            
            # Play each scene once, then stop
            for idx, params in enumerate(music_params):
                print(f"\n[MUSIC] Scene {idx + 1}/{len(music_params)}:")
                print(f"  Arousal: {params['arousal']:.2f}")
                print(f"  Valence: {params['valence']:.2f}")
                print(f"  BPM: {params['bpm']:.1f}")
                print(f"  Danceability: {params['danceability']:.2f}")
                print(f"  Aggressive: {params['aggressive']:.2f}")
                
                # Play the closest track
                player.play_closest(
                    arousal=params['arousal'],
                    valence=params['valence'],
                    bpm=params['bpm'],
                    danceability=params['danceability'],
                    aggressive=params['aggressive']
                )
                
                # Wait 30 seconds before next (except for the last one)
                if idx < len(music_params) - 1:
                    time.sleep(30)
            
            print("\n" + "=" * 60)
            print("✅ All scenes completed! Music playback finished.")
            print("=" * 60)
        
        except Exception as e:
            print(f"\n❌ Error in music player thread: {e}")
            import traceback
            traceback.print_exc()
    
    # Start thread
    music_thread = threading.Thread(target=music_loop, daemon=False)  # daemon=False to wait for completion
    music_thread.start()
    print("\n✅ Music player thread started")
    return music_thread


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
        current_age = int(content.split(':')[1].strip())
    except ValueError:
        print("Invalid age. Exiting.")
        sys.exit(1)
    
    # Configuration
    age_increment = 10
    num_stages = 5
    
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
    
    # Simple version - just send completion flag
    # send_pipeline_complete(ip="127.0.0.1", port=9000)
    
    # Or with metadata - sends both completion flag and texture count
    

    # Signal to coordinator that pipeline is complete
    signal_pipeline_complete()
    
    print("\n" + "=" * 50)
    print("   PIPELINE COMPLETE - Ready for Music")
    print("=" * 50)
    print("\nMusic will start when you press FINISH in the GUI\n")


def start_music_integration():
    """Start music player - called externally after FINISH button is pressed."""
    print("\n[MUSIC] Starting music integration...")
    print("-" * 40)
    
    # Parse music parameters from Gemini output
    music_params = parse_music_params_from_gemini()
    print(f"\nParsed {len(music_params)} music parameter sets from Gemini")
    
    # Start music player thread
    music_thread = start_music_player_thread(music_params)
    
    if music_thread:
        print(f"\nWill play {len(music_params)} tracks (30s each), then stop automatically")
        print("Press Ctrl+C to stop early\n")
        
        try:
            # Wait for the music thread to complete
            music_thread.join()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
    else:
        print("\nMusic player not started")


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


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--music-only":
            start_music_integration()
        elif sys.argv[1] == "--analyze-only":
            pre_analyze_music()
    else:
        run_pipeline()
