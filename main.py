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
from shared_config import calculate_num_stages, get_age_increment
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
                # Format: MUSIC: arousal,valence,bpm,danceability,aggressive
                match = re.search(r'MUSIC:\s*([0-9.,\s]+)', content)
                if match:
                    values = [float(x.strip()) for x in match.group(1).strip().split(',')]
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


def start_music_player_thread(music_params, silent=False):
    """Start music player in separate thread with pre-calculated tracks and crossfade."""
    if not MUSIC_SCORE_AVAILABLE:
        if not silent:
            print("\n⚠️ Music player not available, skipping music integration")
        return None
    
    try:
        import numpy as np
        import sounddevice as sd
        from essentia.standard import MonoLoader
    except ImportError as e:
        if not silent:
            print(f"\n⚠️ Audio playback not available: {e}")
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "musica", "audio")
    
    if not os.path.exists(audio_folder):
        if not silent:
            print(f"⚠️ Audio folder not found: {audio_folder}")
        return None
    
    # ===== PRE-CALCULATE ALL TRACKS BEFORE PLAYBACK =====
    if not silent:
        print("\n[MUSIC] Preparing tracks...")
    
    # Initialize analyzer (uses cache if available)
    analyzer = MusicAnalyzer(audio_folder)
    analyzer.analyze()
    
    # Create player for calculations
    player = MusicPlayer(
        analyzer=analyzer,
        osc_ip="0.0.0.0",
        osc_port=9001,
        playback_duration=30.0
    )
    
    # Pre-calculate all closest tracks
    playlist = []
    for idx, params in enumerate(music_params):
        closest = player.find_closest_track(
            arousal=params['arousal'],
            valence=params['valence'],
            bpm=params['bpm'],
            danceability=params['danceability'],
            aggressive=params['aggressive']
        )
        
        if closest:
            playlist.append({
                'path': closest.get('path', ''),
                'filename': closest.get('filename', ''),
                'params': params,
                'scene': idx + 1
            })
            if not silent:
                print(f"  Scene {idx + 1}: {closest.get('filename', 'Unknown')[:40]}")
        else:
            if not silent:
                print(f"  Scene {idx + 1}: No track found!")
    
    if not playlist:
        if not silent:
            print("❌ No tracks found for any scene!")
        return None
    
    if not silent:
        print(f"✅ {len(playlist)} tracks ready")
    
    # ===== PRE-LOAD ALL AUDIO =====
    if not silent:
        print("[MUSIC] Loading audio...")
    
    sample_rate = 44100
    duration = 30.0  # 30 seconds per track
    crossfade_duration = 2.0  # 2 second crossfade
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    audio_segments = []
    for item in playlist:
        try:
            loader = MonoLoader(filename=item['path'], sampleRate=sample_rate)
            audio = loader()
            
            # Find first onset for better start point
            try:
                from essentia.standard import OnsetRate
                onset_rate = OnsetRate()
                onsets, _ = onset_rate(audio)
                start_time = float(onsets[0]) if len(onsets) > 0 else 0.0
            except:
                start_time = 0.0
            
            start_sample = int(start_time * sample_rate)
            # Get 30 seconds + crossfade buffer
            end_sample = start_sample + int((duration + crossfade_duration) * sample_rate)
            
            if start_sample >= len(audio):
                start_sample = 0
            end_sample = min(end_sample, len(audio))
            
            segment = audio[start_sample:end_sample]
            audio_segments.append(segment)
        except Exception as e:
            if not silent:
                print(f"  ❌ Error loading {item['filename']}: {e}")
            audio_segments.append(np.zeros(int(duration * sample_rate)))
    
    if not silent:
        print("✅ Ready!")
    
    # ===== PLAYBACK WITH SMOOTH CROSSFADE =====
    def music_loop():
        try:
            import numpy as np
            import sounddevice as sd
            from pythonosc import udp_client
            
            # FADE PARAMETERS - longer for smoother transitions
            fade_in_sec = 2.0    # 2 second fade in
            fade_out_sec = 3.0   # 3 second fade out
            scene_duration = 30.0  # Each scene is exactly 30 seconds
            
            for idx, (item, segment) in enumerate(zip(playlist, audio_segments)):
                # Ensure we have enough audio for 30 seconds + fade buffer
                total_needed = int((scene_duration + fade_out_sec) * sample_rate)
                
                if len(segment) < total_needed:
                    # Pad with silence if needed
                    audio_full = np.zeros(total_needed)
                    audio_full[:len(segment)] = segment
                else:
                    audio_full = segment[:total_needed].copy()
                
                # Apply fade in (longer for smooth entry)
                fade_in_samples = int(fade_in_sec * sample_rate)
                if idx == 0:
                    # First track: quick fade in to start fast
                    fade_in_samples = int(0.3 * sample_rate)  # 300ms
                fade_in = np.linspace(0, 1, fade_in_samples)
                audio_full[:fade_in_samples] *= fade_in
                
                # Apply fade out at end
                fade_out_samples = int(fade_out_sec * sample_rate)
                if idx < len(playlist) - 1:
                    # Gradual fade out starting before scene ends
                    fade_start = int((scene_duration - fade_out_sec) * sample_rate)
                    fade_out = np.linspace(1, 0, fade_out_samples)
                    if fade_start >= 0:
                        audio_full[fade_start:fade_start + fade_out_samples] *= fade_out
                else:
                    # Last track: fade out at end
                    fade_start = int((scene_duration - fade_out_sec) * sample_rate)
                    fade_out = np.linspace(1, 0, fade_out_samples)
                    if fade_start >= 0:
                        audio_full[fade_start:fade_start + fade_out_samples] *= fade_out
                    audio_full = audio_full[:int(scene_duration * sample_rate)]
                
                # Play audio IMMEDIATELY
                sd.play(audio_full, sample_rate)
                
                # Print scene info after audio starts
                print(f"[SCENE {item['scene']}/{len(playlist)}] {item['filename'][:35]}...")
                
                # Wait exactly 30 seconds for this scene
                time.sleep(scene_duration)
                
                # For last track, wait for fade to finish
                if idx == len(playlist) - 1:
                    time.sleep(1.0)
            
            sd.stop()
            print("✅ Music finished.")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sd.stop()
    
    # Start thread - music starts IMMEDIATELY
    music_thread = threading.Thread(target=music_loop, daemon=False)
    music_thread.start()
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
    # Parse music parameters FIRST (fast)
    music_params = parse_music_params_from_gemini()
    
    # Start music player - audio starts IMMEDIATELY
    music_thread = start_music_player_thread(music_params, silent=True)
    
    if music_thread:
        # Print info AFTER audio has started
        time.sleep(0.1)  # Small delay to ensure audio is playing
        print(f"🎵 Playing {len(music_params)} scenes (30s each)")
        
        try:
            music_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
            try:
                import sounddevice as sd
                sd.stop()
            except:
                pass
    else:
        print("Music player not available")


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
            # Any other argument means run pipeline (e.g., image path, age)
            run_pipeline()
    else:
        run_pipeline()
