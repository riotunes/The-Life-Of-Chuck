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


# Output directories
CAPTURES_DIR = "captures"
AGED_DIR = "aged_outputs"
UV_DIR = "uv_outputs"



def clear_directory(directory):
    """Remove all files in a directory, create it if it doesn't exist."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"Cleared: {directory}/")


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


if __name__ == "__main__":
    run_pipeline()
