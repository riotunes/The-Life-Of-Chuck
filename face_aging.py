# face_aging.py

import os
import io
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image


def generate_aged_image(client, source_image, current_age, target_age):
    """
    Generate an aged version of the face using Gemini.
    
    Args:
        client: Google GenAI client
        source_image: PIL Image object (previous stage)
        current_age: The age of the input image
        target_age: Target age to generate
        
    Returns:
        PIL Image object or None if failed
    """
    age_difference = target_age - current_age
    
    # Craft the aging prompt
    prompt = f"""Age this person by {age_difference} years, from age {current_age} to age {target_age}.

Apply realistic aging changes for this {age_difference}-year progression:
- Add natural wrinkles and other skin difects (scar bruises) that could develop over {age_difference} years
- Subtle skin texture changes appropriate for age {target_age}
- Natural hair graying/thinning if appropriate
- Slight changes to facial structure (skin elasticity, sagging)

CRITICAL REQUIREMENTS:
1. Preserve the person's identity, expression, pose, lighting, and background exactly
2. Only change features related to aging
3. Make the changes subtle but noticeable - this is a {age_difference}-year progression
4. Output ONLY the transformed image, no text"""

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, source_image],
            config=types.GenerateContentConfig(
                response_modalities=["image", "text"],
            ),
        )
        
        # Extract generated image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Get raw image bytes and convert to PIL Image
                image_bytes = part.inline_data.data
                pil_image = Image.open(io.BytesIO(image_bytes))
                return pil_image
                
        print(f"  Warning: No image in response for age {target_age}")
        return None
        
    except Exception as e:
        print(f"  Error generating age {target_age}: {e}")
        return None


def generate_age_progression(
    input_image_path,
    current_age,
    output_dir="aged_outputs",
    age_increment=10,
    num_stages=7
):
    """
    Generate progressively aged versions of a face.
    Each stage uses the previous stage's output as input (chained generation).
    
    Args:
        input_image_path: Path to the source face image
        current_age: The subject's current age
        output_dir: Folder to save aged images
        age_increment: Years to add per stage (default: 10)
        num_stages: Number of aged versions to generate (default: 5)
        
    Returns:
        List of paths to generated images (including original)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the Gemini client
    client = genai.Client()
    
    # Calculate target ages
    target_ages = [
        current_age + (i * age_increment) 
        for i in range(1, num_stages + 1)
    ]
    
    print(f"Starting age: {current_age}")
    print(f"Target ages: {target_ages}")
    print("-" * 40)
    
    # Load and save original
    original_filename = f"age_{current_age:03d}_original.jpg"
    original_output_path = os.path.join(output_dir, original_filename)
    
    original_image = Image.open(input_image_path)
    # Ensure RGB mode
    if original_image.mode in ("RGBA", "P"):
        original_image = original_image.convert("RGB")
    original_image.save(original_output_path, "JPEG", quality=95)
    
    output_paths = [original_output_path]
    print(f"[{current_age}] Original saved: {original_output_path}")
    
    # Chain generation: each stage uses previous output
    previous_image = original_image
    previous_age = current_age
    
    for target_age in target_ages:
        print(f"[{target_age}] Generating from age {previous_age}...")
        
        aged_image = generate_aged_image(
            client, 
            previous_image, 
            previous_age, 
            target_age
        )
        
        if aged_image:
            # Convert to RGB if necessary
            if aged_image.mode in ("RGBA", "P"):
                aged_image = aged_image.convert("RGB")
            
            filename = f"age_{target_age:03d}.jpg"
            save_path = os.path.join(output_dir, filename)
            aged_image.save(save_path, "JPEG", quality=95)
            output_paths.append(save_path)
            print(f"[{target_age}] Saved: {save_path}")
            
            # Update for next iteration
            previous_image = aged_image
            previous_age = target_age
        else:
            print(f"[{target_age}] Failed to generate - using previous image for next stage")
            # Continue with previous image if generation fails
    
    print("-" * 40)
    print(f"Generated {len(output_paths)} images")
    
    return output_paths


# Run if executed directly
if __name__ == "__main__":
    import sys
    
    # Find the most recent capture
    captures_dir = Path("captures")
    test_image = None
    
    if captures_dir.exists():
        captures = sorted(captures_dir.glob("*.jpg"))
        if captures:
            test_image = str(captures[-1])
    
    if not test_image or not os.path.exists(test_image):
        print("Error: No captured image found")
        print("Please run camera_capture.py first")
        sys.exit(1)
    
    # Get current age from user
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        user_data_path = os.path.join(script_dir, 'user_data.txt')
        with open(user_data_path, 'r') as f:
            content = f.read().strip()
        current_age = int(content.split(':')[1].strip())
    except ValueError:
        print("Invalid age entered")
        sys.exit(1)
    
    print(f"\nUsing image: {test_image}")
    print(f"Will generate ages: {current_age + 10}, {current_age + 20}, {current_age + 30}, {current_age + 40}, {current_age + 50}")
    print("(Each stage builds on the previous one)")
    print()
    
    # Run the age progression
    results = generate_age_progression(
        input_image_path=test_image,
        current_age=current_age,
        age_increment=10,
        num_stages=5
    )
    
    print("\nGenerated images:")
    for path in results:
        print(f"  {path}")
