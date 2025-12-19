import cv2
import numpy as np
import trimesh
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import os
import sys

class CanonicalFaceUVConverter:
    """
    Convert face images to UV texture maps using MediaPipe's canonical face mesh 
    and High-Quality Affine Warping.
    """
    
    def __init__(self, model_path="face_landmarker.task", obj_path="canonical_face_model.obj"):
        """
        Initialize MediaPipe and load the Reference OBJ.
        """
        # 1. Load MediaPipe Face Landmarker
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 2. Load Reference OBJ (Crucial for UV topology)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
            
        print(f"Loading reference mesh: {obj_path}...")
        # process=False prevents trimesh from re-ordering vertices
        self.mesh = trimesh.load(obj_path, process=False)
        self.uvs = self.mesh.visual.uv
        self.faces = self.mesh.faces  # The triangles (e.g., [0, 1, 2])

        # Validation
        if len(self.uvs) != 468:
            print(f"WARNING: OBJ has {len(self.uvs)} vertices. MediaPipe expects 468.")

    def get_face_landmarks(self, image):
        """
        Detect facial landmarks using MediaPipe.
        Returns: numpy array of (x, y) coordinates, or None.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert to pixel coordinates (take first 468 points)
        landmarks = np.array([
            [lm.x * w, lm.y * h] 
            for lm in face_landmarks[:468]
        ], dtype=np.float32)
        
        return landmarks
    
    def create_uv_texture(self, image, output_size=(2048, 2048)):
        """
        Generates the UV texture using Affine Warping (High Quality).
        """
        # 1. Get Landmarks
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            return None
        
        w_out, h_out = output_size
        
        # 2. Prepare Output Canvas (Black background)
        texture_map = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        
        # 3. Process Every Triangle in the Mesh
        for tri_idx in self.faces:
            try:
                # --- A. Get Coordinates ---
                # Source: Triangle on the photo face
                src_tri = landmarks[tri_idx]
                
                # Destination: Triangle on the UV map
                dst_tri_uv = self.uvs[tri_idx]
                dst_tri = np.zeros_like(dst_tri_uv)
                
                # U -> X
                dst_tri[:, 0] = dst_tri_uv[:, 0] * w_out
                # V -> Y (Inverted: (1 - V) to fix "Upside Down" issue)
                dst_tri[:, 1] = (1.0 - dst_tri_uv[:, 1]) * h_out
                
                # --- B. Crop Bounding Boxes (Optimization) ---
                r1 = cv2.boundingRect(src_tri.astype(np.float32))
                r2 = cv2.boundingRect(dst_tri.astype(np.float32))
                
                (x1, y1, w1, h1) = r1
                (x2, y2, w2, h2) = r2

                # Validate bounds
                if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: continue
                
                # Crop source
                img1_cropped = image[y1:y1+h1, x1:x1+w1]
                if img1_cropped.size == 0: continue

                # --- C. Compute Affine Transform ---
                # Offset points to be relative to the crop boxes
                src_tri_cropped = src_tri - np.array([x1, y1])
                dst_tri_cropped = dst_tri - np.array([x2, y2])
                
                warp_mat = cv2.getAffineTransform(
                    src_tri_cropped.astype(np.float32), 
                    dst_tri_cropped.astype(np.float32)
                )

                # --- D. Warp and Mask ---
                img2_cropped = cv2.warpAffine(
                    img1_cropped, 
                    warp_mat, 
                    (w2, h2), 
                    flags=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REFLECT_101
                )

                # Create triangular mask
                mask = np.zeros((h2, w2, 3), dtype=np.float32)
                cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), (1.0, 1.0, 1.0))

                # --- E. Blend into Output ---
                # Ensure we are within bounds of the main texture map
                if y2+h2 > h_out or x2+w2 > w_out: continue
                
                current_area = texture_map[y2:y2+h2, x2:x2+w2].astype(np.float32) / 255.0
                warped_area = img2_cropped.astype(np.float32) / 255.0
                
                final_area = (warped_area * mask) + (current_area * (1.0 - mask))
                texture_map[y2:y2+h2, x2:x2+w2] = (final_area * 255).astype(np.uint8)
            
            except Exception:
                continue

        # 4. Add Alpha Channel
        gray = cv2.cvtColor(texture_map, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(texture_map)
        
        # Result is BGRA
        texture_rgba = cv2.merge([b, g, r, alpha])
        
        return texture_rgba
    
    def close(self):
        """Release resources if needed."""
        if hasattr(self, 'detector'):
            self.detector.close()

def convert_images_to_uv(
    input_dir,
    output_dir="uv_outputs",
    output_size=(2048, 2048),
    model_path="face_landmarker.task",
    obj_path="canonical_face_model.obj"
):
    """
    Batch convert images in directory to UV textures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Converter
    try:
        converter = CanonicalFaceUVConverter(model_path, obj_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

    output_paths = []
    input_path = Path(input_dir)
    
    # Gather images
    extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp"}
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images. Starting High-Quality UV conversion...")
    print("-" * 50)
    
    for i, image_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Error: Could not load {image_file}")
            continue
        
        # Generate UV
        uv_texture = converter.create_uv_texture(image, output_size)
        
        if uv_texture is None:
            print(f"  Warning: No face detected in {image_file.name}")
            continue
        
        # Save
        uv_filename = f"uv_{image_file.stem}.png"
        save_path = os.path.join(output_dir, uv_filename)
        cv2.imwrite(save_path, uv_texture)
        output_paths.append(save_path)
        print(f"  ✓ Saved: {uv_filename}")
    
    converter.close()
    print("-" * 50)
    print(f"✓ Batch processing complete. {len(output_paths)} files generated.")
    return output_paths

# --- MAIN CONFIGURATION ---
if __name__ == "__main__":
    
    # INPUT DIRECTORY (Where your aged photos are)
    INPUT_DIR = "aged_outputs"
    
    # OUTPUT DIRECTORY
    OUTPUT_DIR = "uv_outputs"
    
    # MODEL PATHS (Ensure these files exist in the same folder)
    OBJ_PATH = "canonical_face_model.obj"
    MODEL_PATH = "face_landmarker.task"
    
    # Run
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
    else:
        convert_images_to_uv(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            output_size=(2048, 2048), # High resolution
            model_path=MODEL_PATH,
            obj_path=OBJ_PATH
        )