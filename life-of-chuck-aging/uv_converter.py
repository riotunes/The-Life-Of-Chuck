# improved_uv_converter.py

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import os


class CanonicalFaceUVConverter:
    """Convert face images to UV texture maps using MediaPipe's canonical face mesh."""
    
    def __init__(self, model_path="face_landmarker.task"):
        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # MediaPipe face mesh triangulation (predefined topology)
        self.face_triangles = self._get_face_mesh_triangles()
    
    def _get_face_mesh_triangles(self):
        """
        Get MediaPipe's canonical face mesh triangulation.
        Returns a subset of reliable triangles for texture mapping.
        """
        # These are a subset of MediaPipe's face mesh triangles
        # Selected for reliability and coverage
        return [
            # Face center and cheeks
            [234, 93, 132], [132, 93, 58], [58, 172, 132], [172, 136, 150],
            [150, 149, 176], [176, 148, 152], [152, 377, 400], [400, 378, 379],
            [379, 365, 397], [397, 288, 361], [361, 323, 454], [454, 356, 389],
            
            # Forehead
            [10, 338, 297], [297, 332, 284], [284, 251, 389], [162, 127, 234],
            [234, 127, 93], [10, 109, 67], [67, 103, 54], [54, 21, 162],
            
            # Nose
            [168, 6, 197], [197, 195, 5], [5, 4, 1], [1, 19, 94],
            [94, 2, 164], [164, 393, 391], [391, 322, 410],
            
            # Eyes
            [133, 173, 157], [157, 158, 159], [159, 160, 161], [161, 246, 33],
            [362, 398, 384], [384, 385, 386], [386, 387, 388], [388, 466, 263],
            
            # Mouth region
            [61, 146, 91], [91, 181, 84], [84, 17, 314], [314, 405, 321],
            [321, 375, 291], [61, 185, 40], [40, 39, 37], [37, 0, 267],
            [267, 269, 270], [270, 409, 291],
            
            # Jaw
            [172, 136, 150], [150, 149, 176], [176, 148, 152], [152, 377, 400],
            [58, 132, 93], [93, 234, 127], [127, 162, 21], [21, 54, 103],
        ]
    
    def get_face_landmarks(self, image):
        """
        Detect facial landmarks using MediaPipe.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            numpy array of (x, y) coordinates, or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect faces
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.face_landmarks[0]
        
        h, w = image.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h] 
            for lm in face_landmarks
        ], dtype=np.float32)
        
        return landmarks
    
    def create_uv_texture_simple(
        self, 
        image, 
        output_size=(1024, 1024)
    ):
        """
        Create UV texture using simple perspective transform.
        More reliable than triangulation for basic UV mapping.
        
        Args:
            image: BGR image (OpenCV format)
            output_size: Size of output UV texture (width, height)
            
        Returns:
            Tuple of (UV texture, UV coordinates, landmarks)
        """
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            return None, None, None
        
        h, w = image.shape[:2]
        
        # Calculate face bounds with padding
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        
        # Add 15% padding
        padding = 0.15
        width = max_x - min_x
        height = max_y - min_y
        
        min_x = max(0, min_x - width * padding)
        min_y = max(0, min_y - height * padding)
        max_x = min(w, max_x + width * padding)
        max_y = min(h, max_y + height * padding)
        
        # Source points (face bounding box)
        src_points = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype=np.float32)
        
        # Destination points (full output image)
        dst_points = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp the image
        uv_texture = cv2.warpPerspective(
            image,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Transform landmarks to UV space
        ones = np.ones((len(landmarks), 1))
        landmarks_homogeneous = np.hstack([landmarks, ones])
        uv_coords_homogeneous = matrix @ landmarks_homogeneous.T
        uv_coords = (uv_coords_homogeneous[:2] / uv_coords_homogeneous[2]).T
        uv_coords = uv_coords.astype(np.float32)
        
        return uv_texture, uv_coords, landmarks
    
    def create_uv_texture_mesh(
        self, 
        image, 
        output_size=(1024, 1024)
    ):
        """
        Create UV texture using triangle-based warping with predefined mesh.
        
        Args:
            image: BGR image (OpenCV format)
            output_size: Size of output UV texture (width, height)
            
        Returns:
            Tuple of (UV texture, UV coordinates, landmarks)
        """
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            return None, None, None
        
        h, w = image.shape[:2]
        
        # Calculate face bounds
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        
        padding = 0.15
        width = max_x - min_x
        height = max_y - min_y
        
        min_x = max(0, min_x - width * padding)
        min_y = max(0, min_y - height * padding)
        max_x = min(w, max_x + width * padding)
        max_y = min(h, max_y + height * padding)
        
        # Map landmarks to UV space
        uv_coords = (landmarks - [min_x, min_y]) / [max_x - min_x, max_y - min_y]
        uv_coords = uv_coords * output_size
        uv_coords = uv_coords.astype(np.float32)
        
        # Create output texture
        uv_texture = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        # Warp each triangle
        for tri_indices in self.face_triangles:
            try:
                # Check if all indices are valid
                if any(idx >= len(landmarks) for idx in tri_indices):
                    continue
                
                # Source triangle
                src_tri = landmarks[tri_indices].astype(np.float32)
                
                # Destination triangle
                dst_tri = uv_coords[tri_indices].astype(np.float32)
                
                # Get bounding rectangles
                src_rect = cv2.boundingRect(src_tri)
                dst_rect = cv2.boundingRect(dst_tri)
                
                if src_rect[2] <= 0 or src_rect[3] <= 0:
                    continue
                if dst_rect[2] <= 0 or dst_rect[3] <= 0:
                    continue
                
                # Offset triangles
                src_tri_offset = src_tri - [src_rect[0], src_rect[1]]
                dst_tri_offset = dst_tri - [dst_rect[0], dst_rect[1]]
                
                # Extract source region
                x, y, w_r, h_r = src_rect
                if x < 0 or y < 0 or x + w_r > w or y + h_r > h:
                    continue
                
                src_crop = image[y:y + h_r, x:x + w_r].copy()
                
                if src_crop.size == 0:
                    continue
                
                # Warp triangle
                warp_mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset)
                dst_crop = cv2.warpAffine(
                    src_crop,
                    warp_mat,
                    (dst_rect[2], dst_rect[3]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                
                # Create mask
                mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.float32)
                cv2.fillConvexPoly(mask, dst_tri_offset.astype(np.int32), 1.0)
                
                # Apply to output
                dx, dy, dw, dh = dst_rect
                if dx < 0 or dy < 0 or dx + dw > output_size[0] or dy + dh > output_size[1]:
                    continue
                
                roi = uv_texture[dy:dy + dh, dx:dx + dw]
                if roi.shape[:2] != dst_crop.shape[:2]:
                    continue
                
                # Blend
                mask_3ch = np.stack([mask, mask, mask], axis=2)
                uv_texture[dy:dy + dh, dx:dx + dw] = (
                    dst_crop.astype(np.float32) * mask_3ch +
                    roi.astype(np.float32) * (1 - mask_3ch)
                ).astype(np.uint8)
                
            except:
                continue
        
        return uv_texture, uv_coords, landmarks
    
    def create_masked_uv(
        self, 
        image, 
        output_size=(1024, 1024),
        feather_amount=10,
        use_mesh=True
    ):
        """
        Create a UV texture with alpha mask.
        
        Args:
            image: BGR image (OpenCV format)
            output_size: Size of output UV texture
            feather_amount: Pixels to feather the mask edge
            use_mesh: Use mesh-based warping (slower but more accurate)
            
        Returns:
            RGBA image with mask applied, or None if no face detected
        """
        # Choose method
        if use_mesh:
            uv_texture, uv_coords, landmarks = self.create_uv_texture_mesh(image, output_size)
        else:
            uv_texture, uv_coords, landmarks = self.create_uv_texture_simple(image, output_size)
        
        if uv_texture is None:
            return None
        
        # Create mask from UV coordinates
        mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
        
        # Use convex hull for mask
        hull = cv2.convexHull(uv_coords.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Feather the edges
        if feather_amount > 0:
            kernel_size = feather_amount * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Convert to BGRA and apply mask
        bgra = cv2.cvtColor(uv_texture, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        
        return bgra
    
    def close(self):
        """Release resources."""
        pass


def convert_images_to_uv(
    input_dir,
    output_dir="uv_outputs",
    output_size=(1024, 1024),
    feather_amount=10,
    model_path="face_landmarker.task",
    use_mesh=False
):
    """
    Convert all face images in a directory to UV textures.
    
    Args:
        input_dir: Directory containing face images
        output_dir: Directory to save UV textures
        output_size: Size of output textures (width, height)
        feather_amount: Pixels to feather mask edges
        model_path: Path to MediaPipe face landmarker model
        use_mesh: Use mesh-based warping (False = simple transform, faster)
        
    Returns:
        List of paths to generated UV textures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Download it with:")
        print("  curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return []
    
    converter = CanonicalFaceUVConverter(model_path)
    output_paths = []
    
    # Get all jpg and png images in input directory
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    
    method = "mesh triangulation" if use_mesh else "perspective transform"
    print(f"Converting {len(image_files)} images to UV textures using {method}...")
    print("-" * 40)
    
    for image_file in image_files:
        print(f"Processing: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Error: Could not load {image_file}")
            continue
        
        # Generate masked UV texture (RGBA)
        masked_uv = converter.create_masked_uv(
            image, 
            output_size, 
            feather_amount,
            use_mesh=use_mesh
        )
        
        if masked_uv is None:
            print(f"  Error: No face detected in {image_file.name}")
            continue
        
        # Save masked UV texture as PNG (preserves alpha)
        uv_filename = f"uv_{image_file.stem}.png"
        uv_path = os.path.join(output_dir, uv_filename)
        cv2.imwrite(uv_path, masked_uv)
        output_paths.append(uv_path)
        print(f"  ✓ Saved: {uv_filename}")
    
    converter.close()
    
    print("-" * 40)
    print(f"✓ Converted {len(output_paths)}/{len(image_files)} images to UV textures")
    
    return output_paths


# Run if executed directly
if __name__ == "__main__":
    import sys
    
    # Default input directory
    input_dir = "aged_outputs"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found")
        print("Please run face_aging.py first to generate aged images")
        sys.exit(1)
    
    # Convert to UV textures
    results = convert_images_to_uv(
        input_dir=input_dir,
        output_dir="uv_outputs",
        output_size=(1024, 1024),
        feather_amount=15,
        use_mesh=False  # Set to True for mesh-based warping
    )
    
    if results:
        print("\n✓ Generated UV textures:")
        for path in results:
            print(f"  {path}")
    else:
        print("\n✗ No UV textures were generated")