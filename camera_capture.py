# camera_capture.py

import cv2
import os
from datetime import datetime

def capture_face(output_dir="captures", preview=True):
    """
    Capture a photo from the webcam.
    
    Args:
        output_dir: Folder to save the captured image
        preview: If True, show live preview window
        
    Returns:
        Path to the saved image, or None if cancelled
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    print("Camera opened successfully!")
    print("Press SPACE to capture, ESC to cancel")
    
    captured_path = None
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        if preview:
            # Display instructions on the preview
            display_frame = frame.copy()
            cv2.putText(
                display_frame, 
                "SPACE: Capture | ESC: Cancel", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            cv2.imshow("Life of Chuck - Capture", display_frame)
        
        # Wait for key press (1ms delay for smooth preview)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("Capture cancelled")
            break
            
        elif key == 32:  # SPACE key
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_face_{timestamp}.jpg"
            captured_path = os.path.join(output_dir, filename)
            
            # Save the frame
            cv2.imwrite(captured_path, frame)
            print(f"Image saved to: {captured_path}")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_path


# Run if executed directly
if __name__ == "__main__":
    result = capture_face()
    if result:
        print(f"\nSuccess! Captured image: {result}")
    else:
        print("\nNo image captured")