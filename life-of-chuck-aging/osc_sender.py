from pythonosc import udp_client
import time

def send_pipeline_complete(
    ip="127.0.0.1",
    port=9000,
    address="/pipeline/complete"
):
    """
    Send OSC message to TouchDesigner indicating pipeline completion.
    
    Args:
        ip: IP address of TouchDesigner (default: localhost)
        port: OSC port TouchDesigner is listening on (default: 9000)
        address: OSC address/route (default: /pipeline/complete)
    """
    try:
        # Create OSC client
        client = udp_client.SimpleUDPClient(ip, port)
        
        # Send the message with value 1 (start flag)
        client.send_message(address, 1)
        
        print(f"✓ OSC message sent to {ip}:{port}{address}")
        return True
        
    except Exception as e:
        print(f"✗ Error sending OSC message: {e}")
        return False

def send_with_metadata(
    ip="127.0.0.1",
    port=9000,
    num_textures=0
):
    """
    Send OSC messages with metadata about the generated content.
    
    Args:
        ip: IP address of TouchDesigner
        port: OSC port
        num_textures: Number of UV textures generated
    """
    try:
        client = udp_client.SimpleUDPClient(ip, port)
        
        # Send completion flag
        client.send_message("/pipeline/complete", 1)
        
        # Send number of textures
        client.send_message("/pipeline/count", num_textures)
        
        # Small delay to ensure messages are received
        time.sleep(0.01)
        
        print(f"✓ OSC messages sent to {ip}:{port}")
        print(f"  - /pipeline/complete: 1")
        print(f"  - /pipeline/count: {num_textures}")
        return True
        
    except Exception as e:
        print(f"✗ Error sending OSC messages: {e}")
        return False