"""
Process Coordinator for Life of Chuck

Coordinates two independent processes:
1. life_of_chuck.py (GUI) - collects user data and generates AI biography
2. main.py (aging pipeline) - generates aged images and UV textures

When BOTH processes complete, sends OSC "start" message to TouchDesigner.
"""

import time
from pathlib import Path

# Flag file locations (in project root)
PROJECT_ROOT = Path(__file__).parent
FLAGS_DIR = PROJECT_ROOT / ".process_flags"
GUI_COMPLETE_FLAG = FLAGS_DIR / "gui_complete"
PIPELINE_COMPLETE_FLAG = FLAGS_DIR / "pipeline_complete"


def init_flags():
    """Initialize/reset the flags directory. Call at start of session."""
    FLAGS_DIR.mkdir(exist_ok=True)

    # Clear any existing flags
    if GUI_COMPLETE_FLAG.exists():
        GUI_COMPLETE_FLAG.unlink()
    if PIPELINE_COMPLETE_FLAG.exists():
        PIPELINE_COMPLETE_FLAG.unlink()

    print("[Coordinator] Flags initialized")


def signal_gui_complete():
    """Call when life_of_chuck.py GUI process is complete."""
    FLAGS_DIR.mkdir(exist_ok=True)
    GUI_COMPLETE_FLAG.touch()
    print("[Coordinator] GUI marked complete")
    _check_and_send_start()


def signal_pipeline_complete():
    """Call when main.py aging pipeline is complete."""
    FLAGS_DIR.mkdir(exist_ok=True)
    PIPELINE_COMPLETE_FLAG.touch()
    print("[Coordinator] Pipeline marked complete")
    _check_and_send_start()


def is_gui_complete():
    """Check if GUI has signaled completion."""
    return GUI_COMPLETE_FLAG.exists()


def is_pipeline_complete():
    """Check if pipeline has signaled completion."""
    return PIPELINE_COMPLETE_FLAG.exists()


def both_complete():
    """Check if both processes have completed."""
    return is_gui_complete() and is_pipeline_complete()


def _check_and_send_start():
    """Internal: Check if both done, and send OSC start if so (in background thread)."""
    if both_complete():
        # Send OSC in separate thread so it doesn't block audio start
        def send_osc_delayed():
            print("[Coordinator] Both processes complete! Sending OSC start immediately...")
            _send_osc_start(msg=0)
            _send_osc_start(msg=1)
            _send_num_stages_to_td()
            _cleanup_flags()

        import threading
        osc_thread = threading.Thread(target=send_osc_delayed, daemon=True)
        osc_thread.start()


def _send_osc_start(ip="127.0.0.1", port=9000, msg=None):
    """Send the OSC 'start' message to TouchDesigner."""
    try:
        from pythonosc import udp_client

        client = udp_client.SimpleUDPClient(ip, port)
        client.send_message("/touchdesigner/start", msg)

        print(f"[Coordinator] OSC sent: /touchdesigner/start -> {ip}:{port}")
        return True

    except ImportError:
        print("[Coordinator] Warning: pythonosc not installed")
        return False
    except Exception as e:
        print(f"[Coordinator] Error sending OSC: {e}")
        return False


def _send_num_stages_to_td(ip="127.0.0.1", port=9001):
    """Send the number of life stages to TouchDesigner on port 9001."""
    try:
        from osc_sender import send_num_stages
        from shared_config import get_num_stages_from_user_data

        # Get number of stages based on user's age
        num_stages = get_num_stages_from_user_data(
            user_data_path=str(PROJECT_ROOT / "user_data.txt"),
            default_stages=5
        )

        # Send to TouchDesigner
        send_num_stages(num_stages, ip=ip, port=port)

        return True

    except ImportError as e:
        print(f"[Coordinator] Warning: Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"[Coordinator] Error sending num_stages: {e}")
        return False


def _cleanup_flags():
    """Remove flag files after successful coordination."""
    try:
        if GUI_COMPLETE_FLAG.exists():
            GUI_COMPLETE_FLAG.unlink()
        if PIPELINE_COMPLETE_FLAG.exists():
            PIPELINE_COMPLETE_FLAG.unlink()
        print("[Coordinator] Flags cleaned up")
    except Exception as e:
        print(f"[Coordinator] Warning: Could not cleanup flags: {e}")


# Optional: Blocking wait function
def wait_for_both(timeout=300, check_interval=0.5):
    """
    Block until both processes complete or timeout.

    Args:
        timeout: Maximum seconds to wait (default 5 minutes)
        check_interval: Seconds between checks

    Returns:
        True if both completed, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if both_complete():
            return True
        time.sleep(check_interval)

    return False


def wait_for_pipeline_only(timeout=300, check_interval=0.5):
    """
    Wait only for pipeline completion (used by GUI after text generation).

    Args:
        timeout: Maximum seconds to wait (default 5 minutes)
        check_interval: Seconds between checks

    Returns:
        True if pipeline completed, False if timeout
    """
    start_time = time.time()

    print("[Coordinator] Waiting for aging pipeline to complete...")

    while time.time() - start_time < timeout:
        if is_pipeline_complete():
            print("[Coordinator] Pipeline is complete!")
            return True
        time.sleep(check_interval)

    print("[Coordinator] Timeout waiting for pipeline")
    return False


if __name__ == "__main__":
    # Test/debug mode
    print("Process Coordinator - Debug Mode")
    print(f"Flags directory: {FLAGS_DIR}")
    print(f"GUI complete: {is_gui_complete()}")
    print(f"Pipeline complete: {is_pipeline_complete()}")
    print(f"Both complete: {both_complete()}")
