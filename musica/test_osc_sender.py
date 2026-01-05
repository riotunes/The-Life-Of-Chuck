"""
Test OSC Sender per MusicPlayer
Genera valori casuali e li invia al server OSC.
"""

import random
import time
import argparse

from pythonosc.udp_client import SimpleUDPClient


def generate_random_params():
    """Genera parametri musicali casuali."""
    return {
        'arousal': round(random.uniform(0.0, 1.0), 2),
        'valence': round(random.uniform(0.0, 1.0), 2),
        'bpm': round(random.uniform(60, 180), 1),
        'danceability': round(random.uniform(0.0, 1.0), 2),
        'aggressive': round(random.uniform(0.0, 1.0), 2)
    }


def generate_mood_preset(mood: str):
    """Genera parametri basati su un mood predefinito."""
    presets = {
        'happy': {
            'arousal': round(random.uniform(0.6, 0.9), 2),
            'valence': round(random.uniform(0.7, 1.0), 2),
            'bpm': round(random.uniform(110, 140), 1),
            'danceability': round(random.uniform(0.6, 1.0), 2),
            'aggressive': round(random.uniform(0.1, 0.4), 2)
        },
        'sad': {
            'arousal': round(random.uniform(0.1, 0.4), 2),
            'valence': round(random.uniform(0.0, 0.3), 2),
            'bpm': round(random.uniform(60, 90), 1),
            'danceability': round(random.uniform(0.1, 0.4), 2),
            'aggressive': round(random.uniform(0.0, 0.2), 2)
        },
        'aggressive': {
            'arousal': round(random.uniform(0.7, 1.0), 2),
            'valence': round(random.uniform(0.2, 0.5), 2),
            'bpm': round(random.uniform(120, 160), 1),
            'danceability': round(random.uniform(0.4, 0.8), 2),
            'aggressive': round(random.uniform(0.7, 1.0), 2)
        },
        'relaxed': {
            'arousal': round(random.uniform(0.1, 0.3), 2),
            'valence': round(random.uniform(0.5, 0.8), 2),
            'bpm': round(random.uniform(70, 100), 1),
            'danceability': round(random.uniform(0.2, 0.5), 2),
            'aggressive': round(random.uniform(0.0, 0.2), 2)
        },
        'dance': {
            'arousal': round(random.uniform(0.6, 0.9), 2),
            'valence': round(random.uniform(0.5, 0.8), 2),
            'bpm': round(random.uniform(115, 130), 1),
            'danceability': round(random.uniform(0.8, 1.0), 2),
            'aggressive': round(random.uniform(0.2, 0.5), 2)
        }
    }
    return presets.get(mood, generate_random_params())


def send_osc(client: SimpleUDPClient, address: str, params: dict):
    """Invia i parametri via OSC."""
    client.send_message(address, [
        params['arousal'],
        params['valence'],
        params['bpm'],
        params['danceability'],
        params['aggressive']
    ])


def main():
    parser = argparse.ArgumentParser(description="Test OSC Sender for MusicPlayer")
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                       help='IP del server OSC (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9000,
                       help='Porta OSC (default: 9000)')
    parser.add_argument('--address', type=str, default='/music/play',
                       help='Indirizzo OSC (default: /music/play)')
    parser.add_argument('--mood', type=str, default=None,
                       choices=['happy', 'sad', 'aggressive', 'relaxed', 'dance', 'random'],
                       help='Mood preset (default: random)')
    parser.add_argument('--loop', action='store_true',
                       help='Invia messaggi in loop ogni 35 secondi')
    parser.add_argument('--interval', type=float, default=35.0,
                       help='Intervallo tra messaggi in loop (default: 35s)')
    parser.add_argument('--count', type=int, default=1,
                       help='Numero di messaggi da inviare (default: 1)')
    
    args = parser.parse_args()
    
    # Crea client OSC
    client = SimpleUDPClient(args.ip, args.port)
    
    print("="*50)
    print("🎵 TEST OSC SENDER")
    print("="*50)
    print(f"   Target: {args.ip}:{args.port}")
    print(f"   Address: {args.address}")
    print("="*50)
    
    try:
        if args.loop:
            print(f"\n🔁 Loop mode (ogni {args.interval}s) - Ctrl+C per fermare\n")
            count = 0
            while True:
                count += 1
                if args.mood and args.mood != 'random':
                    params = generate_mood_preset(args.mood)
                else:
                    params = generate_random_params()
                
                print(f"\n[{count}] 📤 Invio messaggio...")
                print(f"    Arousal: {params['arousal']}")
                print(f"    Valence: {params['valence']}")
                print(f"    BPM: {params['bpm']}")
                print(f"    Danceability: {params['danceability']}")
                print(f"    Aggressive: {params['aggressive']}")
                
                send_osc(client, args.address, params)
                print("    ✅ Inviato!")
                
                time.sleep(args.interval)
        else:
            for i in range(args.count):
                if args.mood and args.mood != 'random':
                    params = generate_mood_preset(args.mood)
                else:
                    params = generate_random_params()
                
                print(f"\n[{i+1}/{args.count}] 📤 Invio messaggio...")
                print(f"    Arousal: {params['arousal']}")
                print(f"    Valence: {params['valence']}")
                print(f"    BPM: {params['bpm']}")
                print(f"    Danceability: {params['danceability']}")
                print(f"    Aggressive: {params['aggressive']}")
                
                send_osc(client, args.address, params)
                print("    ✅ Inviato!")
                
                if i < args.count - 1:
                    time.sleep(1)
            
            print("\n✅ Completato!")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Interrotto dall'utente.")


if __name__ == "__main__":
    main()
