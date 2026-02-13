# The Life of Chuck: Multimodal Aging Installation

<p align="center">
  <img src="LIFE%20OF%20CHUCK.png" width="800">
</p>

## Project Overview
## What is this?
**The Life of Chuck** is a deep dive into the blurry line between us and our machines. We usually think of AI as a tool we control, but this project flips that: **If we influence the AI, can the AI influence us back?**

We’re looking at the interaction between a human and an algorithm to see where that relationship actually leads. Can an AI "see" your journey before it happens? Can it guide a person or indicate where their life is headed?

## The Concept
The project is built on a few core provocations:

* **The Feedback Loop:** Your data feeds the machine, but the machine’s output feeds your perception of yourself. It's a recursive cycle.
* **The AI as a Guide:** We explore if an AI can actually map out a life journey, projecting a synthetic future that feels real enough to change the present.
* **Machine Agency:** We’re testing if the "indicated journey" provided by the AI can actually nudge a human’s path.

  
## System Architecture
The installation operates through a coordinated pipeline of three main engines:

1.  **Python Backend:** Handles the GUI workflow, AI text generation (**Gemini 2.0 Flash**), progressive face aging (**Gemini 3 Pro Image**), and audio feature analysis.
2.  **Audio Playback Layer:** A **SuperCollider** engine (`player_crossfade.scd`) that manages real-time audio crossfades and queue management based on emotional track matching.
3.  **Visualization Layer:** A **TouchDesigner** environment that renders 3D face meshes using generated UV textures, synchronized with AI narratives and audio.



## ✨ Key Features
* **Progressive Aging Pipeline:** Uses a "chained generation" strategy where each 10-year aging result becomes the input for the next, preserving identity markers across life stages.
* **UV Texture Conversion:** Converts 2D AI portraits into 2048x2048 UV maps using **MediaPipe** 468-point landmark detection and affine triangle warping.
* **Intelligent Music Selection:** Uses **Essentia** and **TensorFlow** to match tracks to life stages by calculating the Euclidean distance between Arousal, Valence, and BPM parameters.
* **Flag-Based Coordination:** Uses `process_coordinator.py` to synchronize parallel threads (AI text, Image generation, and Music analysis) via file-based flags for non-blocking performance.

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Core Logic** | Python 3.13 |
| **AI / Vision** | MediaPipe, Google Gemini 3 Pro Image, Gemini 2.0 Flash |
| **Music Analysis** | Essentia-Tensorflow (RhythmExtractor2013) |
| **Graphics** | TouchDesigner |
| **Audio Engine** | SuperCollider |
| **Protocols** | OSC (Ports 9000, 9001, 57120) |

## Installation & Setup

1.  **Clone and Install:**
    ```bash
    git clone [https://github.com/riotunes/CPAC-Hackaton.git](https://github.com/riotunes/CPAC-Hackaton.git)
    cd CPAC-Hackaton
    pip install -r requirements.txt
    pip install essentia-tensorflow
    ```

2.   **Configure the Environment:** 
    Create a `.env` file in the root directory of the project and add your Gemini API key:
    ```env
    GEMINI_API_KEY=your_api_key_here 
    ```
## 2. Initialize Audio

3. Boot **SuperCollider** and run the file: 
    ```bash 
    player_crossfade.scd
    ```

4. Open and run the file:
    ```bash
    python life_of_chuck.py 
    ```

## 📂 Project Core Structure
* `life_of_chuck.py`: Main GUI application (Tkinter)
* `main.py`: Handles landmark detection and affine warping to UV space.
* `process_coordinator.py`: Manages the non-blocking flag synchronization between modules
* `musica/music_score.py`: Analyzes audio library and builds the emotional matching cache.
* `camera_to_uv.py`: Handles landmark detection and affine warping to UV space.
---

##  Conclusion

We hope that this journey through the “multitudes” inhabiting your future will not be merely a technological curiosity, but a moment of genuine introspection. We hope you enjoy the experience and, above all, that you feel something: the weight of time, the joy of fulfilled dreams, or the melancholic beauty of change.

---

Authors: Riccardo Tocci, David Gadiaga, Mario Aucelli 
*Created for the CPAC course 2026.*

