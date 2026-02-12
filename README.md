# The Life of Chuck: Multimodal Aging Installation

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

1.  **Biometric Intelligence:** Captures user age via `user_data.txt` and calculates a "stage count" (between 2 and 8 stages) using a life-expectancy curve with random perturbations.
2.  **Vision & Aging Pipeline:** Processes facial landmarks to create aged outputs and UV textures for 3D mapping on the `canonical_face_model.obj`.
3.  **Multimedia Synchronization:** A central coordinator manages file-based flags to sync AI-generated text with SuperCollider audio and TouchDesigner visuals via OSC.



## ✨ Key Features
* **Dynamic Life Modeling:** Uses a Gaussian noise-based model in `shared_config.py` to allow for variability, simulating different life outcomes like "longevity bumps" or "early death".
* **Process Coordination:** Utilizes `process_coordinator.py` to manage independent threads, ensuring the GUI and the aging pipeline complete before triggering the final show.
* **Real-time OSC Communication:** Fully networked via `python-osc`, sending triggers to TouchDesigner (Port 9000) and stage data (Port 9001).
* **Generative Audio:** Integrated SuperCollider scripts (`player.scd`) for real-time soundscapes that react to the aging stages.

## 🛠️ Tech Stack
* **Core Logic:** Python 3.12+
* **AI/Vision:** MediaPipe, OpenCV, Google GenAI
* **3D/Graphics:** Trimesh, TouchDesigner
* **Audio:** SuperCollider
* **Protocol:** OSC (Open Sound Control)

## Installation & Setup

1.  **Clone and Install:**
    ```bash
    git clone [https://github.com/riotunes/CPAC-Hackaton.git](https://github.com/riotunes/CPAC-Hackaton.git)
    cd CPAC-Hackaton
    pip install -r requirements.txt
    ```

2.  **Initialize Audio:**
    Open SuperCollider and boot the server with `player.scd`.

3.  **Run the Installation:**
    ```bash
    python main.py
    ```

## 📂 Project Structure
* `shared_config.py`: Probabilistic life-stage logic.
* `process_coordinator.py`: Syncs the vision pipeline and GUI.
* `face_aging.py`: Handles the visual transformation logic.
* `osc_sender.py`: Manages the communication bridge to external software.

---
*Created for the CPAC Hackathon.*
