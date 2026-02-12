# 🎹 The Life of Chuck: Multimodal Aging Installation

## 🚀 Project Overview
**The Life of Chuck** is an interactive multimedia installation that explores the trajectory of a human life through data, vision, and sound. By capturing a user's current age, the system utilizes a probabilistic model to determine a unique "life path," generating a synthetic AI biography alongside a visual aging transformation of the user's own face.

## 🏗️ System Architecture
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

## 📦 Installation & Setup

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
