import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import sys
import subprocess
import threading
import time
from google import genai
from process_coordinator import init_flags, signal_gui_complete, wait_for_pipeline_only
from shared_config import calculate_num_stages

GEMINI_API_KEY = "AIzaSyAL4NxY6azP6trq8P_RXIApViN_8tvY9_A"

class LifeOfChuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Life of Chuck")
        self.root.geometry("1000x800")
        self.root.configure(bg="black")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bg_folder = os.path.abspath(os.path.join(script_dir, "bg_frames"))
        
        self.bg_frames = []
        self.current_frame = 0
        self.load_background_frames()

        self.vid = cv2.VideoCapture(0)
        self.captured_image = None
        
        self.container = tk.Frame(self.root, bg="black")
        self.container.pack(fill="both", expand=True)
        
        self.pages = {}
        page_list = (StartPage, CameraPage, AgePage , NamePage, DreamPage, BioPage, EndPage)
        
        for PageClass in page_list:
            page_name = PageClass.__name__
            frame = PageClass(parent=self.container, controller=self)
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page("StartPage")
        self.animate_background()

    def start_music_analysis(self):
        """Start music analysis in background thread at app startup."""
        def analyze():
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                main_script = os.path.join(script_dir, "main.py")
                
                print("🎵 Starting background music analysis...")
                # Use Popen instead of run to avoid blocking
                process = subprocess.Popen(
                    ["python", main_script, "--analyze-only"],
                    cwd=script_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                
                # Don't wait for completion, just let it run in background
                print("✅ Music analysis started in background")
            except Exception as e:
                print(f"⚠️ Could not start music analysis: {e}")
        
        threading.Thread(target=analyze, daemon=True).start()

    def load_background_frames(self):
        if os.path.exists(self.bg_folder):
            files = sorted([f for f in os.listdir(self.bg_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in files:
                try:
                    img = Image.open(os.path.join(self.bg_folder, f))
                    img = img.resize((1000, 800), Image.Resampling.LANCZOS)
                    self.bg_frames.append(ImageTk.PhotoImage(img))
                except:
                    continue

    def animate_background(self):
        if self.bg_frames:
            next_img = self.bg_frames[self.current_frame]
            for page in self.pages.values():
                page.canvas.itemconfig(page.bg_image_item, image=next_img)
            self.current_frame = (self.current_frame + 1) % len(self.bg_frames)
        self.root.after(120, self.animate_background)

    def show_page(self, page_name):
        frame = self.pages[page_name]
        frame.tkraise()
        if page_name == "CameraPage":
            frame.start_webcam()
        if page_name == "EndPage":
            frame.generate_future_timeline()

    def fetch_gemini_bio(self, callback):
        def run():
            try:
                # Setup target directory paths
                script_dir = os.path.dirname(os.path.abspath(__file__))
                target_dir = os.path.join(script_dir, "futures_texts")

                # Ensure the 'futures' folder exists
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                user_data_path = os.path.join(script_dir, "user_data.txt")
                with open(user_data_path, "r", encoding="utf-8") as f:
                    data = f.read()
                    
                import re
                age_match = re.search(r"AGE: (\d+)", data)
                current_age = int(age_match.group(1)) if age_match else 30

                # Calculate number of stages dynamically based on age
                num_stages = calculate_num_stages(current_age)
                num_age_blocks = num_stages  # All blocks represent future ages, last one is DEATH

                client = genai.Client(api_key=GEMINI_API_KEY)

                # PROMPT: Focus su lunghezza minima e virgola dopo ogni parola + parametri musicali
                prompt = f"""
                Dati Utente: {data}
                Età attuale: {current_age}

                OBJECTIVE:
                Generate exactly {num_stages} raw image prompts for a generative AI WITH MUSIC PARAMETERS.
                NO SENTENCES. NO STORY. NO VERBS.

                STRUCTURE:
                - Exactly {num_stages} blocks: [AGE X] ({num_age_blocks - 1} random ages) and [DEATH].
                - start from the current age, never go below it.
                - Each block MUST have TWO parts: IMAGE PROMPT and MUSIC PARAMETERS

                STRICT FORMAT RULES:
                1. WORDS: Each image prompt must have EXACTLY 12 words.
                2. CONTENT: Only visual nouns and adjectives.
                3. BANNED: No 'is', 'has', 'the', 'with', 'a', 'of', 'in', 'to'. No verbs.
                4. PUNCTUATION: Every single word MUST be followed by a comma.
                5. NO PERIODS: Zero dots. Only commas.
                6. FOCUS: Lighting, materials, atmosphere, specific objects.
                7. REALISM: Include decay, shadows, and gritty details.
                8. MUSIC: After image prompt, add a line with MUSIC: arousal,valence,bpm,danceability,aggressive

                MUSIC PARAMETERS (all values 0.0-1.0 except BPM which is 60-180):
                - arousal: energy/activation level (0=calm, 1=intense)
                - valence: positivity (0=sad/negative, 1=happy/positive)
                - bpm: tempo in beats per minute (60-180)
                - danceability: rhythmic quality (0=not danceable, 1=very danceable)
                - aggressive: tension/intensity (0=gentle, 1=aggressive)

                EXAMPLE OF DESIRED OUTPUT:
                [AGE 42]
                Rusty, metal, flickering, neon, blue, rain, wet, pavement, silhouette, cinematic, fog, smoke,
                MUSIC: 0.6,0.4,100,0.5,0.3

                """
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt
                )
                
                if response.text:
                    full_story = response.text

                    # Pulizia file precedenti nel target_dir
                    for f_old in os.listdir(target_dir):
                        if f_old.startswith("future_") and f_old.endswith(".txt"):
                            os.remove(os.path.join(target_dir, f_old))

                    # Create music_params folder
                    music_params_dir = os.path.join(script_dir, "music_params")
                    if not os.path.exists(music_params_dir):
                        os.makedirs(music_params_dir)
                    
                    # Clean old music params
                    for f_old in os.listdir(music_params_dir):
                        if f_old.endswith(".txt"):
                            os.remove(os.path.join(music_params_dir, f_old))

                    parts = full_story.split('[')
                    for part in parts:
                        if ']' in part:
                            header, content = part.split(']', 1)
                            clean_header = header.strip().lower().replace(' ', '_')
                            
                            # Separate image prompt from music params
                            lines = content.strip().split('\n')
                            image_lines = []
                            music_line = None
                            
                            for line in lines:
                                line = line.strip()
                                if line.startswith('MUSIC:'):
                                    music_line = line
                                elif line:
                                    image_lines.append(line)
                            
                            # Save image prompt to futures_texts
                            image_filename = os.path.join(target_dir, f"future_{clean_header}.txt")
                            image_content = '\n'.join(image_lines)
                            with open(image_filename, "w", encoding="utf-8") as f:
                                f.write(image_content)
                            
                            # Save music params to music_params folder
                            if music_line:
                                music_filename = os.path.join(music_params_dir, f"music_{clean_header}.txt")
                                with open(music_filename, "w", encoding="utf-8") as f:
                                    f.write(music_line)
                    
                    msg = "Your journey is about to unfold."
                else:
                    raise Exception("Risposta AI vuota")

            except Exception as e:
                print(f"Errore AI: {e}")
                self.emergency_save()
                msg = "Destiny generated (offline mode)."
            
            self.root.after(0, lambda: callback(msg))

        threading.Thread(target=run, daemon=True).start()

    def emergency_save(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(script_dir, "futures_texts")
        os.makedirs(target_dir, exist_ok=True)

        backup = {
            "future_age_40.txt": "Un periodo di grande successo.",
            "future_age_60.txt": "La saggezza ti guida.",
            "future_age_80.txt": "Circondato dall'affetto.",
            "future_death.txt": "Un lascito di ispirazione."
        }
        for name, text in backup.items():
            filepath = os.path.join(target_dir, name)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

# --- STYLING (COSI' COME TI PIACE) ---
TITLE_FONT = ("Georgia", 38, "italic")
ENTRY_FONT = ("Georgia", 22)
BUTTON_STYLE = {"font": ("Georgia", 11, "bold"), "width": 25, "height": 2, "bg": "white", "fg": "black", "relief": "flat"}

class PageWithBackground(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.canvas = tk.Canvas(self, width=1000, height=800, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_item = self.canvas.create_image(0, 0, anchor="nw")

class StartPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.canvas.create_text(500, 300, text="The Life of Chuck", font=TITLE_FONT, fill="white")
        btn = tk.Button(self, text="BEGIN THE JOURNEY", **BUTTON_STYLE, command=lambda: controller.show_page("CameraPage"))
        self.canvas.create_window(500, 520, window=btn)

class CameraPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.is_previewing = False
        self.cam_frame = tk.Frame(self, bg="white", padx=2, pady=2)
        self.cam_view = tk.Label(self.cam_frame, bg="black")
        self.cam_view.pack()
        self.canvas.create_window(500, 380, window=self.cam_frame)
        self.btn_frame = tk.Frame(self, bg="black")
        self.canvas.create_window(500, 650, window=self.btn_frame)
        self.btn_capture = tk.Button(self.btn_frame, text="CAPTURE THE PRESENT", **BUTTON_STYLE, command=self.capture_action)
        self.btn_capture.grid(row=0, column=0)

    def start_webcam(self):
        if not self.is_previewing:
            ret, frame = self.controller.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)
                img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.cam_view.config(image=img)
                self.cam_view.image = img
            self.after(15, self.start_webcam)

    def capture_action(self):
        ret, frame = self.controller.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.is_previewing = True
            self.controller.captured_image = frame
            self.btn_capture.grid_remove()
            
            retake_style = BUTTON_STYLE.copy()
            retake_style.update({"width": 12, "bg": "#444", "fg": "white", "text": "RETAKE"})
            confirm_style = BUTTON_STYLE.copy()
            confirm_style.update({"width": 12, "text": "CONFIRM"})
            
            tk.Button(self.btn_frame, **retake_style, command=self.retake).grid(row=0, column=0, padx=10)
            tk.Button(self.btn_frame, **confirm_style, command=self.confirm).grid(row=0, column=1, padx=10)

    def retake(self):
        self.is_previewing = False
        for widget in self.btn_frame.winfo_children(): widget.destroy()
        self.btn_capture = tk.Button(self.btn_frame, text="CAPTURE THE PRESENT", **BUTTON_STYLE, command=self.capture_action)
        self.btn_capture.grid(row=0, column=0)
        self.start_webcam()

    def confirm(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "chuck_origin.jpg")
        cv2.imwrite(image_path, self.controller.captured_image)
        self.controller.show_page("AgePage") 

class QuestionBase(PageWithBackground):
    def __init__(self, parent, controller, question_text, key, next_page):
        super().__init__(parent, controller)
        self.key, self.next_page = key, next_page
        self.canvas.create_text(500, 300, text=question_text, font=TITLE_FONT, fill="white", width=800)
        self.entry = tk.Entry(self, font=ENTRY_FONT, bg="#111", fg="white", border=0, justify="center")
        self.canvas.create_window(500, 420, window=self.entry, height=60, width=500)
        tk.Button(self, text="CONTINUE", **BUTTON_STYLE, command=self.save_and_next).place(x=350, y=560)

    def save_and_next(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        user_data_path = os.path.join(script_dir, "user_data.txt")
        with open(user_data_path, "a", encoding="utf-8") as f:
            f.write(f"{self.key.upper()}: {self.entry.get()}\n")
        self.controller.show_page(self.next_page)

class AgePage(QuestionBase):
    def __init__(self, parent, controller): 
        super().__init__(parent, controller, "How old are you?", "age", "NamePage")

    def save_and_next(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        user_data_path = os.path.join(script_dir, "user_data.txt")

        age_val = self.entry.get()
        with open(user_data_path, "a", encoding="utf-8") as f:
            f.write(f"AGE: {age_val}\n")

        # Call face aging as soon as age is inserted
        try:
            aging_script = os.path.join(script_dir, "main.py")
            subprocess.Popen(["python", aging_script, "chuck_origin.jpg", age_val], cwd=script_dir)
        except Exception as e:
            print(f"Errore lancio aging: {e}")

        self.controller.show_page(self.next_page)

class NamePage(QuestionBase):
    def __init__(self, parent, controller): 
        super().__init__(parent, controller, "What is your name?", "name", "DreamPage")
class DreamPage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "What is your greatest dream?", "dream", "BioPage")
class BioPage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "Tell us about yourself.", "bio", "EndPage")

class EndPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.status_label = self.canvas.create_text(500, 400, text="Looking at your path among the stars", font=TITLE_FONT, fill="white")
        self.music_ready = False
        self.music_process = None

    def generate_future_timeline(self):
        self.controller.fetch_gemini_bio(self.on_complete)

    def on_complete(self, message):
        # Text generation complete, now wait for aging pipeline
        self.canvas.itemconfig(self.status_label, text=message)
        #self.canvas.create_text(500, 460, text="Waiting for aging process...", font=("Georgia", 18, "italic"), fill="white", tags="waiting")

        # Start pre-loading music in background (while waiting for pipeline)
        threading.Thread(target=self.preload_music, daemon=True).start()
        
        # Start waiting for pipeline in background thread
        threading.Thread(target=self.wait_for_pipeline, daemon=True).start()

    def preload_music(self):
        """Pre-load music analysis in background so it's ready when FINISH is pressed."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            main_script = os.path.join(script_dir, "main.py")
            
            print("🎵 Pre-loading music in background...")
            # Run pre-analysis (this builds cache)
            result = subprocess.run(
                ["python", main_script, "--analyze-only"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Music pre-loaded and ready!")
                self.music_ready = True
            else:
                print(f"⚠️ Music pre-load warning: {result.stderr[:100] if result.stderr else 'unknown'}")
                self.music_ready = True  # Continue anyway
        except Exception as e:
            print(f"⚠️ Music pre-load error: {e}")
            self.music_ready = True  # Continue anyway

    def wait_for_pipeline(self):
        """Wait for aging pipeline to complete, then show FINISH button."""
        # Wait for pipeline (timeout 5 minutes)
        pipeline_done = wait_for_pipeline_only(timeout=300, check_interval=1.0)

        # Update UI on main thread
        if pipeline_done:
            self.controller.root.after(0, self.show_finish_button)
        else:
            self.controller.root.after(0, self.show_timeout_message)

    def show_finish_button(self):
        """Show the FINISH button (called when both processes are done)."""
        self.canvas.delete("waiting")
        #self.canvas.create_text(500, 460, text="All processes complete!", font=("Georgia", 18, "italic"), fill="white")
        tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.finish_and_signal).place(x=350, y=550)

    def show_timeout_message(self):
        """Show timeout message if pipeline didn't complete."""
        self.canvas.delete("waiting")
        #self.canvas.create_text(500, 460, text="Aging process timed out. You can still finish.", font=("Georgia", 16, "italic"), fill="orange")
        tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.finish_and_signal).place(x=350, y=550)

    def finish_and_signal(self):
        # Start music IMMEDIATELY - subprocess.Popen returns instantly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(script_dir, "main.py")
        
        # Launch music player - don't capture output for faster startup
        try:
            # Use os.setsid on Unix to detach process completely
            import os as os_module
            self.music_process = subprocess.Popen(
                ["python", "-u", main_script, "--music-only"],  # -u for unbuffered output
                cwd=script_dir,
                stdout=None,  # Don't capture - faster
                stderr=None,
                start_new_session=True  # Detach from parent
            )
            print("🎵 Music player launched!")
        except Exception as e:
            print(f"⚠️ Could not start music player: {e}")
        
        # Send OSC messages with delay using subprocess (survives GUI close)
        osc_code = '''
import time
from pythonosc import udp_client
print("[OSC] Waiting 5s before sending start...")
time.sleep(5.0)
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client.send_message("/touchdesigner/start", 0)
time.sleep(0.5)
client.send_message("/touchdesigner/start", 1)
print("[OSC] Sent: /touchdesigner/start")
'''
        subprocess.Popen(
            ["python", "-c", osc_code],
            cwd=script_dir,
            stdout=None,
            stderr=None,
            start_new_session=True
        )
        
        signal_gui_complete()  # Signal coordinator that GUI is done
        self.controller.root.destroy()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_path = os.path.join(script_dir, "user_data.txt")

def run_music_analysis_only():
    """Run only music analysis - called with --analyze-music flag."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "main.py")
    
    print("🎵 Starting music analysis...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", main_script, "--analyze-only"],
            cwd=script_dir
        )
        if result.returncode == 0:
            print("✅ Music analysis complete")
        else:
            print("⚠️ Music analysis failed")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Check for music analysis flag
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze-music":
        run_music_analysis_only()
    else:
        # Normal GUI flow
        script_dir = os.path.dirname(os.path.abspath(__file__))
        user_data_path = os.path.join(script_dir, "user_data.txt")
        if os.path.exists(user_data_path):
            os.remove(user_data_path)
        init_flags()  # Reset coordination flags at start
        root = tk.Tk()
        app = LifeOfChuckApp(root)
        root.mainloop()
    