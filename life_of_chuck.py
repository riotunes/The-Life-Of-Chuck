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

APP_WIDTH = 1920
APP_HEIGHT = 1080
CENTER_X = APP_WIDTH // 2
CENTER_Y = APP_HEIGHT // 2
# Shift UI elements slightly further left for better visual centering on wide screens
UI_CENTER_X = CENTER_X - int(APP_WIDTH * 0.10)

class LifeOfChuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Life of Chuck")
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.attributes("-fullscreen", True)
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
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
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
                    img = img.resize((APP_WIDTH, APP_HEIGHT), Image.Resampling.LANCZOS)
                    self.bg_frames.append(ImageTk.PhotoImage(img))
                except:
                    continue

    def animate_background(self):
        if self.bg_frames:
            next_img = self.bg_frames[self.current_frame]
            for page in self.pages.values():
                page.canvas.itemconfig(page.bg_image_item, image=next_img)
            self.current_frame = (self.current_frame + 1) % len(self.bg_frames)
        self.root.after(30, self.animate_background)

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
                Generate exactly {num_stages} raw image prompts for a stream diffusion generative AI WITH MUSIC PARAMETERS.
                JUST SENTENCES

                STRUCTURE:
                - Exactly {num_stages} blocks: [AGE X] ({num_age_blocks - 1} random ages) and [DEATH].
                - start from the current age, never go below it.
                - Each block MUST have TWO parts: IMAGE PROMPT and MUSIC PARAMETERS

                STRICT FORMAT RULES:
                1. SENTENCES: Each image prompt must have exactly 5 sentences, that are concise and highly descriptive.
                2. USER: Understand the gender and the cultural background of the user from the language, name and bio.
                3. LENGTH: EXTREMELY CONCISE. Max 6 words per sentence.
                4. CONTENT: Emotional scene of the user's life age. Social context, environment, significant objects.
                5. BANNED: No selfie, no self-portrait.
                6. PUNCTUATION: Each sentence must end with a comma, except the last one which ends with a period.
                7. NO PERIODS: Zero dots. Only commas.
                8. FOCUS: Emotion, social context, dreams, passions, objects, lighting, colors.
                9. REALISM: Must be realistic and cinematic, NOT abstract or surreal.
                10. MUSIC: After image prompt, add a line with MUSIC: arousal,valence,bpm, acousticness, instrumentalness

                MUSIC PARAMETERS (all values 0.0-1.0 except BPM which is 60-180):
                - arousal: energy/activation level (0=calm, 1=intense)
                - valence: positivity (0=sad/negative, 1=happy/positive), you can use 0.4-0.6 for neutral
                - bpm: tempo in beats per minute (60-180)
                - acousticness: presence of acoustic sounds (0=electronic, 1=acoustic)
                - instrumentalness: presence of instrumental sounds (0=with vocals, 1=instrumental)

                EXAMPLE OF DESIRED OUTPUT:
                [AGE 42]
                Glass walls. A heavy oak desk. The smell of expensive leather and stale coffee. Outside, the city is a grid of cold, indifferent lights.
                MUSIC: 0.6,0.2,100,0.5,0.3

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
    
    def reset_app(self):
        """
        Riavvia completamente l'applicazione senza chiudere la GUI.
        Pulisce tutti i dati, resetta tutte le pagine e torna alla StartPage.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Pulisci user_data.txt
        user_data_path = os.path.join(script_dir, "user_data.txt")
        if os.path.exists(user_data_path):
            try:
                os.remove(user_data_path)
            except:
                pass
        
        # 2. Pulisci l'immagine catturata
        self.captured_image = None
        chuck_origin_path = os.path.join(script_dir, "chuck_origin.jpg")
        if os.path.exists(chuck_origin_path):
            try:
                os.remove(chuck_origin_path)
            except:
                pass
        
        # 3. Pulisci cartelle di output (aged_outputs, uv_outputs, futures_texts, music_params)
        folders_to_clean = ["aged_outputs", "uv_outputs", "futures_texts", "music_params"]
        for folder in folders_to_clean:
            folder_path = os.path.join(script_dir, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    try:
                        os.remove(os.path.join(folder_path, f))
                    except:
                        pass
        
        # 4. Resetta tutte le pagine
        for page_name, page in self.pages.items():
            if hasattr(page, 'reset_page'):
                page.reset_page()
        
        # 5. Reinizializza i flag di coordinamento
        init_flags()
        
        # 6. Torna alla StartPage
        self.show_page("StartPage")

# --- STYLING (COSI' COME TI PIACE) ---
TITLE_FONT = ("DejaVu Sans Mono", 48, "bold")
ENTRY_FONT = ("DejaVu Sans Mono", 26, "normal")
BUTTON_STYLE = {"font": ("DejaVu Sans Mono", 14, "bold"), "width": 28, "height": 2, "bg": "white", "fg": "black", "relief": "flat"}

class PageWithBackground(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.canvas = tk.Canvas(self, width=APP_WIDTH, height=APP_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_item = self.canvas.create_image(0, 0, anchor="nw")

    def create_outlined_text(self, x, y, text, font, fill="white", outline="black", outline_width=2, **kwargs):
        # Draw a simple outline by rendering the text slightly offset in four directions
        for dx, dy in ((-outline_width, 0), (outline_width, 0), (0, -outline_width), (0, outline_width)):
            self.canvas.create_text(x + dx, y + dy, text=text, font=font, fill=outline, anchor="center", **kwargs)
        return self.canvas.create_text(x, y, text=text, font=font, fill=fill, anchor="center", **kwargs)

class StartPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_outlined_text(UI_CENTER_X, int(APP_HEIGHT * 0.35), text="The Life of Chuck", font=TITLE_FONT, fill="white")
        btn = tk.Button(self, text="BEGIN THE JOURNEY", **BUTTON_STYLE, command=lambda: controller.show_page("CameraPage"))
        self.canvas.create_window(UI_CENTER_X, int(APP_HEIGHT * 0.65), window=btn, anchor="center")

class CameraPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.is_previewing = False
        self.cam_frame = tk.Frame(self, bg="white", padx=2, pady=2)
        self.cam_view = tk.Label(self.cam_frame, bg="black")
        self.cam_view.pack()
        self.canvas.create_window(UI_CENTER_X, int(APP_HEIGHT * 0.47), window=self.cam_frame, anchor="center")
        self.btn_frame = tk.Frame(self, bg="black")
        self.canvas.create_window(UI_CENTER_X, int(APP_HEIGHT * 0.8), window=self.btn_frame, anchor="center")
        self.btn_capture = tk.Button(self.btn_frame, text="CAPTURE THE PRESENT", **BUTTON_STYLE, command=self.capture_action)
        self.btn_capture.grid(row=0, column=0)

    def reset_page(self):
        """Resetta la pagina allo stato iniziale."""
        self.is_previewing = False
        # Rimuovi tutti i widget dal btn_frame e ricrea il bottone capture
        for widget in self.btn_frame.winfo_children():
            widget.destroy()
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
            retake_style.update({"width": 12, "bg": "white", "fg": "black", "text": "RETAKE"})
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
        self.create_outlined_text(UI_CENTER_X, int(APP_HEIGHT * 0.35), text=question_text, font=TITLE_FONT, fill="white", width=APP_WIDTH * 0.7)
        self.entry = tk.Entry(self, font=ENTRY_FONT, bg="#111", fg="white", border=0, justify="center")
        self.canvas.create_window(UI_CENTER_X, int(APP_HEIGHT * 0.5), window=self.entry, height=60, width=600, anchor="center")
        tk.Button(self, text="CONTINUE", **BUTTON_STYLE, command=self.save_and_next).place(x=UI_CENTER_X, y=int(APP_HEIGHT * 0.7), anchor="center")

    def reset_page(self):
        """Resetta la pagina svuotando il campo di input."""
        self.entry.delete(0, tk.END)

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
        self.status_label = self.create_outlined_text(UI_CENTER_X, int(APP_HEIGHT * 0.55), text="Looking at your path among the stars", font=TITLE_FONT, fill="white")
        self.music_ready = False
        self.music_process = None
        self.finish_button = None

    def reset_page(self):
        """Resetta la pagina allo stato iniziale."""
        self.music_ready = False
        self.music_process = None
        # Rimuovi il bottone FINISH se esiste
        if self.finish_button:
            self.finish_button.destroy()
            self.finish_button = None
        # Resetta il testo dello status
        self.canvas.itemconfig(self.status_label, text="Looking at your path among the stars")

    def generate_future_timeline(self):
        self.controller.fetch_gemini_bio(self.on_complete)

    def on_complete(self, message):
        # Text generation complete, now wait for aging pipeline
        self.canvas.itemconfig(self.status_label, text=message)

        # Start pre-loading music in background (analysis only)
        threading.Thread(target=self.preload_music, daemon=True).start()
        
        # Wait for both: pipeline + music analysis
        threading.Thread(target=self.wait_for_pipeline_and_music, daemon=True).start()

    def preload_music(self):
        """Pre-load music analysis in background so it's ready when FINISH is pressed."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            main_script = os.path.join(script_dir, "main.py")
            
            print("🎵 Pre-loading music in background (analysis only)...")
            result = subprocess.run(
                ["python", main_script, "--analyze-only"],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Music pre-analysis complete (cache ready)")
                self.music_ready = True
            else:
                print(f"⚠️ Music pre-analysis warning: {result.stderr[:200] if result.stderr else 'unknown'}")
                self.music_ready = True  # allow proceeding anyway
        except Exception as e:
            print(f"⚠️ Music pre-analysis error: {e}")
            self.music_ready = True  # allow proceeding anyway

    def wait_for_pipeline_and_music(self):
        """Wait for aging pipeline to complete AND music analysis to be ready."""
        pipeline_done = wait_for_pipeline_only(timeout=300, check_interval=1.0)
        if not pipeline_done:
            # pipeline timeout -> still allow finish, but log
            print("[Coordinator] Pipeline timed out, allowing Finish.")
        # Wait until music_ready flag is true
        while not self.music_ready:
            time.sleep(0.5)
        # Show finish on main thread
        self.controller.root.after(0, self.show_finish_button)

    def show_finish_button(self):
        """Show the FINISH button (called when both processes are done)."""
        self.canvas.delete("waiting")
        self.finish_button = tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.finish_and_signal)
        self.finish_button.place(x=UI_CENTER_X, y=int(APP_HEIGHT * 0.7), anchor="center")

    def finish_and_signal(self):
        # Nascondi il bottone FINISH dopo che è stato premuto
        if self.finish_button:
            self.finish_button.destroy()
            self.finish_button = None
        
        # Avvia integrazione musica (selezione + invio sequenziale a SC)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(script_dir, "main.py")
        try:
            self.music_process = subprocess.Popen(
                ["python", "-u", main_script, "--music-only"],
                cwd=script_dir,
                stdout=None,
                stderr=None,
                start_new_session=True
            )
            print("🎵 Music integration launched (sending playlist to SuperCollider)...")
        except Exception as e:
            print(f"⚠️ Could not start music integration: {e}")

        # OSC verso TouchDesigner immediati
        osc_code = '''
import time
from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
client.send_message("/touchdesigner/start", 0)
time.sleep(0.2)
client.send_message("/touchdesigner/start", 1)
print("[OSC] Sent: /touchdesigner/start (0 and 1)")
'''
        subprocess.Popen(
            ["python", "-c", osc_code],
            cwd=script_dir,
            stdout=None,
            stderr=None,
            start_new_session=True
        )

        # Segnala GUI completa ma NON chiude la finestra
        signal_gui_complete()

        # Calcola num_stages dall'età per determinare il tempo di attesa
        import re
        user_data_path = os.path.join(script_dir, "user_data.txt")
        try:
            with open(user_data_path, "r", encoding="utf-8") as f:
                data = f.read()
            age_match = re.search(r"AGE: (\d+)", data)
            current_age = int(age_match.group(1)) if age_match else 30
            num_stages = calculate_num_stages(current_age)
        except:
            num_stages = 5  # fallback

        # Mostra messaggio e nascondi elementi, poi programma il reset dopo num_stages * 30 secondi
        wait_time_ms = num_stages * 30 * 1000  # millisecondi
        self.canvas.itemconfig(self.status_label, text="")
        self.controller.root.after(wait_time_ms, self.controller.reset_app)

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
    