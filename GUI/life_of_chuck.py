import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import subprocess
import threading
import time
from google import genai

GEMINI_API_KEY = "AIzaSyAL4NxY6azP6trq8P_RXIApViN_8tvY9_A"

class LifeOfChuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Life of Chuck")
        self.root.geometry("1000x800")
        self.root.configure(bg="black")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bg_folder = os.path.abspath(os.path.join(script_dir, "..", "bg_frames"))
        
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
                with open("user_data.txt", "r", encoding="utf-8") as f:
                    data = f.read()
                
                import re
                age_match = re.search(r"AGE: (\d+)", data)
                current_age = int(age_match.group(1)) if age_match else 30
                
                client = genai.Client(api_key=GEMINI_API_KEY)
                
                # PROMPT: Focus su lunghezza minima e virgola dopo ogni parola
                prompt = f"""
                Dati Utente: {data}
                Età attuale: {current_age}
                
                OBIETTIVO:
                Scrivi una biografia futura narrativa, logica e molto dettagliata. 
                Decidi tu l'età del decesso (longevità casuale).
                Dividi in blocchi decennali: [AGE X], [AGE Y]... e [DEATH].

                REGOLE MANDATORIE:
                1. Ogni blocco deve contenere una narrazione di ALMENO 150 PAROLE. Sii estremamente descrittivo riguardo al lavoro e al raggiungimento dei sogni.
                2. Ogni singola parola deve essere seguita da una virgola (esempio: Il, successo, arriva, dopo, anni, di, fatica,).
                3. Non usare punti fermi, usa solo virgole dopo ogni parola.
                4. Focus: Evoluzione professionale, traguardi del sogno, collaborazioni, impatto nel settore. No descrizioni fisiche.
                5. Lingua: Inglese.
                6. Tieni a questo punto unicamente le parole chiave della biografia più soignificative
                
                IMPORTANTE: Se la narrazione di un blocco è inferiore alle 150 parole, espandi i dettagli lavorativi e i pensieri della persona.
                """
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt
                )
                
                if response.text:
                    full_story = response.text
                    
                    # Pulizia file precedenti
                    for f_old in os.listdir("."):
                        if f_old.startswith("future_") and f_old.endswith(".txt"):
                            os.remove(f_old)

                    parts = full_story.split('[')
                    for part in parts:
                        if ']' in part:
                            header, content = part.split(']', 1)
                            clean_header = header.strip().lower().replace(' ', '_')
                            filename = f"future_{clean_header}.txt"
                            
                            # Rimuoviamo eventuali righe vuote e puliamo lo spazio finale
                            text_content = content.strip()
                            
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(text_content)
                    
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
        backup = {
            "future_age_40.txt": "Un periodo di grande successo.",
            "future_age_60.txt": "La saggezza ti guida.",
            "future_age_80.txt": "Circondato dall'affetto.",
            "future_death.txt": "Un lascito di ispirazione."
        }
        for name, text in backup.items():
            with open(name, "w", encoding="utf-8") as f:
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
        cv2.imwrite("chuck_origin.jpg", self.controller.captured_image)
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
        with open("user_data.txt", "a", encoding="utf-8") as f:
            f.write(f"{self.key.upper()}: {self.entry.get()}\n")
        self.controller.show_page(self.next_page)

class AgePage(QuestionBase):
    def __init__(self, parent, controller): 
        super().__init__(parent, controller, "How old are you?", "age", "NamePage")

    def save_and_next(self):
        age_val = self.entry.get()
        with open("user_data.txt", "a", encoding="utf-8") as f:
            f.write(f"AGE: {age_val}\n")
        
        # Call face aging as soon as age is inserted
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            aging_script = os.path.abspath(os.path.join(script_dir, "..", "life-of-chuck-aging", "main.py"))
            subprocess.Popen(["python", aging_script, "chuck_origin.jpg", age_val])
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

    def generate_future_timeline(self):
        self.controller.fetch_gemini_bio(self.on_complete)

    def on_complete(self, message):
        self.canvas.itemconfig(self.status_label, text=message)
        tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.controller.root.destroy).place(x=350, y=550)

if __name__ == "__main__":
    if os.path.exists("user_data.txt"): os.remove("user_data.txt")
    root = tk.Tk()
    app = LifeOfChuckApp(root)
    root.mainloop()