import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import subprocess
import threading
import time
from openai import OpenAI

# --- CONFIGURAZIONE CHIAVE ---
# Usa la tua chiave sk-proj...
OPENAI_API_KEY = "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"

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
        page_list = (StartPage, CameraPage, NamePage, AgePage, DreamPage, BioPage, EndPage)
        
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
                except: continue

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
        if page_name == "CameraPage": frame.start_webcam()
        if page_name == "EndPage": frame.generate_future_timeline()

    def fetch_openai_bio(self, callback):
        """Metodo di generazione con Debug Error"""
        def run():
            try:
                if not os.path.exists("user_data.txt"):
                    with open("user_data.txt", "w") as f: f.write("NAME: Chuck\nAGE: 30\n")

                with open("user_data.txt", "r", encoding="utf-8") as f:
                    user_info = f.read()
                
                print(">>> Tentativo chiamata OpenAI...")
                client = OpenAI(api_key=OPENAI_API_KEY)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Sei un biografo futurista. Scrivi minimo 100 parole."},
                        {"role": "user", "content": f"Dati: {user_info}. Genera bio con tag: [AGE 40], [AGE 60], [AGE 80], [DEATH]."}
                    ]
                )
                
                full_story = response.choices[0].message.content
                print(">>> Risposta ricevuta!")

                parts = full_story.split('[')
                for part in parts:
                    if ']' in part:
                        header, content = part.split(']', 1)
                        filename = f"future_{header.strip().lower().replace(' ', '_')}.txt"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(content.strip())
                
                msg = "Destino scritto da OpenAI!"
            except Exception as e:
                # GUARDA QUI NEL TERMINALE PER L'ERRORE
                print(f"\n!!! ERRORE CRITICO OPENAI: {e}\n")
                self.emergency_save()
                msg = "Stelle silenziose (Offline Mode)."
            
            self.root.after(0, lambda: callback(msg))

        threading.Thread(target=run, daemon=True).start()

    def emergency_save(self):
        backup = {"future_age_40.txt": "Successo.", "future_age_60.txt": "Viaggi.", "future_age_80.txt": "Pace.", "future_death.txt": "Eredità."}
        for name, text in backup.items():
            with open(name, "w", encoding="utf-8") as f: f.write(text)

# --- STYLING ---
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
            tk.Button(self.btn_frame, text="RETAKE", **{**BUTTON_STYLE, "width": 12}, bg="#444", fg="white", command=self.retake).grid(row=0, column=0, padx=10)
            tk.Button(self.btn_frame, text="CONFIRM", **{**BUTTON_STYLE, "width": 12}, command=self.confirm).grid(row=0, column=1, padx=10)

    def retake(self):
        self.is_previewing = False
        for widget in self.btn_frame.winfo_children(): widget.destroy()
        self.btn_capture = tk.Button(self.btn_frame, text="CAPTURE THE PRESENT", **BUTTON_STYLE, command=self.capture_action)
        self.btn_capture.grid(row=0, column=0)
        self.start_webcam()

    def confirm(self):
        cv2.imwrite("chuck_origin.jpg", self.controller.captured_image)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            aging_script = os.path.abspath(os.path.join(script_dir, "..", "face_aging.py"))
            subprocess.Popen(["python", aging_script, "chuck_origin.jpg"])
        except: pass
        self.controller.show_page("NamePage")

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

class NamePage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "What is your name?", "name", "AgePage")
class AgePage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "How old are you?", "age", "DreamPage")
class DreamPage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "What is your greatest dream?", "dream", "BioPage")
class BioPage(QuestionBase):
    def __init__(self, parent, controller): super().__init__(parent, controller, "Tell us about yourself.", "bio", "EndPage")

class EndPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.status_label = self.canvas.create_text(500, 400, text="Interrogando il tempo...", font=TITLE_FONT, fill="white")

    def generate_future_timeline(self):
        self.controller.fetch_openai_bio(self.on_complete)

    def on_complete(self, message):
        self.canvas.itemconfig(self.status_label, text=message)
        tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.controller.root.destroy).place(x=350, y=550)

if __name__ == "__main__":
    root = tk.Tk()
    app = LifeOfChuckApp(root)
    root.mainloop()