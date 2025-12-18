import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class LifeOfChuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Life of Chuck")
        self.root.geometry("1000x800")
        self.root.configure(bg="black")
        
        self.bg_folder = "bg_frames" 
        self.bg_frames = []
        self.current_frame = 0
        self.load_background_frames()

        self.vid = cv2.VideoCapture(0)
        self.captured_image = None
        self.user_responses = {}
        
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
                img = Image.open(os.path.join(self.bg_folder, f))
                img = img.resize((1000, 800), Image.Resampling.LANCZOS)
                self.bg_frames.append(ImageTk.PhotoImage(img))

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

# --- REFINED STYLING ---
TITLE_FONT = ("Georgia", 38, "italic")
TEXT_FONT = ("Georgia", 16)
ENTRY_FONT = ("Georgia", 22)
BUTTON_STYLE = {
    "font": ("Georgia", 11, "bold"), 
    "width": 25, 
    "height": 2, 
    "bg": "white", 
    "fg": "black", 
    "relief": "flat",
    "activebackground": "#ddd"
}

class PageWithBackground(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller # This line fixes the error you saw
        self.canvas = tk.Canvas(self, width=1000, height=800, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_item = self.canvas.create_image(0, 0, anchor="nw")

# --- 1. START ---
class StartPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.canvas.create_text(500, 300, text="The Life of Chuck", font=TITLE_FONT, fill="white")
        self.canvas.create_text(500, 390, text="A conversation between who you are now\nand the person you are becoming.", 
                                font=TEXT_FONT, fill="white", width=700, justify="center")
        btn = tk.Button(self, text="BEGIN THE JOURNEY", **BUTTON_STYLE, command=lambda: controller.show_page("CameraPage"))
        self.canvas.create_window(500, 520, window=btn)

# --- 2. CAMERA ---
class CameraPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.is_previewing = False
        self.canvas.create_text(500, 100, text="Face your multitudes.", font=TITLE_FONT, fill="white")
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
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
            tk.Button(self.btn_frame, text="RETAKE", font=BUTTON_STYLE["font"], bg="#444", fg="white", command=self.retake, width=12).grid(row=0, column=0, padx=5)
            tk.Button(self.btn_frame, text="CONFIRM", font=BUTTON_STYLE["font"], bg="white", fg="black", command=self.confirm, width=12).grid(row=0, column=1, padx=5)

    def retake(self):
        self.is_previewing = False
        for widget in self.btn_frame.winfo_children(): widget.destroy()
        self.btn_capture = tk.Button(self.btn_frame, text="CAPTURE THE PRESENT", **BUTTON_STYLE, command=self.capture_action)
        self.btn_capture.grid(row=0, column=0)
        self.start_webcam()

    def confirm(self):
        cv2.imwrite("chuck_origin.jpg", self.controller.captured_image)
        self.controller.show_page("NamePage")

# --- QUESTION PAGES ---
class QuestionBase(PageWithBackground):
    def __init__(self, parent, controller, question_text, key, next_page):
        super().__init__(parent, controller)
        self.key = key
        self.next_page = next_page
        self.canvas.create_text(500, 300, text=question_text, font=TITLE_FONT, fill="white", width=800, justify="center")
        self.entry = tk.Entry(self, font=ENTRY_FONT, bg="#111", fg="white", insertbackground="white", border=0, justify="center")
        self.canvas.create_window(500, 420, window=self.entry, height=60, width=500)
        btn = tk.Button(self, text="CONTINUE", **BUTTON_STYLE, command=self.save_and_next)
        self.canvas.create_window(500, 560, window=btn)

    def save_and_next(self):
        answer = self.entry.get()
        # Save to file immediately so nothing is lost
        with open("user_data.txt", "a") as f:
            f.write(f"{self.key.upper()}: {answer}\n")
        self.controller.show_page(self.next_page)

class NamePage(QuestionBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "What is your name?", "name", "AgePage")

class AgePage(QuestionBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "How old are you?", "age", "DreamPage")

class DreamPage(QuestionBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "What is your greatest dream?", "dream", "BioPage")

class BioPage(QuestionBase):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Tell us about yourself.", "bio", "EndPage")

class EndPage(PageWithBackground):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.canvas.create_text(500, 400, text="The transformation begins.", font=TITLE_FONT, fill="white")
        btn = tk.Button(self, text="FINISH", **BUTTON_STYLE, command=self.finish_app)
        self.canvas.create_window(500, 550, window=btn)

    def finish_app(self):
        messagebox.showinfo("Journey Complete", "Your data has been saved to user_data.txt")
        self.controller.root.destroy() # Fixed this to close the app properly

if __name__ == "__main__":
    root = tk.Tk()
    app = LifeOfChuckApp(root)
    root.mainloop()