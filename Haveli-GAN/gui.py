# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import torch
from torchvision import transforms
import numpy as np
import threading

# Import your model classes from model.py
from model import Generator, StyleEncoder

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

# --- IMPORTANT: Update these paths to your trained model files ---
GEN_CHECKPOINT = "./checkpoints/gen_epoch_100.pth" 
STYLE_ENC_CHECKPOINT = "./checkpoints/style_enc_epoch_100.pth"

class FrescoRestorationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Haveli Fresco Restoration")
        self.geometry("1000x500")

        # --- Model Loading ---
        self.generator = None
        self.style_encoder = None
        self.models_loaded = False
        self.load_models()

        # --- Image Placeholders ---
        self.original_img = None
        self.mask_img = None
        
        # --- UI Layout ---
        # Main frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display frames
        image_frame = tk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        titles = ["Original Damaged Image", "Generated Damage Mask", "AI Reconstructed Image"]
        self.image_labels = []
        for i, title_text in enumerate(titles):
            frame = tk.Frame(image_frame, bd=2, relief=tk.SUNKEN)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            image_frame.grid_columnconfigure(i, weight=1)
            
            title = tk.Label(frame, text=title_text, font=("Helvetica", 12, "bold"))
            title.pack(side=tk.TOP, pady=5)
            
            label = tk.Label(frame, text="Load an image to start", bg="gray90")
            label.pack(fill=tk.BOTH, expand=True)
            self.image_labels.append(label)

        # Button and Status frame
        bottom_frame = tk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.btn_load = tk.Button(bottom_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_restore = tk.Button(bottom_frame, text="Restore Image", state=tk.DISABLED, command=self.start_restore_thread)
        self.btn_restore.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(bottom_frame, text="Ready. Please load models.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        
        self.check_model_status()

    def check_model_status(self):
        if self.models_loaded:
            self.status_label.config(text=f"Models loaded successfully on {DEVICE.upper()}. Ready.")
        else:
            self.status_label.config(text="Error: Model checkpoint files not found. Check paths in gui.py.")
            messagebox.showerror("Model Loading Error", f"Could not load model files from:\n{GEN_CHECKPOINT}\n{STYLE_ENC_CHECKPOINT}\nPlease check the paths and try again.")
            self.btn_load.config(state=tk.DISABLED)

    def load_models(self):
        try:
            self.generator = Generator().to(DEVICE)
            self.generator.load_state_dict(torch.load(GEN_CHECKPOINT, map_location=DEVICE))
            self.generator.eval()

            self.style_encoder = StyleEncoder().to(DEVICE)
            self.style_encoder.load_state_dict(torch.load(STYLE_ENC_CHECKPOINT, map_location=DEVICE))
            self.style_encoder.eval()
            self.models_loaded = True
        except FileNotFoundError:
            self.models_loaded = False

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return

        # Store original image
        self.original_img = Image.open(file_path).convert("RGB")
        
        # Display original image
        self.display_image(self.original_img, 0)
        
        # Automatically generate and display mask
        self.mask_img = self.create_mask(self.original_img)
        self.display_image(self.mask_img, 1)

        # Clear previous result and enable button
        self.image_labels[2].config(image='', text="Ready to Restore")
        self.btn_restore.config(state=tk.NORMAL)
        self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")

    def create_mask(self, img):
        """
        Creates a binary mask from the image.
        This is a simple heuristic: it identifies very light areas (plaster loss)
        and very dark areas as damage. You can tune the thresholds.
        """
        gray_img = ImageOps.grayscale(img)
        np_img = np.array(gray_img)
        
        # Thresholds for damage (0-255 scale)
        light_threshold = 220  # Identifies bright white spots
        dark_threshold = 30   # Identifies very dark stains
        
        mask = np.where((np_img > light_threshold) | (np_img < dark_threshold), 255, 0)
        return Image.fromarray(mask.astype(np.uint8))

    def display_image(self, img, label_index):
        # Resize for display
        display_size = (300, 300)
        img_display = img.copy()
        img_display.thumbnail(display_size)
        
        photo_img = ImageTk.PhotoImage(img_display)
        self.image_labels[label_index].config(image=photo_img)
        self.image_labels[label_index].image = photo_img # Keep a reference!
    
    def start_restore_thread(self):
        """Use a thread to prevent the GUI from freezing during inference."""
        self.btn_restore.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.status_label.config(text="Restoring... this may take a moment.")
        
        # Run inference in a separate thread
        thread = threading.Thread(target=self.restore_image)
        thread.daemon = True
        thread.start()

    def restore_image(self):
        # Pre-processing transforms
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        # Prepare tensors
        damaged_tensor = transform(self.original_img).unsqueeze(0).to(DEVICE)
        mask_tensor = mask_transform(self.mask_img).unsqueeze(0).to(DEVICE)
        # Use the damaged image itself as the style reference for simplicity in the GUI
        style_ref_tensor = damaged_tensor

        # Perform Restoration
        with torch.no_grad():
            style_vector = self.style_encoder(style_ref_tensor)
            restored_tensor = self.generator(damaged_tensor, mask_tensor, style_vector)

        # Post-process for display
        restored_tensor = restored_tensor.squeeze(0).cpu()
        restored_tensor = restored_tensor * 0.5 + 0.5 # De-normalize from [-1, 1] to [0, 1]
        restored_img = transforms.ToPILImage()(restored_tensor)

        # Update GUI from the main thread
        self.after(0, self.update_ui_after_restore, restored_img)

    def update_ui_after_restore(self, restored_img):
        self.display_image(restored_img, 2)
        self.status_label.config(text="Restoration complete!")
        self.btn_restore.config(state=tk.NORMAL)
        self.btn_load.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = FrescoRestorationApp()
    app.mainloop()