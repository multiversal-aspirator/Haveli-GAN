#!/usr/bin/env python3
"""
Haveli-GAN GUI Interface
========================

A simple graphical interface for the Haveli-GAN fresco restoration system.
This provides an easy-to-use interface for restoring damaged Indian paintings.

Requirements:
    pip install tkinter pillow
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import sys

# Add the current directory to path to import our inference module
sys.path.append(os.path.dirname(__file__))

try:
    from inference_haveli_gan import HaveliGANInference
except ImportError as e:
    print(f"Error importing inference module: {e}")
    print("Make sure inference_haveli_gan.py is in the same directory")
    exit(1)


class HaveliGANGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Haveli-GAN Fresco Restoration")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.inference_engine = None
        self.damaged_image_path = None
        self.mask_path = None
        self.restored_image = None
        
        self.setup_ui()
        self.initialize_model()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Haveli-GAN Fresco Restoration", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Damaged image selection
        ttk.Label(input_frame, text="Damaged Image:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.damaged_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.damaged_path_var, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2)
        ttk.Button(input_frame, text="Browse", 
                  command=self.browse_damaged_image).grid(row=0, column=2, pady=2)
        
        # Mask selection (optional)
        ttk.Label(input_frame, text="Damage Mask (optional):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.mask_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.mask_path_var, state="readonly").grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2)
        ttk.Button(input_frame, text="Browse", 
                  command=self.browse_mask).grid(row=1, column=2, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.restore_button = ttk.Button(control_frame, text="Restore Image", 
                                        command=self.restore_image, state="disabled")
        self.restore_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Run Demo", 
                  command=self.run_demo).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Result", 
                  command=self.save_result, state="disabled").pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        image_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Before image
        self.before_frame = ttk.Frame(image_frame)
        self.before_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        ttk.Label(self.before_frame, text="Damaged Image", font=("Arial", 12, "bold")).pack()
        self.before_label = ttk.Label(self.before_frame, text="No image selected")
        self.before_label.pack(expand=True)
        
        # After image
        self.after_frame = ttk.Frame(image_frame)
        self.after_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        ttk.Label(self.after_frame, text="Restored Image", font=("Arial", 12, "bold")).pack()
        self.after_label = ttk.Label(self.after_frame, text="Click 'Restore Image' to begin")
        self.after_label.pack(expand=True)
    
    def initialize_model(self):
        """Initialize the Haveli-GAN model in a separate thread."""
        def init_thread():
            try:
                self.progress_var.set("Loading Haveli-GAN model...")
                self.progress_bar.start()
                
                self.inference_engine = HaveliGANInference()
                
                self.progress_bar.stop()
                self.progress_var.set("Model loaded successfully!")
                
                # Enable restore button
                self.restore_button.config(state="normal")
                
            except Exception as e:
                self.progress_bar.stop()
                self.progress_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Model Loading Error", 
                                   f"Failed to load Haveli-GAN model:\n{str(e)}")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def browse_damaged_image(self):
        """Browse for a damaged image file."""
        file_path = filedialog.askopenfilename(
            title="Select Damaged Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.damaged_image_path = file_path
            self.damaged_path_var.set(file_path)
            self.display_image(file_path, self.before_label, max_size=(300, 300))
            self.progress_var.set("Damaged image loaded")
    
    def browse_mask(self):
        """Browse for a mask file."""
        file_path = filedialog.askopenfilename(
            title="Select Damage Mask",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.mask_path = file_path
            self.mask_path_var.set(file_path)
            self.progress_var.set("Mask loaded")
    
    def display_image(self, image_path, label_widget, max_size=(300, 300)):
        """Display an image in a label widget."""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            label_widget.config(image=photo, text="")
            label_widget.image = photo  # Keep a reference
            
        except Exception as e:
            label_widget.config(text=f"Error loading image:\n{str(e)}")
    
    def restore_image(self):
        """Restore the selected damaged image."""
        if not self.damaged_image_path:
            messagebox.showwarning("No Image", "Please select a damaged image first.")
            return
        
        if not self.inference_engine:
            messagebox.showerror("Model Not Ready", "The model is still loading. Please wait.")
            return
        
        def restore_thread():
            try:
                self.progress_var.set("Restoring image...")
                self.progress_bar.start()
                
                # Perform restoration
                self.restored_image = self.inference_engine.restore_image(
                    self.damaged_image_path,
                    self.mask_path
                )
                
                self.progress_bar.stop()
                self.progress_var.set("Image restored successfully!")
                
                # Display restored image
                self.display_restored_image()
                
                # Enable save button
                for widget in self.root.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Frame):
                                for button in child.winfo_children():
                                    if isinstance(button, ttk.Button) and button.cget("text") == "Save Result":
                                        button.config(state="normal")
                
            except Exception as e:
                self.progress_bar.stop()
                self.progress_var.set(f"Restoration failed: {str(e)}")
                messagebox.showerror("Restoration Error", 
                                   f"Failed to restore image:\n{str(e)}")
        
        threading.Thread(target=restore_thread, daemon=True).start()
    
    def display_restored_image(self):
        """Display the restored image."""
        try:
            # Create a temporary copy for display
            display_image = self.restored_image.copy()
            display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.after_label.config(image=photo, text="")
            self.after_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.after_label.config(text=f"Error displaying restored image:\n{str(e)}")
    
    def save_result(self):
        """Save the restored image."""
        if not self.restored_image:
            messagebox.showwarning("No Result", "No restored image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Restored Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.restored_image.save(file_path, quality=95)
                self.progress_var.set(f"Image saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Restored image saved successfully:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}")
    
    def run_demo(self):
        """Run the demo restoration."""
        if not self.inference_engine:
            messagebox.showerror("Model Not Ready", "The model is still loading. Please wait.")
            return
        
        def demo_thread():
            try:
                self.progress_var.set("Running demo...")
                self.progress_bar.start()
                
                self.inference_engine.run_demo()
                
                self.progress_bar.stop()
                self.progress_var.set("Demo completed!")
                
                messagebox.showinfo("Demo Complete", 
                                  "Demo restoration completed!\nResults saved in './demo_restoration' folder.")
                
            except Exception as e:
                self.progress_bar.stop()
                self.progress_var.set(f"Demo failed: {str(e)}")
                messagebox.showerror("Demo Error", f"Demo failed:\n{str(e)}")
        
        threading.Thread(target=demo_thread, daemon=True).start()


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = HaveliGANGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()
