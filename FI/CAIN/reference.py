import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import torch
import cv2
import numpy as np
from datetime import datetime

# Import CAIN components
from model.cain import CAIN
from model.common import InOutPaddings, sub_mean


class VideoInterpolatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CAIN Video Frame Interpolator")
        self.root.geometry("600x500")
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.fps_multiplier = tk.IntVar(value=2)
        self.progress = tk.DoubleVar()
        self.status = tk.StringVar(value="Ready")
        
        self.model = None
        self.processing = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="CAIN Video Frame Interpolator", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Input Video", padding=10)
        input_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side='left', padx=5)
        tk.Button(input_frame, text="Browse", command=self.browse_input).pack(side='left')
        
        # Model Frame
        model_frame = ttk.LabelFrame(self.root, text="Model", padding=10)
        model_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Entry(model_frame, textvariable=self.model_path, width=50).pack(side='left', padx=5)
        tk.Button(model_frame, text="Browse", command=self.browse_model).pack(side='left')
        tk.Button(model_frame, text="Load", command=self.load_model).pack(side='left', padx=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(settings_frame, text="FPS Multiplier:").pack(side='left', padx=5)
        fps_options = ttk.Combobox(settings_frame, textvariable=self.fps_multiplier, 
                                  values=[2, 4, 8], width=10)
        fps_options.pack(side='left', padx=5)
        
        # Output Frame
        output_frame = ttk.LabelFrame(self.root, text="Output Video", padding=10)
        output_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side='left', padx=5)
        tk.Button(output_frame, text="Browse", command=self.browse_output).pack(side='left')
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.pack(fill='x', padx=20, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, 
                                           maximum=100, length=500)
        self.progress_bar.pack(pady=5)
        
        tk.Label(progress_frame, textvariable=self.status).pack()
        
        # Control Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.process_btn = tk.Button(button_frame, text="Process Video", 
                                    command=self.process_video, 
                                    bg='green', fg='white', font=('Arial', 12))
        self.process_btn.pack(side='left', padx=10)
        
        tk.Button(button_frame, text="Exit", command=self.root.quit, 
                 bg='red', fg='white', font=('Arial', 12)).pack(side='left')
        
        # Info Frame
        info_frame = ttk.LabelFrame(self.root, text="Info", padding=10)
        info_frame.pack(fill='both', expand=True, padx=20, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, width=70)
        self.info_text.pack()
        self.info_text.insert('1.0', "Welcome to CAIN Video Interpolator!\n\n"
                                     "1. Select input video\n"
                                     "2. Load trained CAIN model\n"
                                     "3. Choose FPS multiplier\n"
                                     "4. Process video")
        self.info_text.config(state='disabled')
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            # Auto-generate output path
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_interpolated{ext}")
            self.update_info()
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch files", "*.pth *.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def load_model(self):
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model checkpoint file")
            return
        
        try:
            self.status.set("Loading model...")
            self.root.update()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CAIN(depth=3)
            model = torch.nn.DataParallel(model).to(device)
            
            checkpoint = torch.load(self.model_path.get(), map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            
            self.model = model
            self.device = device
            
            self.status.set("Model loaded successfully!")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            self.status.set("Failed to load model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def update_info(self):
        if self.input_path.get() and os.path.exists(self.input_path.get()):
            cap = cv2.VideoCapture(self.input_path.get())
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            info = f"Video Info:\n"
            info += f"Resolution: {width}x{height}\n"
            info += f"FPS: {fps:.2f}\n"
            info += f"Frames: {frame_count}\n"
            info += f"Duration: {duration:.2f} seconds\n"
            info += f"Output FPS: {fps * self.fps_multiplier.get():.2f}"
            
            self.info_text.config(state='normal')
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info)
            self.info_text.config(state='disabled')
    
    def interpolate_frame(self, frame1, frame2):
        """Interpolate a single frame."""
        def to_tensor(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            return frame.to(self.device)
        
        im1 = to_tensor(frame1)
        im2 = to_tensor(frame2)
        
        with torch.no_grad():
            im1, m1 = sub_mean(im1)
            im2, m2 = sub_mean(im2)
            
            paddingInput, paddingOutput = InOutPaddings(im1)
            im1 = paddingInput(im1)
            im2 = paddingInput(im2)
            
            out, _ = self.model(im1, im2)
            
            out = paddingOutput(out)
            out += (m1 + m2) / 2
        
        out = torch.clamp(out.squeeze(0), 0, 1)
        out = out.cpu().numpy().transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        
        return out
    
    def process_video_thread(self):
        """Process video in a separate thread."""
        try:
            # Open input video
            cap = cv2.VideoCapture(self.input_path.get())
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = fps * self.fps_multiplier.get()
            out = cv2.VideoWriter(self.output_path.get(), fourcc, out_fps, (width, height))
            
            # Read all frames
            self.status.set("Reading frames...")
            frames = []
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                self.progress.set((i + 1) / frame_count * 30)
                self.root.update()
            
            cap.release()
            
            # Process frames
            self.status.set("Interpolating frames...")
            total_output_frames = len(frames) + (len(frames) - 1) * (self.fps_multiplier.get() - 1)
            output_count = 0
            
            for i in range(len(frames) - 1):
                # Write original frame
                out.write(frames[i])
                output_count += 1
                
                # Interpolate frames
                if self.fps_multiplier.get() == 2:
                    interpolated = self.interpolate_frame(frames[i], frames[i + 1])
                    out.write(interpolated)
                    output_count += 1
                elif self.fps_multiplier.get() == 4:
                    # Recursive interpolation
                    mid = self.interpolate_frame(frames[i], frames[i + 1])
                    quarter = self.interpolate_frame(frames[i], mid)
                    three_quarter = self.interpolate_frame(mid, frames[i + 1])
                    
                    out.write(quarter)
                    out.write(mid)
                    out.write(three_quarter)
                    output_count += 3
                
                progress = 30 + (output_count / total_output_frames * 70)
                self.progress.set(progress)
                self.root.update()
            
            # Write last frame
            out.write(frames[-1])
            out.release()
            
            self.progress.set(100)
            self.status.set("Processing complete!")
            messagebox.showinfo("Success", f"Video saved to:\n{self.output_path.get()}")
            
        except Exception as e:
            self.status.set("Processing failed!")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.processing = False
            self.process_btn.config(state='normal')
    
    def process_video(self):
        # Validation
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify output path")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        # Start processing
        self.processing = True
        self.process_btn.config(state='disabled')
        self.progress.set(0)
        
        # Run in separate thread
        thread = threading.Thread(target=self.process_video_thread)
        thread.start()


def main():
    root = tk.Tk()
    app = VideoInterpolatorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()