import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup, 
                             QComboBox, QProgressBar, QFrame, QSizePolicy, QSlider, QMessageBox,
                             QCheckBox, QGroupBox, QTabWidget, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QBrush, QLinearGradient, QPen, QFont, QPainterPath, QIcon, QPolygon
from PyQt5.QtCore import Qt, QRect, QRectF, QSize, QThread, pyqtSignal, QTimer, QPoint
import time
import shutil
import subprocess
import platform

try:
    from omegaconf import DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("OmegaConf not available - Real-ESRGAN API features will be limited")

sys.path.append(r"C:\Users\hongs\Documents\SR and FI\SR\Real-ESRGAN")
sys.path.append(r"C:\Users\hongs\Documents\SR and FI\SR\SwinIR")

sys.path.append(r"C:\Users\hongs\Documents\SR and FI\FI\CAIN")

# Import Real-ESRGAN modules
try:
    from real_esrgan.apis.super_resolution import SuperResolutionInferencer
    from real_esrgan.apis.video_super_resolution import VideoSuperResolutionInferencer
    from real_esrgan.apis.large_image_super_resolution import LargeImageSuperResolutionInferencer
    from real_esrgan.utils.envs import select_device
    from real_esrgan.utils.imgproc import image_to_tensor, tensor_to_image
    from real_esrgan.engine.backend import SuperResolutionBackend
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Real-ESRGAN modules not available")

# Import SwinIR modules
try:
    import SR.SwinIR.model as swinir_model
    from SR.SwinIR.imgproc import preprocess_one_image, tensor_to_image as swinir_tensor_to_image
    from SR.SwinIR.utils import load_pretrained_state_dict
    SWINIR_AVAILABLE = True
except ImportError:
    SWINIR_AVAILABLE = False
    print("SwinIR modules not available")

# Import CAIN modules
try:
    from FI.CAIN.model.cain import CAIN
    CAIN_AVAILABLE = True
except ImportError:
    CAIN_AVAILABLE = False
    print("CAIN modules not available")

sys.path.append(r"C:\Users\hongs\Documents\SR and FI\FI\RIFE")

try:
    from FI.RIFE.model.RIFE_HDv3 import Model as RIFEModel
    RIFE_AVAILABLE = True
except Exception as e:
    RIFE_AVAILABLE = False
    print("❌ Error loading RIFE module:", e)

class RoundedImageLabel(QLabel):
    """Custom QLabel with rounded corners for displaying images"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.border_radius = 10
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2C2C2C;")
        self.setContentsMargins(10, 10, 10, 10)
        self._pixmap = None
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(self._pixmap)
        self.update()
        
    def paintEvent(self, event):
        if self._pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Create rounded rect path
            rect = QRect(10, 10, self.width() - 20, self.height() - 20)
            
            # Clip the painter to the rounded rectangle
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), self.border_radius, self.border_radius)
            painter.setClipPath(path)
            
            # Calculate scaled size while maintaining aspect ratio
            pixmap_size = self._pixmap.size()
            scaled_size = pixmap_size.scaled(rect.size(), Qt.KeepAspectRatio)
            
            # Center the pixmap in the rounded rect
            x = rect.x() + (rect.width() - scaled_size.width()) / 2
            y = rect.y() + (rect.height() - scaled_size.height()) / 2
            
            # Draw pixmap
            painter.drawPixmap(QRect(int(x), int(y), scaled_size.width(), scaled_size.height()), self._pixmap)
        else:
            # Draw placeholder with text if no image
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            rect = QRect(10, 10, self.width() - 20, self.height() - 20)
            
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), self.border_radius, self.border_radius)
            painter.setClipPath(path)
            
            # Draw rounded rect background
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#2C2C2C")))
            painter.drawRoundedRect(rect, self.border_radius, self.border_radius)
            
            # Draw text
            painter.setPen(QPen(QColor("#808080")))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(rect, Qt.AlignCenter, "No Media Selected")

class ClickableSlider(QSlider):
    """Custom QSlider that jumps to clicked position immediately"""
    
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        
    def mousePressEvent(self, event):
        """Override mouse press to jump directly to clicked position"""
        if event.button() == Qt.LeftButton:
            # Calculate the position based on where the user clicked
            if self.orientation() == Qt.Horizontal:
                # For horizontal slider
                slider_length = self.width() - self.style().pixelMetric(self.style().PM_SliderLength)
                click_pos = event.x() - (self.style().pixelMetric(self.style().PM_SliderLength) // 2)
                
                # Ensure click position is within valid range
                click_pos = max(0, min(click_pos, slider_length))
                
                # Calculate value based on position
                value_range = self.maximum() - self.minimum()
                new_value = self.minimum() + (click_pos / slider_length) * value_range
                
                # Set the new value
                self.setValue(int(new_value))
                
                # Emit the sliderPressed signal to start seeking
                self.sliderPressed.emit()
                
            else:
                # For vertical slider (if needed in future)
                slider_length = self.height() - self.style().pixelMetric(self.style().PM_SliderLength)
                click_pos = self.height() - event.y() - (self.style().pixelMetric(self.style().PM_SliderLength) // 2)
                
                click_pos = max(0, min(click_pos, slider_length))
                value_range = self.maximum() - self.minimum()
                new_value = self.minimum() + (click_pos / slider_length) * value_range
                
                self.setValue(int(new_value))
                self.sliderPressed.emit()
        
        # Call the parent implementation for other functionality
        super().mousePressEvent(event)

class CAINVideoWorker(QThread):
    """Worker thread for CAIN frame interpolation with iterative approach"""
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(np.ndarray)
    processing_completed = pyqtSignal(str, dict)  # Add video_info parameter
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, output_path, model_path, device, num_iterations=1):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.device = device
        self.num_iterations = num_iterations
        
    def interpolate_frames_iterative(self, model, I0, I1, num_iterations):
        """
        Iterative interpolation that builds frames step by step
        For x2: creates 1 frame between I0 and I1
        For x4: first creates 1 frame, then creates frames between existing frames
        For x8: repeats the process one more time
        """
        device = I0.device
        
        # Start with the two original frames
        frames = [I0, I1]
        
        # Perform interpolation for each iteration
        for iteration in range(num_iterations):
            new_frames = []
            
            # For each pair of consecutive frames, generate an intermediate frame
            for i in range(len(frames) - 1):
                # Add the current frame
                new_frames.append(frames[i])
                
                # Generate intermediate frame between frames[i] and frames[i+1]
                with torch.no_grad():
                    # Ensure frames are properly formatted
                    frame_a = frames[i].detach()
                    frame_b = frames[i + 1].detach()
                    
                    # Clear gradients if any
                    if frame_a.requires_grad:
                        frame_a = frame_a.detach()
                    if frame_b.requires_grad:
                        frame_b = frame_b.detach()
                    
                    # Generate intermediate frame
                    middle, _ = model(frame_a, frame_b)
                    
                    # Ensure the output is properly detached and clamped
                    middle = middle.detach().clone()
                    middle = torch.clamp(middle, 0.0, 1.0)
                    
                    # Add the intermediate frame
                    new_frames.append(middle)
                    
                    # Clear cache after each interpolation to prevent memory issues
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Add the last frame
            new_frames.append(frames[-1])
            
            # Update frames list for next iteration
            frames = new_frames
            
            # Additional memory cleanup between iterations
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Return only the interpolated frames (exclude original frames)
        # frames[0] is I0, frames[-1] is I1, so we return frames[1:-1]
        return frames[1:-1]
        
    def run(self):
        try:
            if not CAIN_AVAILABLE:
                self.error_occurred.emit("CAIN modules not available")
                return
            
            self.progress_updated.emit(5)
            
            # Initialize CAIN model
            device = torch.device(self.device)
            model = CAIN(depth=3)
            model = torch.nn.DataParallel(model).to(device)
            
            # Load checkpoint
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                del checkpoint
            else:
                self.error_occurred.emit(f"CAIN model not found at: {self.model_path}")
                return
            
            model.eval()
            self.progress_updated.emit(15)
            
            # Auto clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate output FPS
            output_fps = fps * (2 ** self.num_iterations)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, output_fps, (width, height))
            
            if not writer.isOpened():
                # Fallback to AVI
                fallback_path = str(Path(self.output_path).with_suffix('.avi'))
                writer = cv2.VideoWriter(fallback_path, cv2.VideoWriter_fourcc(*'XVID'), output_fps, (width, height))
                if writer.isOpened():
                    self.output_path = fallback_path
                else:
                    self.error_occurred.emit("Failed to initialize video writer")
                    return
            
            self.progress_updated.emit(25)
            
            # Process video frames
            ret, prev_frame = cap.read()
            if not ret:
                self.error_occurred.emit("Cannot read first frame")
                return
            
            frame_idx = 0
            
            # Keep track of previous interpolated frames for temporal consistency
            prev_interpolated_frames = None
            
            with torch.no_grad():
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR frames to RGB for CAIN model (CAIN expects RGB)
                    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                    curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    
                    # Ensure frames are float32 and normalized
                    prev_frame_rgb = prev_frame_rgb.astype(np.float32) / 255.0
                    curr_frame_rgb = curr_frame_rgb.astype(np.float32) / 255.0
                    
                    # Apply slight denoising for x4 and x8 to reduce artifacts
                    if self.num_iterations > 1:
                        prev_frame_rgb = cv2.bilateralFilter((prev_frame_rgb * 255).astype(np.uint8), 5, 50, 50).astype(np.float32) / 255.0
                        curr_frame_rgb = cv2.bilateralFilter((curr_frame_rgb * 255).astype(np.uint8), 5, 50, 50).astype(np.float32) / 255.0
                    
                    # Convert frames to tensors (RGB format, CHW)
                    prev_tensor = torch.from_numpy(prev_frame_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    curr_tensor = torch.from_numpy(curr_frame_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    
                    # Ensure tensors are contiguous and in [0, 1] range
                    prev_tensor = torch.clamp(prev_tensor.contiguous(), 0.0, 1.0)
                    curr_tensor = torch.clamp(curr_tensor.contiguous(), 0.0, 1.0)
                    
                    # Write original frame (BGR format for video file)
                    writer.write(prev_frame)
                    
                    # Clear cache before interpolation
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Generate interpolated frames using iterative method
                    try:
                        interpolated_list = self.interpolate_frames_iterative(model, prev_tensor, curr_tensor, self.num_iterations)
                    except Exception as e:
                        print(f"Interpolation error at frame {frame_idx}: {e}")
                        # Fallback: create simple blend frames
                        num_frames = (2 ** self.num_iterations) - 1
                        interpolated_list = []
                        for i in range(num_frames):
                            alpha = (i + 1) / (num_frames + 1)
                            blend = (1 - alpha) * prev_tensor + alpha * curr_tensor
                            interpolated_list.append(blend)
                    
                    # Process and write interpolated frames
                    for idx, interpolated in enumerate(interpolated_list):
                        try:
                            # Ensure tensor is properly formatted
                            frame_tensor = interpolated.detach().clone()
                            frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                            
                            # Check for invalid values
                            if torch.isnan(frame_tensor).any() or torch.isinf(frame_tensor).any():
                                print(f"Warning: Invalid tensor values detected at frame {frame_idx}, interp {idx}")
                                # Use linear interpolation as fallback
                                alpha = (idx + 1) / (len(interpolated_list) + 1)
                                frame_tensor = (1 - alpha) * prev_tensor + alpha * curr_tensor
                                frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                            
                            # Convert to numpy (CHW -> HWC)
                            frame_np = frame_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                            
                            # Ensure values are in [0, 1] range
                            frame_np = np.clip(frame_np, 0.0, 1.0)
                            
                            # Apply temporal consistency for x4 and x8
                            if self.num_iterations > 1 and prev_interpolated_frames is not None and idx < len(prev_interpolated_frames):
                                # Blend with corresponding frame from previous pair
                                blend_factor = 0.15  # 15% blend
                                frame_np = (1 - blend_factor) * frame_np + blend_factor * prev_interpolated_frames[idx]
                                frame_np = np.clip(frame_np, 0.0, 1.0)
                            
                            # Convert to uint8 RGB
                            frame_rgb = (frame_np * 255.0).round().astype(np.uint8)
                            
                            # Ensure frame has correct shape
                            if frame_rgb.shape[:2] != (height, width):
                                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                            
                            # Convert RGB to BGR for video writing
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            writer.write(frame_bgr)
                            
                            # Emit preview frame (RGB format for display)
                            if idx == 0:  # Only emit first interpolated frame for preview
                                self.frame_processed.emit(frame_rgb.copy())
                                
                        except Exception as e:
                            print(f"Error processing interpolated frame: {e}")
                            # Write a duplicate of previous frame as fallback
                            writer.write(prev_frame)
                    
                    # Store interpolated frames for temporal consistency
                    if self.num_iterations > 1:
                        prev_interpolated_frames = [f.squeeze(0).cpu().numpy().transpose(1, 2, 0) 
                                                   for f in interpolated_list]
                    
                    prev_frame = curr_frame
                    
                    frame_idx += 1
                    progress = 25 + int(70 * frame_idx / frame_count)
                    self.progress_updated.emit(progress)
                    
                    # Clear cache more frequently for higher multipliers
                    if device.type == 'cuda':
                        if self.num_iterations > 1 and frame_idx % 10 == 0:
                            torch.cuda.empty_cache()
                        elif frame_idx % 50 == 0:
                            torch.cuda.empty_cache()
            
            # Write last frame
            writer.write(prev_frame)
            
            cap.release()
            writer.release()
            
            # Final cache clear
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Create enhanced video info for Frame Interpolation
            enhanced_video_info = {
                'width': width,
                'height': height,
                'fps': output_fps,  # This is the enhanced FPS
                'frame_count': frame_count * (2 ** self.num_iterations)  # Interpolated frame count
            }
            
            self.progress_updated.emit(100)
            self.processing_completed.emit(self.output_path, enhanced_video_info)
            
        except Exception as e:
            self.error_occurred.emit(f"CAIN processing error: {str(e)}")
            # Cleanup on error
            try:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                if 'writer' in locals() and writer.isOpened():
                    writer.release()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except:
                pass

class RIFEVideoWorker(QThread):
    """Worker thread for RIFE frame interpolation"""
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(np.ndarray)
    processing_completed = pyqtSignal(str, dict)  # Add video_info parameter
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, output_path, model_path, device, exp=1):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.device = device
        self.exp = exp
        self.scale = 1.0  # Fixed scale
        self.fp16 = False  # Disabled FP16
        
    def load_model(self, model_path, device):
        """Load RIFE model the same way as inference_video.py"""
        from FI.RIFE.model.IFNet_HDv3 import IFNet
        
        model = IFNet().eval()
        ckpt = torch.load(model_path, map_location=device)
        
        # Handle checkpoint with 'module.' prefix
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt, strict=True)
        
        model = model.to(device)
        return model
        
    def run(self):
        try:
            if not RIFE_AVAILABLE:
                self.error_occurred.emit("RIFE modules not available")
                return
            
            self.progress_updated.emit(5)
            
            # Initialize RIFE model
            device = torch.device(self.device)
            torch.set_grad_enabled(False)
            
            # Auto clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            model = self.load_model(self.model_path, device)
            
            self.progress_updated.emit(15)
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate output FPS
            output_fps = fps * (2 ** self.exp)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, output_fps, (width, height))
            
            if not writer.isOpened():
                # Fallback to AVI
                fallback_path = str(Path(self.output_path).with_suffix('.avi'))
                writer = cv2.VideoWriter(fallback_path, cv2.VideoWriter_fourcc(*'XVID'), output_fps, (width, height))
                if writer.isOpened():
                    self.output_path = fallback_path
                else:
                    self.error_occurred.emit("Failed to initialize video writer")
                    return
            
            self.progress_updated.emit(25)
            
            # Process video frames
            ret, prev_frame = cap.read()
            if not ret:
                self.error_occurred.emit("Cannot read first frame")
                return
            
            # Prepare padding for RIFE
            tmp = max(32, int(32 / self.scale))
            ph = ((height - 1) // tmp + 1) * tmp
            pw = ((width - 1) // tmp + 1) * tmp
            padding = (0, pw - width, 0, ph - height)
            
            frame_idx = 0
            
            # Convert first frame BGR to RGB
            prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            I1 = torch.from_numpy(np.transpose(prev_frame_rgb, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
            I1 = torch.nn.functional.pad(I1, padding)
            
            with torch.no_grad():
                while True:
                    ret, curr_frame = cap.read()
                    if not ret:
                        break
                    
                    I0 = I1
                    # Convert current frame BGR to RGB
                    curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    I1 = torch.from_numpy(np.transpose(curr_frame_rgb, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.0
                    I1 = torch.nn.functional.pad(I1, padding)
                    
                    # Write original frame (BGR format for video file)
                    writer.write(prev_frame)
                    
                    # Generate interpolated frames using the inference method from inference_video.py
                    output = self.make_inference(model, I0, I1, 2**self.exp-1)
                    
                    for mid in output:
                        # RIFE outputs RGB format, convert to BGR for video writing
                        mid_frame_rgb = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:height, :width]
                        mid_frame_bgr = cv2.cvtColor(mid_frame_rgb, cv2.COLOR_RGB2BGR)
                        writer.write(mid_frame_bgr)
                        
                        # Emit preview frame (already in RGB format for display)
                        self.frame_processed.emit(mid_frame_rgb)
                    
                    prev_frame = curr_frame
                    
                    frame_idx += 1
                    progress = 25 + int(70 * frame_idx / frame_count)
                    self.progress_updated.emit(progress)
                    
                    # Clear cache periodically
                    if frame_idx % 50 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Write last frame
            writer.write(prev_frame)
            
            cap.release()
            writer.release()
            
            # Final cache clear
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Create enhanced video info for Frame Interpolation
            enhanced_video_info = {
                'width': width,
                'height': height,
                'fps': output_fps,  # This is the enhanced FPS
                'frame_count': frame_count * (2 ** self.exp)  # Interpolated frame count
            }
            
            self.progress_updated.emit(100)
            self.processing_completed.emit(self.output_path, enhanced_video_info)
            
        except Exception as e:
            self.error_occurred.emit(f"RIFE processing error: {str(e)}")
    
    def make_inference(self, model, I0, I1, n):
        """Recursive function to generate intermediate frames - adapted from inference_video.py"""
        with torch.no_grad():
            def recur(I0, I1, n):
                # Use the same inference call as inference_video.py
                middle = model(torch.cat((I0, I1), 1), [4/self.scale, 2/self.scale, 1/self.scale])[2][2]
                if n == 1:
                    return [middle]
                first_half = recur(I0, middle, n // 2)
                second_half = recur(middle, I1, n // 2)
                if n % 2:
                    return first_half + [middle] + second_half
                else:
                    return first_half + second_half
            return recur(I0, I1, n)

class RealESRGANImageWorker(QThread):
    """Worker thread for Real-ESRGAN image processing using official API"""
    progress_updated = pyqtSignal(int)
    processing_completed = pyqtSignal(np.ndarray, str)  # Add save_path parameter
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path, model_path, device, scale_factor, save_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.device = device
        self.scale_factor = scale_factor
        self.save_path = save_path
        
    def run(self):
        try:
            if not REALESRGAN_AVAILABLE:
                self.error_occurred.emit("Real-ESRGAN modules not available")
                return
                
            self.progress_updated.emit(10)
            
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            
            # Auto clear GPU cache
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            
            # Create a temporary config-like object
            if not OMEGACONF_AVAILABLE:
                self.error_occurred.emit("OmegaConf not available. Please install omegaconf package.")
                return
                
            from omegaconf import DictConfig
            config = DictConfig({
                'DEVICE': self.device,
                'WEIGHTS_PATH': self.model_path,
                'INPUTS': str(Path(self.image_path).parent),
                'OUTPUT': './temp_output',
                'UPSCALE_FACTOR': self.scale_factor
            })
            
            self.progress_updated.emit(20)
            
            # Always use SuperResolutionInferencer as requested
            inferencer = SuperResolutionInferencer(config)
            self.progress_updated.emit(30)
            
            # Warmup
            inferencer.warmup()
            self.progress_updated.emit(40)
            
            # Process using direct inference - SuperResolutionInferencer doesn't have infer method, use manual processing
            image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            
            self.progress_updated.emit(50)
            
            # Process using SuperResolutionInferencer processing pipeline
            lr_tensor = inferencer.pre_process(image)
            self.progress_updated.emit(60)
            
            with torch.no_grad():
                sr_tensor = inferencer.model(lr_tensor)
            
            self.progress_updated.emit(80)
            sr_image = inferencer.post_process(sr_tensor)
            self.progress_updated.emit(90)
            
            # Convert BGR back to RGB for display (post_process returns BGR)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
            
            # Handle different scale factors
            if self.scale_factor != 4:
                orig_h, orig_w = image.shape[:2]
                target_h, target_w = orig_h * self.scale_factor, orig_w * self.scale_factor
                sr_image = cv2.resize(sr_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Save the image if save_path is provided
            if self.save_path:
                # Convert RGB to BGR for saving
                save_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_path, save_image)
            
            # Final cache clear
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            
            self.progress_updated.emit(100)
            self.processing_completed.emit(sr_image, self.save_path)
                
        except torch.cuda.OutOfMemoryError:
            # Clear cache when OOM occurs
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            self.error_occurred.emit("CUDA out of memory. Try using CPU.")
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

class RealESRGANVideoWorker(QThread):
    """Worker thread for Real-ESRGAN video processing (manual loop only)"""
    progress_updated   = pyqtSignal(int)
    frame_processed    = pyqtSignal(np.ndarray)
    processing_completed = pyqtSignal(str, dict)  # Add video_info parameter
    error_occurred     = pyqtSignal(str)
    path_changed       = pyqtSignal(str)

    def __init__(self, video_path, output_path, model_path, device, scale_factor, codec='XVID'):
        super().__init__()
        self.video_path   = video_path
        self.output_path  = output_path
        self.model_path   = model_path
        self.device       = device
        self.scale_factor = scale_factor
        self.codec        = codec
        self.inferencer   = None

    def run(self):
        if not REALESRGAN_AVAILABLE:
            self.error_occurred.emit("Real-ESRGAN modules not available")
            return

        # Skip API processing and go directly to manual processing
        # to have full control over the output filename and color handling
        self.manual_video_processing()

    def manual_video_processing(self):
        try:
            # 1) Warm up Real-ESRGAN
            torch.backends.cudnn.benchmark = True
            from omegaconf import DictConfig
            from pathlib import Path
            
            # Auto clear GPU cache
            if self.device != 'cpu':
                torch.cuda.empty_cache()

            cfg = DictConfig({
                'DEVICE':        self.device,
                'WEIGHTS_PATH':  self.model_path,
                'INPUTS':        self.video_path,
                'OUTPUT':        str(Path(self.output_path).parent),
                'UPSCALE_FACTOR': self.scale_factor
            })
            self.inferencer = VideoSuperResolutionInferencer(cfg)
            self.inferencer.warmup()
            self.inferencer.model.eval()

            # 2) Open source video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("Cannot open input video.")
                return
            fps   = cap.get(cv2.CAP_PROP_FPS)
            nfrm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # 3) Prepare VideoWriter (with codec fallback)
            out_w, out_h = w * self.scale_factor, h * self.scale_factor
            ext = Path(self.output_path).suffix.lower()
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext in ('.mp4','.mov') else 'XVID'))
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                # fallback to .avi
                fallback = str(Path(self.output_path).with_suffix('.avi'))
                writer = cv2.VideoWriter(fallback, cv2.VideoWriter_fourcc(*'XVID'),
                                        fps, (out_w, out_h))
                if writer.isOpened():
                    self.output_path = fallback
                    self.path_changed.emit(f"Codec fallback: saving as {fallback}")
                else:
                    self.error_occurred.emit("Failed to initialize VideoWriter.")
                    return

            # 4) Process frames
            cap = cv2.VideoCapture(self.video_path)
            idx, skip = 0, max(1, nfrm // 20)
            self.progress_updated.emit(5)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # preprocess & inference (frame is BGR uint8)
                self.inferencer.model.model.float()
                frame_norm = frame.astype(np.float32) / 255.0
                lr_tensor  = self.inferencer.pre_process(frame_norm)
                with torch.no_grad():
                    sr_tensor = self.inferencer.model(lr_tensor)
                sr_out = self.inferencer.post_process(sr_tensor)  # could be float32 [0–1] or uint8 [0–255]

                # Detect dtype and convert to BGR uint8 correctly
                if sr_out.dtype == np.float32 or sr_out.dtype == np.float64:
                    # float image in [0,1]
                    sr_frame_bgr = (np.clip(sr_out, 0, 1) * 255.0).round().astype(np.uint8)
                elif sr_out.dtype == np.uint8:
                    # already uint8 BGR
                    sr_frame_bgr = sr_out
                else:
                    # fallback: cast to uint8
                    sr_frame_bgr = np.clip(sr_out, 0, 255).astype(np.uint8)

                # write BGR directly
                writer.write(sr_frame_bgr)

                # emit preview in RGB
                if idx % skip == 0:
                    sr_frame_rgb = cv2.cvtColor(sr_frame_bgr, cv2.COLOR_BGR2RGB)
                    self.frame_processed.emit(sr_frame_rgb)

                idx += 1
                prog = 5 + int(90 * idx / max(1, nfrm))
                self.progress_updated.emit(prog)
                
                # Clear cache periodically
                if idx % 50 == 0 and self.device != 'cpu':
                    torch.cuda.empty_cache()

            cap.release()
            writer.release()
            
            # Final cache clear
            if self.device != 'cpu':
                torch.cuda.empty_cache()

            # 5) Create enhanced video info
            enhanced_video_info = {
                'width': out_w,
                'height': out_h,
                'fps': fps,
                'frame_count': nfrm
            }

            # 6) Done!
            self.progress_updated.emit(100)
            self.processing_completed.emit(self.output_path, enhanced_video_info)

        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {e}")
            # cleanup
            try:
                if 'cap' in locals() and cap.isOpened(): cap.release()
                if 'writer' in locals() and writer.isOpened(): writer.release()
                if self.device != 'cpu': torch.cuda.empty_cache()
            except:
                pass


class SwinIRImageWorker(QThread):
    """Worker thread for SwinIR image processing using official API"""
    progress_updated = pyqtSignal(int)
    processing_completed = pyqtSignal(np.ndarray, str)  # Add save_path parameter
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_path, model_path, device, scale_factor, save_path):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.device = device
        self.scale_factor = scale_factor
        self.save_path = save_path
        
    def run(self):
        try:
            if not SWINIR_AVAILABLE:
                self.error_occurred.emit("SwinIR modules not available")
                return
                
            device = torch.device(self.device)
            self.progress_updated.emit(10)
            
            # Auto clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Read original image and preprocess
            input_tensor = preprocess_one_image(self.image_path, False, False, device)
            self.progress_updated.emit(20)
            
            # Initialize the model
            sr_model = self.build_model("swinir_default_sr_x4", device)
            self.progress_updated.emit(30)
            
            # Load model weights
            sr_model = load_pretrained_state_dict(sr_model, False, self.model_path)
            self.progress_updated.emit(40)
            
            # Set model to evaluation mode
            sr_model.eval()
            
            self.progress_updated.emit(50)
            
            # Clear GPU cache if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Process image
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()  # Clear cache before processing
                sr_tensor = sr_model(input_tensor)
            
            self.progress_updated.emit(80)
            
            # Convert tensor to image - SwinIR tensor_to_image returns RGB format
            sr_image = swinir_tensor_to_image(sr_tensor, False, False)
            
            # SwinIR tensor_to_image returns RGB format, so no need to convert for display
            # (The GUI expects RGB format for display)
            
            # For non-x4 scaling, resize the output
            if self.scale_factor != 4:
                # Read original image to get dimensions
                orig_image = cv2.imread(self.image_path)
                orig_h, orig_w = orig_image.shape[:2]
                target_h, target_w = orig_h * self.scale_factor, orig_w * self.scale_factor
                
                # Resize the SR image to the target size
                sr_image = cv2.resize(sr_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
            self.progress_updated.emit(90)
            
            # Save the image if save_path is provided
            if self.save_path:
                # Convert RGB to BGR for saving
                save_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_path, save_image)
            
            # Final cache clear
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Emit completion signal with the processed image (RGB format)
            self.progress_updated.emit(100)
            self.processing_completed.emit(sr_image, self.save_path)
            
        except torch.cuda.OutOfMemoryError:
            self.error_occurred.emit("CUDA out of memory. Try using CPU.")
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
            
    def build_model(self, model_arch_name, device):
        """Build the SwinIR model"""
        # Initialize the super-resolution model
        sr_model = swinir_model.__dict__[model_arch_name](in_channels=3,
                                                        out_channels=3,
                                                        channels=64)
        sr_model = sr_model.to(device)
        return sr_model

class SwinIRVideoWorker(QThread):
    """Worker thread for SwinIR video processing using official API"""
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(np.ndarray)
    processing_completed = pyqtSignal(str, dict)  # Add video_info parameter
    error_occurred = pyqtSignal(str)
    path_changed = pyqtSignal(str)  # Signal for when output path changes due to codec fallback
    
    def __init__(self, video_path, output_path, model_path, device, scale_factor, codec='XVID'):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.device = device
        self.scale_factor = scale_factor
        self.codec = codec
        
    def run(self):
        try:
            if not SWINIR_AVAILABLE:
                self.error_occurred.emit("SwinIR modules not available")
                return
                
            device = torch.device(self.device)
            self.progress_updated.emit(10)
            
            # Auto clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Initialize the model
            sr_model = self.build_model("swinir_default_sr_x4", device)
            self.progress_updated.emit(15)
            
            # Load model weights
            sr_model = load_pretrained_state_dict(sr_model, False, self.model_path)
            self.progress_updated.emit(20)
            
            # Set model to evaluation mode
            sr_model.eval()
            
            self.progress_updated.emit(25)
            
            # Clear GPU cache if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Open the video file
            cap = cv2.VideoCapture(self.video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate output dimensions
            out_width = width * self.scale_factor
            out_height = height * self.scale_factor
            
            # Create video writer with appropriate codec
            if self.codec == 'mp4v':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.codec == 'XVID':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Default fallback
                
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (out_width, out_height))
            
            # Check if video writer is opened successfully
            if not writer.isOpened():
                # Try fallback codec
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                alt_path = str(Path(self.output_path).with_suffix('.avi'))
                writer = cv2.VideoWriter(alt_path, fourcc, fps, (out_width, out_height))
                if writer.isOpened():
                    self.output_path = alt_path
                    self.path_changed.emit(f"Using fallback codec - output saved as: {alt_path}")
                else:
                    self.error_occurred.emit("Failed to initialize video writer. Please try a different output format.")
                    return
            
            # Process each frame
            frame_idx = 0
            success, frame = cap.read()
            
            while success and frame_idx < frame_count:
                # Save frame to temp file (SwinIR preprocessing expects a file path)
                temp_frame_path = f"temp_frame_{frame_idx}.png"
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Process frame using the same logic as image processing
                    input_tensor = preprocess_one_image(temp_frame_path, False, False, device)
                    
                    with torch.no_grad():
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()  # Clear cache for each frame
                        sr_tensor = sr_model(input_tensor)
                    
                    # Convert tensor to image - SwinIR tensor_to_image returns RGB format
                    sr_frame = swinir_tensor_to_image(sr_tensor, False, False)
                    
                    # For non-x4 scaling, resize the output
                    if self.scale_factor != 4:
                        target_h, target_w = height * self.scale_factor, width * self.scale_factor
                        sr_frame = cv2.resize(sr_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # SwinIR tensor_to_image returns RGB, but OpenCV expects BGR for video writing
                    sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
                    
                    # Write frame (now in correct BGR format)
                    writer.write(sr_frame_bgr)
                    
                    # Emit frame for preview (sr_frame is already in RGB format)
                    self.frame_processed.emit(sr_frame)
                    
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.error_occurred.emit("CUDA out of memory during video processing. Try using CPU.")
                    return
                    
                finally:
                    # Remove temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                
                # Read next frame
                success, frame = cap.read()
                frame_idx += 1
                
                # Update progress
                progress = 25 + int(75 * frame_idx / frame_count)
                self.progress_updated.emit(progress)
                
                # Clear cache periodically
                if frame_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Release resources
            cap.release()
            writer.release()
            
            # Final cache clear
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Create enhanced video info
            enhanced_video_info = {
                'width': out_width,
                'height': out_height,
                'fps': fps,
                'frame_count': frame_count
            }
            
            # Signal completion
            self.progress_updated.emit(100)
            self.processing_completed.emit(self.output_path, enhanced_video_info)
            
        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {str(e)}")
    
    def build_model(self, model_arch_name, device):
        """Build the SwinIR model"""
        # Initialize the super-resolution model
        sr_model = swinir_model.__dict__[model_arch_name](in_channels=3,
                                                        out_channels=3,
                                                        channels=64)
        sr_model = sr_model.to(device)
        return sr_model

class StyleWidget(QWidget):
    """Base widget with dynamic gradient background"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gradient_colors = [
            ("#1A2980", "#26D0CE"),  # Blue gradient for images
            ("#667eea", "#764ba2"),  # Purple gradient for videos
            ("#FF6B6B", "#4ECDC4"),  # Red-teal gradient for frame interpolation
        ]
        self.current_gradient = 0
        
    def set_gradient(self, gradient_index):
        """Set the gradient theme"""
        self.current_gradient = gradient_index
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        colors = self.gradient_colors[self.current_gradient]
        gradient.setColorAt(0, QColor(colors[0]))
        gradient.setColorAt(1, QColor(colors[1]))
        painter.fillRect(self.rect(), gradient)

class ProcessingTab(QWidget):
    """Enhanced processing tab with improved video playback and color handling"""
    def __init__(self, tab_type, parent=None):
        super().__init__(parent)
        self.tab_type = tab_type  # "image", "video", or "interpolation"
        self.original_media = None
        self.enhanced_media = None
        self.current_view = "original"
        
        # Store video information
        self.original_video_info = {}
        self.enhanced_video_info = {}
        
        # Model paths
        self.realesrgan_model_path = r"C:\Users\hongs\Documents\SR and FI\SR\Real-ESRGAN\results\train\realesrgan_x4-df2k_degradation1\weights\g_best_checkpoint.pkl"
        self.swinir_model_path = r"C:\Users\hongs\Documents\SR and FI\SR\SwinIR\results\SwinIRNet_default_sr_x4-DFO2K\best.tar"
        self.cain_model_path = r"C:\Users\hongs\Documents\SR and FI\FI\CAIN\best.pth"
        self.rife_model_path = r"C:\Users\hongs\Documents\SR and FI\FI\RIFE\train_log\flownet.pkl"
        
        # Video-specific attributes
        self.media_player = None
        self.current_video_path = None
        self.enhanced_video_path = None  # Store enhanced video path
        self.saved_file_path = None  # Store saved file path for open location
        self.current_player_path = None  # Track current player's video path
        self.is_playing = False
        self.video_duration = 0
        self.video_fps = 30.0  # Default FPS, will be updated when loading video
        
        self.setup_ui()
        self.update_availability_status()
    
    def setup_ui(self):
        """Setup the UI for the tab"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Create left panel (controls)
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("""
            QWidget#leftPanel {
                background-color: rgba(255, 255, 255, 30);
                border-radius: 15px;
            }
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4A7BFF;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5D8BFF;
            }
            QPushButton:pressed {
                background-color: #3A6BF0;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QRadioButton, QComboBox, QCheckBox, QSpinBox {
                color: white;
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 80);
                border: 1px solid rgba(255, 255, 255, 100);
                border-radius: 5px;
                padding: 5px;
                color: black;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid black;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #4A7BFF;
                selection-color: white;
                border: 1px solid rgba(0, 0, 0, 50);
            }
            QProgressBar {
                border: none;
                border-radius: 5px;
                text-align: center;
                background-color: rgba(255, 255, 255, 50);
            }
            QProgressBar::chunk {
                background-color: #4A7BFF;
                border-radius: 5px;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 2px solid rgba(255, 255, 255, 50);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        
        # Method selection group
        if self.tab_type == "interpolation":
            method_group = QGroupBox("Frame Interpolation Method")
            method_layout = QVBoxLayout(method_group)
            
            self.method_group = QButtonGroup(self)
            self.cain_radio = QRadioButton("CAIN")
            self.rife_radio = QRadioButton("RIFE")
            
            self.method_group.addButton(self.cain_radio, 1)
            self.method_group.addButton(self.rife_radio, 2)
            self.rife_radio.setChecked(True)  # RIFE is generally more stable
            
            self.cain_radio.toggled.connect(self.update_method)
            self.rife_radio.toggled.connect(self.update_method)
            
            method_layout.addWidget(self.cain_radio)
            method_layout.addWidget(self.rife_radio)
            left_layout.addWidget(method_group)
        else:
            method_group = QGroupBox("Super-Resolution Method")
            method_layout = QVBoxLayout(method_group)
            
            self.method_group = QButtonGroup(self)
            self.realesrgan_radio = QRadioButton("Real-ESRGAN")
            self.swinir_radio = QRadioButton("SwinIR")
            
            self.method_group.addButton(self.realesrgan_radio, 1)
            self.method_group.addButton(self.swinir_radio, 2)
            self.realesrgan_radio.setChecked(True)
            
            self.realesrgan_radio.toggled.connect(self.update_method)
            self.swinir_radio.toggled.connect(self.update_method)
            
            method_layout.addWidget(self.realesrgan_radio)
            method_layout.addWidget(self.swinir_radio)
            left_layout.addWidget(method_group)
        
        # File selection
        if self.tab_type == "interpolation":
            file_type = "Video"
        else:
            file_type = "Image" if self.tab_type == "image" else "Video"
        
        file_label = QLabel(f"Input {file_type}:")
        left_layout.addWidget(file_label)
        
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("font-weight: normal; color: #f0f0f0;")
        self.file_path_label.setWordWrap(True)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        browse_button.setFixedWidth(100)
        
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(browse_button)
        left_layout.addLayout(file_layout)
        
        # Parameters based on tab type
        if self.tab_type == "interpolation":
            self.setup_interpolation_controls(left_layout)
        else:
            self.setup_sr_controls(left_layout)
        
        # Device selection
        device_label = QLabel("Processing Device:")
        left_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.addItem("CUDA (GPU)", "cuda:0")
        self.device_combo.addItem("CPU", "cpu")
        left_layout.addWidget(self.device_combo)
        
        # Process button
        if self.tab_type == "interpolation":
            process_text = "Interpolate Video"
        else:
            process_text = f"Process {file_type}"
        
        self.process_button = QPushButton(process_text)
        self.process_button.setFixedHeight(50)
        self.process_button.clicked.connect(self.process_file)
        self.process_button.setDisabled(True)
        left_layout.addWidget(self.process_button)
        
        # Progress bar (more prominent for video)
        if self.tab_type in ["video", "interpolation"]:
            # Video processing progress section
            progress_group = QGroupBox("Processing Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFixedHeight(25)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid rgba(255, 255, 255, 50);
                    border-radius: 12px;
                    text-align: center;
                    background-color: rgba(255, 255, 255, 20);
                    font-weight: bold;
                    font-size: 12px;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #4A7BFF, stop:1 #26D0CE);
                    border-radius: 10px;
                }
            """)
            progress_layout.addWidget(self.progress_bar)
            
            # Processing time info
            self.time_label = QLabel("Ready to process")
            self.time_label.setStyleSheet("font-weight: normal; color: #f0f0f0; font-size: 11px;")
            self.time_label.setAlignment(Qt.AlignCenter)
            progress_layout.addWidget(self.time_label)
            
            left_layout.addWidget(progress_group)
        else:
            # Simple progress bar for image
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFixedHeight(15)
            left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: normal; color: #f0f0f0; font-size: 12px;")
        left_layout.addWidget(self.status_label)
        
        # Add spacer
        left_layout.addStretch()
        
        # Create right panel (media display)
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_panel.setStyleSheet("""
            QWidget#rightPanel {
                background-color: rgba(255, 255, 255, 30);
                border-radius: 15px;
            }
        """)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # Media display area
        if self.tab_type in ["video", "interpolation"]:
            # Video display with player controls
            self.setup_video_display(right_layout)
        else:
            # Image display
            self.media_label = RoundedImageLabel()
            right_layout.addWidget(self.media_label)
        
        # Toggle and save buttons
        button_layout = QHBoxLayout()
        
        self.toggle_frame = QFrame()
        self.toggle_frame.setFixedHeight(40)
        toggle_layout = QHBoxLayout(self.toggle_frame)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        
        self.original_button = QPushButton("Original")
        if self.tab_type == "interpolation":
            self.enhanced_button = QPushButton("Interpolated")
        else:
            self.enhanced_button = QPushButton("Enhanced")
        
        # Style toggle buttons
        toggle_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 20);
                color: white;
                border: none;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #4A7BFF;
            }
            QPushButton#leftButton {
                border-top-left-radius: 20px;
                border-bottom-left-radius: 20px;
            }
            QPushButton#rightButton {
                border-top-right-radius: 20px;
                border-bottom-right-radius: 20px;
            }
        """
        
        self.original_button.setObjectName("leftButton")
        self.enhanced_button.setObjectName("rightButton")
        self.toggle_frame.setStyleSheet(toggle_style)
        
        self.original_button.setCheckable(True)
        self.enhanced_button.setCheckable(True)
        self.original_button.setChecked(True)
        
        self.original_button.clicked.connect(lambda: self.toggle_view("original"))
        self.enhanced_button.clicked.connect(lambda: self.toggle_view("enhanced"))
        
        toggle_layout.addWidget(self.original_button)
        toggle_layout.addWidget(self.enhanced_button)
        
        # Open file location button
        self.open_location_button = QPushButton("Open file location")
        self.open_location_button.setFixedHeight(40)
        self.open_location_button.clicked.connect(self.open_file_location)
        self.open_location_button.setDisabled(True)

        button_layout.addWidget(self.toggle_frame)
        button_layout.addStretch()
        button_layout.addWidget(self.open_location_button)

        right_layout.addLayout(button_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

    def open_file_location(self):
        """Open file location with improved path handling"""
        # Determine which file path to use based on current view and tab type
        file_path = None
        
        if self.current_view == "enhanced":
            if self.tab_type == "image" and self.saved_file_path:
                file_path = self.saved_file_path
            elif self.tab_type in ["video", "interpolation"] and self.enhanced_video_path:
                file_path = self.enhanced_video_path
        
        # Fallback to original/input file if enhanced not available
        if not file_path:
            if self.tab_type in ["video", "interpolation"] and self.current_video_path:
                file_path = self.current_video_path
            elif self.file_path_label.text() != "No file selected":
                file_path = self.file_path_label.text()
        
        if file_path and os.path.exists(file_path):
            folder = os.path.dirname(file_path)
            try:
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", folder])
                else:  # Linux
                    subprocess.Popen(["xdg-open", folder])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open file location: {str(e)}")
        else:
            QMessageBox.information(self, "No file", "File not found or not saved yet.")

    def setup_interpolation_controls(self, layout):
        """Setup controls specific to frame interpolation"""
        # FPS Multiplier
        fps_label = QLabel("FPS Multiplier:")
        layout.addWidget(fps_label)
        
        fps_layout = QHBoxLayout()
        self.fps_group = QButtonGroup(self)
        
        self.fps_x2 = QRadioButton("x2")
        self.fps_x4 = QRadioButton("x4")
        self.fps_x8 = QRadioButton("x8")
        
        self.fps_group.addButton(self.fps_x2, 1)
        self.fps_group.addButton(self.fps_x4, 2)
        self.fps_group.addButton(self.fps_x8, 3)
        self.fps_x2.setChecked(True)
        
        fps_layout.addWidget(self.fps_x2)
        fps_layout.addWidget(self.fps_x4)
        fps_layout.addWidget(self.fps_x8)
        layout.addLayout(fps_layout)
    
    def setup_sr_controls(self, layout):
        """Setup controls specific to super-resolution"""
        # Scaling factor
        scale_label = QLabel("Scaling Factor:")
        layout.addWidget(scale_label)
        
        scale_layout = QHBoxLayout()
        self.scale_group = QButtonGroup(self)
        
        self.scale_x2 = QRadioButton("x2")
        self.scale_x3 = QRadioButton("x3")
        self.scale_x4 = QRadioButton("x4")
        
        self.scale_group.addButton(self.scale_x2, 2)
        self.scale_group.addButton(self.scale_x3, 3)
        self.scale_group.addButton(self.scale_x4, 4)
        self.scale_x4.setChecked(True)
        
        scale_layout.addWidget(self.scale_x2)
        scale_layout.addWidget(self.scale_x3)
        scale_layout.addWidget(self.scale_x4)
        layout.addLayout(scale_layout)
    
    def setup_video_display(self, layout):
        """Setup video display with enhanced playback controls"""
        # Video display area
        self.media_label = RoundedImageLabel()
        layout.addWidget(self.media_label)
        
        # Video controls frame
        video_controls = QFrame()
        video_controls.setFixedHeight(80)
        video_controls_layout = QVBoxLayout(video_controls)
        video_controls_layout.setContentsMargins(0, 5, 0, 5)
        video_controls_layout.setSpacing(8)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        # Play/Pause button with icons
        self.play_button = QPushButton()
        self.play_button.setFixedSize(40, 40)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 10);
                border-radius: 20px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 20);
                border-radius: 20px;
            }
            QPushButton:disabled {
                background-color: transparent;
            }
        """)
        
        # Load icons
        self.load_playback_icons()
        
        # Set initial play icon
        self.play_button.setIcon(self.play_icon)
        self.play_button.setIconSize(QSize(35, 35))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setDisabled(True)
        
        # Video progress slider - USE CUSTOM SLIDER
        self.video_progress = ClickableSlider(Qt.Horizontal)
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        self.video_progress.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 30);
                height: 8px;
                background: rgba(255, 255, 255, 20);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 2px solid #4A7BFF;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::sub-page:horizontal {
                background: #4A7BFF;
                border-radius: 4px;
            }
        """)
        
        # Add seeking state flags
        self.is_seeking = False
        self.was_playing_before_seek = False
        
        # Connect signals for proper seeking behavior
        self.video_progress.sliderPressed.connect(self.start_seeking)
        self.video_progress.sliderMoved.connect(self.seek_video)
        self.video_progress.sliderReleased.connect(self.end_seeking)
        self.video_progress.setDisabled(True)
        
        # Time labels
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setStyleSheet("color: white; font-size: 11px; font-weight: bold;")
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setStyleSheet("color: white; font-size: 11px; font-weight: bold;")
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.current_time_label)
        playback_layout.addWidget(self.video_progress, 1)
        playback_layout.addWidget(self.total_time_label)
        
        video_controls_layout.addLayout(playback_layout)
        
        # Rest of the video controls setup...
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setStyleSheet("color: black; font-size: 10px; font-weight: bold;")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        video_controls_layout.addWidget(self.video_info_label)
        
        self.view_indicator = QLabel("Viewing: Original")
        self.view_indicator.setStyleSheet("color: black; font-size:10px; font-weight: bold;")
        self.view_indicator.setAlignment(Qt.AlignCenter)
        video_controls_layout.addWidget(self.view_indicator)
        
        if self.tab_type == "interpolation":
            codec_info = QLabel("Frame Interpolation • Supported: MP4, AVI, MOV")
        else:
            codec_info = QLabel("Supported: MP4, AVI, MOV • Codecs: H.264, XVID")
        codec_info.setStyleSheet("color: black; font-size:10px; font-weight: bold;")
        codec_info.setAlignment(Qt.AlignCenter)
        video_controls_layout.addWidget(codec_info)
        
        layout.addWidget(video_controls)
        
        # Timer for updating video progress
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_progress)

    def load_playback_icons(self):
        """Load play and pause icons from image files"""
        try:
            # Try to load play icon
            if os.path.exists("img/play.png"):
                self.play_icon = QIcon("img/play.png")
            else:
                # Fallback: create a simple play icon or use text
                self.play_icon = QIcon()
                print("Warning: img/play.png not found, using fallback")
            
            # Try to load pause icon
            if os.path.exists("img/pause.png"):
                self.pause_icon = QIcon("img/pause.png")
            else:
                # Fallback: create a simple pause icon or use text
                self.pause_icon = QIcon()
                print("Warning: img/pause.png not found, using fallback")
                
            # If icons couldn't be loaded, create fallback text-based icons
            if self.play_icon.isNull():
                self.create_fallback_icons()
                
        except Exception as e:
            print(f"Error loading icons: {e}")
            self.create_fallback_icons()

    def create_fallback_icons(self):
        """Create simple fallback icons if image files are not found"""
        # Create play icon (triangle)
        play_pixmap = QPixmap(24, 24)
        play_pixmap.fill(Qt.transparent)
        painter = QPainter(play_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.NoPen)
        
        # Draw triangle
        triangle = QPolygon([
            QPoint(6, 4),
            QPoint(6, 20),
            QPoint(20, 12)
        ])
        painter.drawPolygon(triangle)
        painter.end()
        self.play_icon = QIcon(play_pixmap)
        
        # Create pause icon (two rectangles)
        pause_pixmap = QPixmap(24, 24)
        pause_pixmap.fill(Qt.transparent)
        painter = QPainter(pause_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.NoPen)
        
        # Draw two rectangles
        painter.drawRect(6, 4, 4, 16)
        painter.drawRect(14, 4, 4, 16)
        painter.end()
        self.pause_icon = QIcon(pause_pixmap)

    def start_seeking(self):
        """Called when user starts seeking"""
        self.is_seeking = True
        self.was_playing_before_seek = self.is_playing
        if self.is_playing:
            self.pause_video()
        # Immediately seek to the new position
        self.seek_video()

    def end_seeking(self):
        """Called when user finishes seeking"""
        self.is_seeking = False
        # Resume playback if it was playing before seeking
        if self.was_playing_before_seek:
            self.play_video()
    
    # Video Playback Methods
    def toggle_playback(self):
        """Toggle video playback with improved error handling"""
        if not self.current_video_path and not hasattr(self, 'enhanced_video_path'):
            QMessageBox.information(self, "No Video", "Please load a video file first.")
            return
            
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
    
    def play_video(self):
        """Start video playback with smart path selection and position preservation"""
        # Determine which video to play based on current view
        video_path = None
        
        if self.current_view == "enhanced" and hasattr(self, 'enhanced_video_path') and self.enhanced_video_path:
            if os.path.exists(self.enhanced_video_path):
                video_path = self.enhanced_video_path
            else:
                QMessageBox.warning(self, "File Not Found", "Enhanced video file not found. Playing original.")
                video_path = self.current_video_path
        else:
            video_path = self.current_video_path
        
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", "Video file not found or invalid.")
            return
            
        try:
            # Check if we need to create a new player or if we can reuse the existing one
            need_new_player = False
            current_position = 0
            
            if self.media_player:
                # Check if the current player is for the same video file
                current_video_path = getattr(self, 'current_player_path', None)
                if current_video_path != video_path:
                    # Different video file, need new player
                    need_new_player = True
                    self.media_player.release()
                else:
                    # Same video file, preserve current position
                    current_position = self.media_player.get(cv2.CAP_PROP_POS_FRAMES)
                    
                    # Check if we're at the end of the video
                    if current_position >= self.video_duration - 1:
                        # At the end, reset to beginning
                        current_position = 0
                        self.media_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_progress.setValue(0)
            else:
                # No existing player
                need_new_player = True
                
            # Create new player if needed
            if need_new_player:
                self.media_player = cv2.VideoCapture(video_path)
                if not self.media_player.isOpened():
                    QMessageBox.warning(self, "Playback Error", "Could not open video file for playback")
                    return
                    
                # Store the current player's video path
                self.current_player_path = video_path
                
                # Update video duration and FPS for the current video
                self.video_duration = int(self.media_player.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.media_player.get(cv2.CAP_PROP_FPS)
                
                # Get current progress bar position and seek to it if it's not at the beginning
                progress_position = self.video_progress.value()
                if progress_position > 0:
                    target_frame = int((progress_position / 100) * self.video_duration)
                    self.media_player.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    current_position = target_frame
                
            # Calculate correct timer interval based on video FPS
            if self.video_fps > 0:
                timer_interval = int(1000 / self.video_fps)  # Convert to milliseconds
            else:
                timer_interval = 33  # Fallback to ~30 FPS
            
            # Start playback
            self.is_playing = True
            self.play_button.setIcon(self.pause_icon)  # Change to pause icon
            self.video_timer.start(timer_interval)  # Use actual video FPS
            
            # Update view indicator
            if self.tab_type == "interpolation":
                view_text = "Interpolated" if video_path == getattr(self, 'enhanced_video_path', None) else "Original"
            else:
                view_text = "Enhanced" if video_path == getattr(self, 'enhanced_video_path', None) else "Original"
            self.view_indicator.setText(f"👁️ Playing: {view_text} ({self.video_fps:.1f} FPS)")
            
            # Update the current frame display immediately
            self.update_video_frame()
            
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", f"Could not play video: {str(e)}")
            
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_button.setIcon(self.play_icon)  # Change to play icon
        self.video_timer.stop()
        
        # Update view indicator
        if self.tab_type == "interpolation":
            view_text = "Interpolated" if self.current_view == "enhanced" else "Original"
        else:
            view_text = "Enhanced" if self.current_view == "enhanced" else "Original"
        self.view_indicator.setText(f"👁️ Viewing: {view_text}")

    def seek_video(self):
        """Seek video to slider position with validation"""
        if not self.media_player or self.video_duration == 0:
            return
            
        position = self.video_progress.value()
        frame_number = int((position / 100) * self.video_duration)
        
        # Ensure frame number is within valid range
        frame_number = max(0, min(frame_number, self.video_duration - 1))
        
        try:
            self.media_player.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.update_video_frame()
            
            # Update time labels immediately
            if self.video_fps > 0:
                current_time = int(frame_number / self.video_fps)
                self.current_time_label.setText(f"{current_time//60:02d}:{current_time%60:02d}")
                
        except Exception as e:
            print(f"Error seeking video: {e}")
    
    def update_video_progress(self):
        """Update video progress during playback"""
        if not self.media_player or not self.is_playing or self.is_seeking:
            return
            
        try:
            current_frame = self.media_player.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((current_frame / self.video_duration) * 100) if self.video_duration > 0 else 0
            
            # Ensure progress is within valid range
            progress = max(0, min(progress, 100))
            
            # Only update slider if user is not seeking
            if not self.is_seeking:
                self.video_progress.setValue(progress)
            
            # Update time labels using actual video FPS
            if self.video_fps > 0:
                current_time = int(current_frame / self.video_fps)
                total_time = int(self.video_duration / self.video_fps)
                
                self.current_time_label.setText(f"{current_time//60:02d}:{current_time%60:02d}")
                self.total_time_label.setText(f"{total_time//60:02d}:{total_time%60:02d}")
            
            # Read and display next frame
            self.update_video_frame()
            
        except Exception as e:
            print(f"Error updating video progress: {e}")
            self.pause_video()
    
    def update_video_frame(self):
        """Update the displayed video frame during playback"""
        if not self.media_player:
            return
            
        try:
            ret, frame = self.media_player.read()
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Ensure contiguous array
                if not frame_rgb.flags['C_CONTIGUOUS']:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                
                # Convert to QPixmap and display
                h, w, c = frame_rgb.shape
                q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.media_label.setPixmap(pixmap)
            else:
                # End of video - reset to beginning
                if not self.is_seeking:
                    self.pause_video()
                    self.video_progress.setValue(0)
                    if self.media_player:
                        self.media_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Show first frame
                        ret, frame = self.media_player.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            if not frame_rgb.flags['C_CONTIGUOUS']:
                                frame_rgb = np.ascontiguousarray(frame_rgb)
                            h, w, c = frame_rgb.shape
                            q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_img)
                            self.media_label.setPixmap(pixmap)
                            self.media_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
        except Exception as e:
            print(f"Error updating video frame: {e}")
            if not self.is_seeking:
                self.pause_video()
    
    # File Management Methods
    def update_availability_status(self):
        """Update the availability status of methods"""
        if self.tab_type == "interpolation":
            if not CAIN_AVAILABLE:
                self.cain_radio.setEnabled(False)
                self.cain_radio.setText("CAIN (Not Available)")
                
            if not RIFE_AVAILABLE:
                self.rife_radio.setEnabled(False)
                self.rife_radio.setText("RIFE (Not Available)")
                
            # If RIFE is not available but CAIN is, switch to CAIN
            if not RIFE_AVAILABLE and CAIN_AVAILABLE:
                self.cain_radio.setChecked(True)
                self.update_method()
        else:
            if not REALESRGAN_AVAILABLE:
                self.realesrgan_radio.setEnabled(False)
                self.realesrgan_radio.setText("Real-ESRGAN (Not Available)")
                
            if not SWINIR_AVAILABLE:
                self.swinir_radio.setEnabled(False)
                self.swinir_radio.setText("SwinIR (Not Available)")
                
            # If Real-ESRGAN is not available but SwinIR is, switch to SwinIR
            if not REALESRGAN_AVAILABLE and SWINIR_AVAILABLE:
                self.swinir_radio.setChecked(True)
                self.update_method()
    
    def update_method(self):
        """Update UI based on selected method"""
        if self.tab_type == "interpolation":
            method = "cain" if self.cain_radio.isChecked() else "rife"
        else:
            method = "realesrgan" if self.realesrgan_radio.isChecked() else "swinir"
        
        # Reset file selection
        self.file_path_label.setText("No file selected")
        self.process_button.setDisabled(True)
        self.original_media = None
        self.enhanced_media = None
        self.enhanced_video_path = None
        self.saved_file_path = None
        if hasattr(self, 'media_label'):
            self.media_label.setPixmap(QPixmap())
        
        self.status_label.setText(f"Ready - {method.upper()} selected")
    
    def browse_file(self):
        """Open file dialog to select a file"""
        if self.tab_type in ["video", "interpolation"]:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", 
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;MP4 Videos (*.mp4);;AVI Videos (*.avi);;MOV Videos (*.mov);;All Files (*)"
            )
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
            )
        
        if file_name:
            self.file_path_label.setText(file_name)
            if self.tab_type == "image":
                self.load_image(file_name)
            else:
                self.load_video_preview(file_name)
            self.process_button.setEnabled(True)
            self.enhanced_button.setEnabled(False)
            self.enhanced_button.setChecked(False)
            self.original_button.setChecked(True)
            self.open_location_button.setEnabled(True)  # Enable for original file
            self.enhanced_video_path = None  # Reset enhanced video path
            self.saved_file_path = None
            self.toggle_view("original")
    
    def load_image(self, path):
        """Load the selected image"""
        try:
            # Load image with OpenCV
            self.original_media = cv2.imread(path)
            self.original_media = cv2.cvtColor(self.original_media, cv2.COLOR_BGR2RGB)
            
            # Ensure the array is contiguous in memory
            if not self.original_media.flags['C_CONTIGUOUS']:
                self.original_media = np.ascontiguousarray(self.original_media)
            
            # Convert to QPixmap and display
            h, w, c = self.original_media.shape
            q_img = QImage(self.original_media.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.media_label.setPixmap(pixmap)
            
            # Reset enhanced media
            self.enhanced_media = None
            self.status_label.setText(f"Image loaded - {w}x{h}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def load_video_preview(self, path):
        """Load a preview frame from the video and setup video playback"""
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Store original video info
            self.original_video_info = {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count
            }
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_media = frame_rgb
                
                # Ensure the array is contiguous
                if not self.original_media.flags['C_CONTIGUOUS']:
                    self.original_media = np.ascontiguousarray(self.original_media)
                
                # Display the frame
                h, w, c = self.original_media.shape
                q_img = QImage(self.original_media.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.media_label.setPixmap(pixmap)
                
                # Reset enhanced media
                self.enhanced_media = None
                self.enhanced_video_path = None
                self.enhanced_video_info = {}
                
                # Setup video player for video tab
                if self.tab_type in ["video", "interpolation"]:
                    self.current_video_path = path
                    self.video_duration = frame_count
                    self.video_fps = fps  # Store actual video FPS
                    duration_seconds = int(frame_count / fps) if fps > 0 else 0
                    
                    # Enable video controls
                    self.play_button.setEnabled(True)
                    self.video_progress.setEnabled(True)
                    
                    # Update video info with original info
                    self.update_video_info_display()
                    self.total_time_label.setText(f"{duration_seconds//60:02d}:{duration_seconds%60:02d}")
                    
                    # Reset video player state
                    if self.media_player:
                        self.media_player.release()
                        self.media_player = None
                    self.is_playing = False
                    self.play_button.setIcon(self.play_icon)
                    self.video_timer.stop()
                    
                    # Update view indicator
                    self.view_indicator.setText("👁️ Viewing: Original")
                
                self.status_label.setText(f"Video loaded - {width}x{height}, {frame_count} frames, {fps:.1f} FPS")
            else:
                QMessageBox.critical(self, "Error", "Failed to read video frame")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video preview: {str(e)}")
    
    def update_video_info_display(self):
        """Update video info display based on current view"""
        if self.tab_type not in ["video", "interpolation"]:
            return
            
        if self.current_view == "original" and self.original_video_info:
            info = self.original_video_info
            self.video_info_label.setText(f"{info['width']}×{info['height']}")
        elif self.current_view == "enhanced" and self.enhanced_video_info:
            info = self.enhanced_video_info
            self.video_info_label.setText(f"{info['width']}×{info['height']}")
        else:
            self.video_info_label.setText("No video loaded")
    
    # Processing Methods
    def process_file(self):
        """Process the selected file"""
        if self.original_media is None:
            return
        
        if self.tab_type == "interpolation":
            current_method = "cain" if self.cain_radio.isChecked() else "rife"
            device = self.device_combo.currentData()
            self.process_interpolation(current_method, device)
        else:
            current_method = "realesrgan" if self.realesrgan_radio.isChecked() else "swinir"
            device = self.device_combo.currentData()
            scale_factor = self.scale_group.checkedId()
            
            if self.tab_type == "video":
                self.process_video(current_method, device, scale_factor)
            else:
                self.process_image(current_method, device, scale_factor)
    
    def process_interpolation(self, method, device):
        """Process video with frame interpolation"""
        # Get output path
        output_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Interpolated Video", "", 
            "MP4 Video (*.mp4);;AVI Video (*.avi);;MOV Video (*.mov)"
        )
        
        if not output_path:
            return
        
        # Get parameters
        fps_multiplier = self.fps_group.checkedId()
        
        # Setup worker thread based on method
        if method == "cain":
            if not CAIN_AVAILABLE:
                QMessageBox.critical(self, "Error", "CAIN is not available")
                return
            
            model_path = self.cain_model_path
            
            self.video_worker = CAINVideoWorker(
                self.file_path_label.text(),
                output_path,
                model_path,
                device,
                fps_multiplier
            )
        else:  # RIFE
            if not RIFE_AVAILABLE:
                QMessageBox.critical(self, "Error", "RIFE is not available")
                return
            
            model_path = self.rife_model_path
            
            self.video_worker = RIFEVideoWorker(
                self.file_path_label.text(),
                output_path,
                model_path,
                device,
                fps_multiplier
            )
        
        # Connect signals
        self.video_worker.progress_updated.connect(self.update_progress)
        self.video_worker.frame_processed.connect(self.update_preview_frame)
        self.video_worker.processing_completed.connect(self.video_processing_done)
        self.video_worker.error_occurred.connect(self.processing_error)
        
        # Disable UI while processing
        self.process_button.setEnabled(False)
        self.process_button.setText("Interpolating...")
        self.status_label.setText(f"Interpolating video with {method.upper()}... x{2**fps_multiplier} FPS")
        
        # Disable video controls during processing
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(False)
            self.video_progress.setEnabled(False)
            # Start processing timer
            self.processing_start_time = time.time()
        
        # Start processing
        self.video_worker.start()
    
    def process_image(self, method, device, scale_factor):
        """Process the image with the selected method"""
        # Get save path for image
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Enhanced Image", "", 
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tiff);;BMP Image (*.bmp);;WEBP Image (*.webp)"
        )
        
        if not save_path:
            return
        
        # Setup worker thread based on method
        if method == "realesrgan":
            if not REALESRGAN_AVAILABLE:
                QMessageBox.critical(self, "Error", "Real-ESRGAN is not available")
                return
            model_path = self.realesrgan_model_path
            device_param = "0" if device == "cuda:0" else "cpu"
            
            self.worker = RealESRGANImageWorker(
                self.file_path_label.text(),
                model_path,
                device_param,
                scale_factor,
                save_path
            )
        else:  # SwinIR
            if not SWINIR_AVAILABLE:
                QMessageBox.critical(self, "Error", "SwinIR is not available")
                return
            model_path = self.swinir_model_path
            
            self.worker = SwinIRImageWorker(
                self.file_path_label.text(),
                model_path,
                device,
                scale_factor,
                save_path
            )
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_completed.connect(self.processing_done)
        self.worker.error_occurred.connect(self.processing_error)
        
        # Disable UI while processing
        self.process_button.setEnabled(False)
        self.process_button.setText("Processing...")
        self.status_label.setText(f"Processing with {method.upper()}...")
        
        # Start processing
        self.worker.start()
    
    def process_video(self, method, device, scale_factor):
        """Process the video with the selected method"""
        # Get output path with multiple format support
        output_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Enhanced Video", "", 
            "MP4 Video (*.mp4);;AVI Video (*.avi);;MOV Video (*.mov)"
        )
        
        if not output_path:
            return
        
        # Determine codec based on file extension
        file_ext = Path(output_path).suffix.lower()
        if file_ext == '.mp4':
            codec = 'mp4v'
        elif file_ext == '.mov':
            codec = 'mp4v'
        else:  # .avi or default
            codec = 'XVID'
        
        # Setup worker thread based on method
        if method == "realesrgan":
            if not REALESRGAN_AVAILABLE:
                QMessageBox.critical(self, "Error", "Real-ESRGAN is not available")
                return
            model_path = self.realesrgan_model_path
            device_param = "0" if device == "cuda:0" else "cpu"
            
            self.video_worker = RealESRGANVideoWorker(
                self.file_path_label.text(),
                output_path,
                model_path,
                device_param,
                scale_factor,
                codec
            )
        else:  # SwinIR
            if not SWINIR_AVAILABLE:
                QMessageBox.critical(self, "Error", "SwinIR is not available")
                return
            model_path = self.swinir_model_path
            
            self.video_worker = SwinIRVideoWorker(
                self.file_path_label.text(),
                output_path,
                model_path,
                device,
                scale_factor,
                codec
            )
        
        # Connect signals
        self.video_worker.progress_updated.connect(self.update_progress)
        self.video_worker.frame_processed.connect(self.update_preview_frame)
        self.video_worker.processing_completed.connect(self.video_processing_done)
        self.video_worker.error_occurred.connect(self.processing_error)
        self.video_worker.path_changed.connect(self.show_path_change_info)
        
        # Disable UI while processing
        self.process_button.setEnabled(False)
        self.process_button.setText("Processing Video...")
        codec_name = "H.264" if codec == "mp4v" else "XVID"
        self.status_label.setText(f"Processing video with {method.upper()} • {codec_name} codec")
        
        # Disable video controls during processing
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(False)
            self.video_progress.setEnabled(False)
            # Start processing timer for video
            self.processing_start_time = time.time()
        
        # Start processing
        self.video_worker.start()
    
    # Processing Event Handlers
    def show_path_change_info(self, message):
        """Show information about path changes due to codec fallback"""
        QMessageBox.information(self, "Output Format Changed", message)
        self.status_label.setText("Codec fallback applied")
    
    def update_preview_frame(self, frame):
        """Update the preview with the processed frame"""
        self.enhanced_media = frame.copy()
        
        # Ensure proper format
        if not self.enhanced_media.flags['C_CONTIGUOUS']:
            self.enhanced_media = np.ascontiguousarray(self.enhanced_media)
        
        if self.enhanced_media.dtype != np.uint8:
            if self.enhanced_media.max() <= 1.0:
                self.enhanced_media = (self.enhanced_media * 255).astype(np.uint8)
            else:
                self.enhanced_media = np.clip(self.enhanced_media, 0, 255).astype(np.uint8)
        
        # Display the enhanced frame
        h, w, c = self.enhanced_media.shape
        q_img = QImage(self.enhanced_media.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.media_label.setPixmap(pixmap)
        
        # Enable enhanced view and auto-switch
        self.enhanced_button.setEnabled(True)
        self.enhanced_button.setChecked(True)
        self.original_button.setChecked(False)
        self.current_view = "enhanced"
        
        # Update view indicator for video
        if self.tab_type in ["video", "interpolation"]:
            if self.tab_type == "interpolation":
                self.view_indicator.setText("👁️ Viewing: Interpolated (Processing)")
            else:
                self.view_indicator.setText("👁️ Viewing: Enhanced (Processing)")
    
    def update_progress(self, value):
        """Update progress bar with enhanced time estimation"""
        self.progress_bar.setValue(value)
        
        # Update time info for video processing
        if self.tab_type in ["video", "interpolation"] and hasattr(self, 'time_label'):
            if value > 0:
                # Calculate estimated time remaining
                if hasattr(self, 'processing_start_time'):
                    elapsed = time.time() - self.processing_start_time
                    if value > 5:  # Avoid division by very small numbers
                        estimated_total = (elapsed / value) * 100
                        remaining = estimated_total - elapsed
                        
                        if remaining > 60:
                            time_text = f"~{int(remaining//60)}m {int(remaining%60)}s remaining"
                        else:
                            time_text = f"~{int(remaining)}s remaining"
                        
                        self.time_label.setText(f"Processing... {value}% • {time_text}")
                    else:
                        self.time_label.setText(f"Processing... {value}%")
                else:
                    self.time_label.setText(f"Processing... {value}%")
            else:
                self.time_label.setText("Initializing...")
    
    def processing_done(self, enhanced_media, save_path):
        """Handle completed image processing"""
        self.enhanced_media = enhanced_media
        self.saved_file_path = save_path  # Store saved file path
        
        # Ensure the array is contiguous in memory and convert to proper format
        if not self.enhanced_media.flags['C_CONTIGUOUS']:
            self.enhanced_media = np.ascontiguousarray(self.enhanced_media)
            
        # Scale values to 0-255 range and convert to uint8
        if self.enhanced_media.dtype != np.uint8:
            if self.enhanced_media.max() <= 1.0:
                self.enhanced_media = (self.enhanced_media * 255).astype(np.uint8)
            else:
                self.enhanced_media = self.enhanced_media.astype(np.uint8)
                
        # Create QImage with proper data format
        h, w, c = self.enhanced_media.shape
        q_img = QImage(self.enhanced_media.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Display enhanced media
        self.media_label.setPixmap(pixmap)
        
        # Enable UI
        self.process_button.setEnabled(True)
        self.process_button.setText(f"Process {self.tab_type.title()}")
        self.open_location_button.setEnabled(True)
        self.enhanced_button.setEnabled(True)
        self.enhanced_button.setChecked(True)
        self.original_button.setChecked(False)
        self.current_view = "enhanced"
        self.status_label.setText(f"Processing completed - {w}x{h}")
        
        # Show success message
        QMessageBox.information(self, "Success", f"Image processed and saved successfully!\n\nLocation: {save_path}")
    
    def video_processing_done(self, output_path, enhanced_video_info=None):
        """Handle completed video processing with enhanced features"""
        # Store the enhanced video path for playback
        self.enhanced_video_path = output_path
        
        # Store enhanced video info if provided
        if enhanced_video_info:
            self.enhanced_video_info = enhanced_video_info
        
        # Enable UI elements
        self.process_button.setEnabled(True)
        if self.tab_type == "interpolation":
            self.process_button.setText("Interpolate Video")
        else:
            self.process_button.setText("Process Video")
        
        # Enable view switching for videos
        if self.enhanced_media is not None:
            self.enhanced_button.setEnabled(True)
            self.enhanced_button.setChecked(True)
            self.original_button.setChecked(False)
            self.current_view = "enhanced"
            # Update video info display
            self.update_video_info_display()
        
        if self.tab_type == "interpolation":
            self.status_label.setText("Video interpolation completed")
        else:
            self.status_label.setText("Video processing completed")
        
        # Re-enable video controls
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(True)
            self.video_progress.setEnabled(True)
            if hasattr(self, 'time_label'):
                self.time_label.setText("Processing completed!")
        
        # Enable open location button
        self.open_location_button.setEnabled(True)
        
        # Update view indicator
        if hasattr(self, 'view_indicator'):
            if self.tab_type == "interpolation":
                self.view_indicator.setText("👁️ Viewing: Interpolated")
            else:
                self.view_indicator.setText("👁️ Viewing: Enhanced")
        
        # Show enhanced success message
        file_ext = Path(output_path).suffix.upper()
        file_name = Path(output_path).name
        
        if self.tab_type == "interpolation":
            success_title = "🎬 Frame Interpolation Complete!"
            features_text = "✨ Features now available:\n• Switch between Original and Interpolated views\n• Play both versions using video controls\n• Compare frame rates side by side"
        else:
            success_title = "🎉 Video Processing Complete!"
            features_text = "✨ Features now available:\n• Switch between Original and Enhanced views\n• Play both versions using video controls\n• Seamless comparison during playback"
        
        QMessageBox.information(self, success_title, 
                              f"Video processed successfully!\n\n"
                              f"📁 File: {file_name}\n"
                              f"📄 Format: {file_ext}\n"
                              f"📍 Location: {output_path}\n\n"
                              f"{features_text}")
    
    def processing_error(self, error_message):
        """Handle processing errors with improved cleanup"""
        QMessageBox.critical(self, "Processing Error", error_message)
        self.process_button.setEnabled(True)
        
        if self.tab_type == "interpolation":
            self.process_button.setText("Interpolate Video")
        else:
            self.process_button.setText(f"Process {self.tab_type.title()}")
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Error occurred during processing")
        
        # Auto clear GPU cache after error
        try:
            if hasattr(self, 'device_combo') and 'cuda' in self.device_combo.currentData():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Re-enable video controls if video tab
        if self.tab_type in ["video", "interpolation"] and hasattr(self, 'play_button'):
            self.play_button.setEnabled(True)
            self.video_progress.setEnabled(True)
            if hasattr(self, 'time_label'):
                self.time_label.setText("Processing failed")
            if hasattr(self, 'view_indicator'):
                self.view_indicator.setText("👁️ Viewing: Original")
    
    # View Management Methods
    def toggle_view(self, view):
        """Enhanced toggle between original and enhanced views with video support"""
        if view == self.current_view:
            return
        
        # Store current playback state and position
        was_playing = False
        current_progress = 0
        
        if hasattr(self, 'is_playing') and self.is_playing:
            was_playing = True
            current_progress = self.video_progress.value()
            self.pause_video()
        elif hasattr(self, 'video_progress'):
            current_progress = self.video_progress.value()
                
        if view == "original" and self.original_media is not None:
            # Display original frame
            h, w, c = self.original_media.shape
            
            if not self.original_media.flags['C_CONTIGUOUS']:
                self.original_media = np.ascontiguousarray(self.original_media)
                
            q_img = QImage(self.original_media.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.media_label.setPixmap(pixmap)
            
            self.original_button.setChecked(True)
            self.enhanced_button.setChecked(False)
            self.current_view = "original"
            
            # Update view indicator and video info for video
            if self.tab_type in ["video", "interpolation"]:
                if hasattr(self, 'view_indicator'):
                    self.view_indicator.setText("👁️ Viewing: Original")
                self.update_video_info_display()
                
                # Reset media player to force reload with correct video path
                if hasattr(self, 'media_player') and self.media_player:
                    self.media_player.release()
                    self.media_player = None
                    if hasattr(self, 'current_player_path'):
                        delattr(self, 'current_player_path')
            
        elif view == "enhanced" and self.enhanced_media is not None:
            # Display enhanced frame
            h, w, c = self.enhanced_media.shape
            
            if not self.enhanced_media.flags['C_CONTIGUOUS']:
                self.enhanced_media = np.ascontiguousarray(self.enhanced_media)
            
            # Ensure proper format
            if self.enhanced_media.dtype != np.uint8:
                if self.enhanced_media.max() <= 1.0:
                    self.enhanced_media = (self.enhanced_media * 255).astype(np.uint8)
                else:
                    self.enhanced_media = self.enhanced_media.astype(np.uint8)
                    
            q_img = QImage(self.enhanced_media.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.media_label.setPixmap(pixmap)
            
            self.original_button.setChecked(False)
            self.enhanced_button.setChecked(True)
            self.current_view = "enhanced"
            
            # Update view indicator and video info for video
            if self.tab_type in ["video", "interpolation"]:
                if hasattr(self, 'view_indicator'):
                    if self.tab_type == "interpolation":
                        self.view_indicator.setText("👁️ Viewing: Interpolated")
                    else:
                        self.view_indicator.setText("👁️ Viewing: Enhanced")
                self.update_video_info_display()
                
                # Reset media player to force reload with correct video path
                if hasattr(self, 'media_player') and self.media_player:
                    self.media_player.release()
                    self.media_player = None
                    if hasattr(self, 'current_player_path'):
                        delattr(self, 'current_player_path')
        
        # Restore playback state for videos
        if self.tab_type in ["video", "interpolation"] and was_playing:
            # Small delay to ensure the view has switched properly
            QTimer.singleShot(100, lambda: self.restore_playback_state(current_progress, True))
        elif self.tab_type in ["video", "interpolation"] and current_progress > 0:
            # Preserve position even if not playing
            QTimer.singleShot(100, lambda: self.restore_playback_state(current_progress, False))

    def restore_playback_state(self, progress, should_play):
        """Helper method to restore playback state after view switching"""
        try:
            if hasattr(self, 'video_progress'):
                self.video_progress.setValue(progress)
                
            if should_play:
                self.play_video()
            else:
                # Just seek to the position without playing
                self.seek_video()
                
        except Exception as e:
            print(f"Error restoring playback state: {e}")

            
class CombinedSRGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle("Super-Resolution & Frame Interpolation Suite")
        self.setMinimumSize(1200, 700)
        
        # Main layout with gradient background
        self.main_widget = StyleWidget()
        self.setCentralWidget(self.main_widget)
        
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 20);
                color: white;
                padding: 12px 32px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 140px;
                min-height: 20px;
                text-align: center;
            }
            QTabBar::tab:selected {
                background: rgba(255, 255, 255, 40);
                color: white;
            }
            QTabBar::tab:hover {
                background: rgba(255, 255, 255, 30);
            }
        """)
        
        # Create tabs
        self.image_tab = ProcessingTab("image")
        self.video_tab = ProcessingTab("video")
        self.interpolation_tab = ProcessingTab("interpolation")
        
        self.tab_widget.addTab(self.image_tab, "Image Processing")
        self.tab_widget.addTab(self.video_tab, "Video Processing")
        self.tab_widget.addTab(self.interpolation_tab, "Frame Interpolation")
        
        # Connect tab change signal to update gradient
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tab_widget)
        
        # Set initial gradient
        self.on_tab_changed(0)
    
    def on_tab_changed(self, index):
        """Handle tab change to update gradient theme"""
        # 0 = image tab (blue gradient), 1 = video tab (purple gradient), 2 = interpolation tab (red-teal gradient)
        self.main_widget.set_gradient(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CombinedSRGUI()
    window.show()
    sys.exit(app.exec_())