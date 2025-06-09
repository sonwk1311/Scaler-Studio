# Super-Resolution & Frame Interpolation Suite

A comprehensive PyQt5-based GUI application for AI-powered image and video enhancement, featuring super-resolution and frame interpolation capabilities.

## 🌟 Features

### 🖼️ Image Processing
- **Super-Resolution**: Enhance image quality and resolution
- **Multiple Models**: Real-ESRGAN and SwinIR support
- **Scaling Options**: 2x, 3x, and 4x upscaling
- **Format Support**: PNG, JPEG, TIFF, BMP, WEBP

### 🎬 Video Processing  
- **Video Super-Resolution**: Enhance video quality frame by frame
- **Multiple Codecs**: MP4 (H.264), AVI (XVID), MOV support
- **Progress Tracking**: Detailed progress with time estimation

### 🎭 Frame Interpolation
- **Smooth Motion**: Increase video frame rates for smoother playback
- **Multiple Algorithms**: CAIN and RIFE interpolation methods
- **FPS Multipliers**: 2x, 4x, and 8x frame rate enhancement
- **Temporal Consistency**: Advanced algorithms for natural motion

## 🚀 Installation

### 1. Clone Repository Structure
Create the following directory structure:
```
SR and FI/
├── SR/
│   ├── Real-ESRGAN/
│   └── SwinIR/
├── FI/
│   ├── CAIN/
│   └── RIFE/
├── img
└── GUI.py
```

### 2. Install Python Dependencies
```bash
pip install torch torchvision opencv-python PyQt5 omegaconf numpy
```
### 3. Download Pre-trained Models

#### Model Paths (Update in GUI.py)
```python
# Update these paths in the ProcessingTab class:
self.realesrgan_model_path = "path/to/realesrgan_model.pkl"
self.swinir_model_path = "path/to/swinir_model.tar" 
self.cain_model_path = "path/to/cain_model.pth"
self.rife_model_path = "path/to/rife_model.pkl"
```

#### Download Links
[https://drive.google.com/drive/folders/1RhvBy_p_gUmcEEOpSp1nl57IDLPrQWSf?usp=drive_link]

## 🎯 Usage Guide

### Image Processing Tab
1. **Select Method**: Choose between Real-ESRGAN or SwinIR
2. **Choose File**: Click "Browse" to select an image
3. **Set Scale**: Select 2x, 3x, or 4x upscaling
4. **Select Device**: Choose GPU (CUDA) or CPU processing
5. **Process**: Click "Process Image" and choose save location
6. **Compare**: Use toggle buttons to compare original vs enhanced

### Video Processing Tab  
1. **Select Method**: Choose Real-ESRGAN or SwinIR
2. **Load Video**: Browse and select video file
3. **Configure**: Set scaling factor and processing device
4. **Process**: Click "Process Video" and set output path
5. **Monitor**: Watch real-time progress and preview
6. **Playback**: Use video controls to play and compare results

### Frame Interpolation Tab
1. **Select Method**: Choose between CAIN or RIFE  
2. **Load Video**: Select input video file
3. **Set Multiplier**: Choose 2x, 4x, or 8x frame rate increase
4. **Configure Device**: Select GPU or CPU processing
5. **Interpolate**: Click "Interpolate Video" and set output
6. **Compare**: Play original vs interpolated versions

## 🎮 Video Player Controls

### Playback Features
- **Play/Pause**: Space bar or click play button
- **Seek**: Click anywhere on progress bar to jump
- **View Toggle**: Switch between original and processed content
- **Time Display**: Current time and total duration
- **Progress Bar**: Visual playback progress with seeking

### Keyboard Shortcuts
- `Space`: Play/Pause
- `Left/Right Arrow`: Frame-by-frame navigation (when paused)
- `1`: Switch to original view  
- `2`: Switch to enhanced/interpolated view

## ⚙️ Configuration

### Device Selection
- **CUDA (GPU)**: Faster processing, requires NVIDIA GPU
- **CPU**: Slower but more compatible, no GPU required

### Quality Settings
- **Real-ESRGAN**: Better for realistic images and videos
- **SwinIR**: Better for anime/artwork content
- **CAIN**: Good general-purpose frame interpolation
- **RIFE**: Advanced interpolation with better motion handling

### Output Formats
- **Images**: PNG, JPEG, TIFF, BMP, WEBP
- **Videos**: MP4 (H.264), AVI (XVID), MOV

## 🙏 Acknowledgments

- **Real-ESRGAN**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **SwinIR**: [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)  
- **CAIN**: [myungsub/CAIN](https://github.com/myungsub/CAIN)
- **RIFE**: [megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
- **PyQt5**: GUI framework
- **OpenCV**: Video processing

**Note**: This application requires significant computational resources. GPU processing is highly recommended for optimal performance.
