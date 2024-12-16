# FrameScope

<div align="center">
    <img src="static/favicon.svg" alt="FrameScope Logo" width="120">
    <h3>Analyze colors, scenes and moods in videos</h3>
</div>


<div align="center">
    <img src="https://github.com/user-attachments/assets/2d70dd60-441f-41c4-bb17-80ee80d26f8b" alt="framescope">
</div>


FrameScope is a powerful web application that analyzes videos to extract color palettes, detect scenes, and identify moods from individual frames. It provides an intuitive interface for video analysis and filtering results based on various criteria.

## Features

### 🎨 Color Analysis
- Extracts dominant color palettes from video frames
- Displays color percentages and hex codes
- One-click color code copying
- Color-based frame filtering

### 🎬 Scene Detection
- Identifies various scene types:
  - Environment (indoor/outdoor)
  - Time (day/night)
  - Setting (urban/nature)
  - Style (vintage/modern)
  - Composition (minimalist/busy)
  - Lighting (bright/dark)
  - Color richness (colorful/monochrome)

### 🎭 Mood Recognition
- Detects emotional attributes:
  - Basic emotions (happy/sad)
  - Atmosphere (dramatic/peaceful)
  - Temperature (warm/cold)
  - Energy (action/calm)
  - Intensity (energetic/relaxed)

### ⚙️ Frame Capture Modes
- **Detailed** (1-2s intervals): Best for short videos
- **Balanced** (3-5s intervals): Good for most videos
- **Overview** (8-10s intervals): For long videos
- **Custom**: User-defined intervals (1-60s)

### 🔍 Smart Filtering
- Filter frames by:
  - Scene types
  - Emotional attributes
  - Color presence
- Multiple filter combination support
- Real-time results updating

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- GPU support (optional, for better performance)

### Step 1: Clone the Repository
```bash
git clone https://github.com/bruuno-studio/framescope.git
cd framescope
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python src/app.py
```

The application will be available at `http://localhost:5000`

## Usage Guide

### 1. Upload Video
- Click the upload area or drag & drop your video
- Supported formats: MP4, AVI, MOV, MKV
- Preview your video before analysis

### 2. Select Frame Capture Mode
- Choose based on video length and detail needs
- Auto-adjustment based on video duration:
  - Under 1 min: User selected
  - 1-5 min: Min 3s intervals
  - 5-15 min: Min 5s intervals
  - 15-30 min: Min 10s intervals
  - Over 30 min: Min 15s intervals

### 3. Analyze Video
- Click "Analyze Video" to start processing
- Progress bar shows completion status
- Results appear automatically when done

### 4. Filter Results
- Use dropdown menus to select scenes/emotions
- Click color squares to filter by color
- Combine multiple filters for precise results
- Remove filters by clicking the × symbol

### 5. Interact with Results
- Click color squares to copy hex codes
- Scroll through scene and emotion details
- View frame timestamps and analysis data

## Performance Optimization

### GPU Acceleration
- Automatically uses GPU if available
- Significantly faster processing
- Supports CUDA-enabled NVIDIA GPUs

### Memory Management
- Efficient frame processing
- Automatic garbage collection
- Optimized for long videos

## FAQ

### Q: What video formats are supported?
A: MP4, AVI, MOV, and MKV formats are supported. For best results, use MP4 format.

### Q: How many frames can be analyzed?
A: Up to 100 frames per video for optimal performance and memory usage.

### Q: Is there a maximum video length?
A: No strict limit, but longer videos will use larger intervals between frames.

### Q: How accurate is the scene/mood detection?
A: The analysis uses advanced AI models with typical accuracy of 85-90% for scenes and 75-80% for moods.

### Q: Can I analyze multiple videos simultaneously?
A: Currently, the application processes one video at a time for optimal performance.

### Q: Does it work offline?
A: Yes, once installed, the application runs completely offline.

## Troubleshooting

### Common Issues

1. **Video Won't Upload**
   - Check file format compatibility
   - Ensure file size is reasonable
   - Try refreshing the page

2. **Analysis Stops**
   - Check available memory
   - Try a larger frame interval
   - Ensure stable internet connection

3. **Slow Performance**
   - Enable GPU acceleration if available
   - Close other resource-intensive applications
   - Try processing shorter video segments

4. **Filter Not Working**
   - Clear all filters and try again
   - Refresh the page
   - Check for minimum confidence thresholds

## Technical Details

### Architecture
- Frontend: HTML5, CSS3, JavaScript
- Backend: Python, Flask
- Analysis: OpenCV, TensorFlow
- Color Processing: ColorThief

### System Requirements
- OS: Windows 10+, macOS 10.14+, Linux
- RAM: 8GB minimum, 16GB recommended
- Storage: 1GB free space
- GPU: Optional, CUDA-compatible

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License

## Acknowledgments

- Color analysis powered by ColorThief
- Scene detection using TensorFlow models
- Icons from Heroicons

---

<div align="center">
    Made with ❤️
</div> 
