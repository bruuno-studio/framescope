import os
import time
from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from colorthief import ColorThief
from PIL import Image
import io
import ffmpeg
import torch
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import random
import gc

# Get the absolute path of the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, 
           template_folder=TEMPLATE_DIR,
           static_folder=STATIC_DIR)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
FRAMES_FOLDER = os.path.join(STATIC_DIR, 'frames')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FRAMES = 100  # Maximum number of frames to analyze
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(FRAMES_FOLDER).mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_old_files():
    """Clean up old files from uploads and frames folders"""
    for folder in [UPLOAD_FOLDER, FRAMES_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path) and (time.time() - os.path.getmtime(file_path)) > 3600:  # 1 hour
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")

class VideoAnalyzer:
    def __init__(self):
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize CLIP model for scene understanding and captioning
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize image captioning model
        self.caption_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(self.device)
        self.caption_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)  # Increased for better parallelization
        
        # Set models to evaluation mode
        self.clip_model.eval()
        self.caption_model.eval()
        
        # Common caption templates for better descriptions
        self.caption_templates = [
            "This image shows {}",
            "The scene depicts {}",
            "In this frame, {}",
            "A view of {}",
            "The image captures {}"
        ]

    def extract_frames(self, video_path, interval=2):
        """Extract frames from video at specified interval"""
        frames = []
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Error: Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Adjust interval based on video length
            if duration > 1800:  # > 30 minutes
                interval = max(interval, 15)  # 15 seconds for very long videos
            elif duration > 900:  # 15-30 minutes
                interval = max(interval, 10)  # 10 seconds
            elif duration > 300:  # 5-15 minutes
                interval = max(interval, 5)   # 5 seconds
            elif duration > 60:   # 1-5 minutes
                interval = max(interval, 3)   # 3 seconds
            # For videos under 1 minute, use the user-selected interval
            
            frame_interval = int(fps * interval)
            frames_to_extract = min(MAX_FRAMES, total_frames // frame_interval)
            
            frame_count = 0
            frames_extracted = 0
            
            while cap.isOpened() and frames_extracted < frames_to_extract:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Resize frame to reduce memory usage
                    height, width = frame.shape[:2]
                    if width > 1280:  # Limit max width
                        scale = 1280 / width
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    
                    frames.append(frame)
                    frame_filename = f"frame_{uuid.uuid4()}.jpg"
                    frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(os.path.join('static', 'frames', frame_filename))
                    frames_extracted += 1
                
                frame_count += 1
                
                # Release memory periodically
                if frame_count % 100 == 0:
                    gc.collect()
            
            cap.release()
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
            
        return frames, frame_paths
    
    def generate_caption(self, frame):
        """Generate descriptive caption for the frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize image if too large
            if pil_image.size[0] > 640:
                pil_image.thumbnail((640, 640))
            
            inputs = self.caption_processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to GPU if available
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
                outputs = self.caption_model.generate(
                    **inputs,
                    max_length=30,
                    num_return_sequences=2,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            
            # Move outputs back to CPU for processing
            outputs = outputs.cpu()
            
            captions = []
            for output in outputs:
                caption = self.caption_processor.decode(output, skip_special_tokens=True)
                captions.append(caption)
            
            # Get the most detailed caption
            main_caption = max(captions, key=len)
            
            # Add some variety to the caption presentation
            template = random.choice(self.caption_templates)
            formatted_caption = template.format(main_caption.lower())
            
            return {
                'main_caption': formatted_caption,
                'alternative_captions': captions
            }
        except Exception as e:
            print(f"Error generating caption: {e}")
            return {
                'main_caption': "Unable to generate caption",
                'alternative_captions': []
            }

    def analyze_frame(self, frame):
        """Analyze a single frame (for parallel processing)"""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            results = {
                'color_palette': self.get_color_palette(frame),
                'scene_analysis': self.analyze_scene(frame)
            }
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return results
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return None

    def get_color_palette(self, frame, color_count=5):
        """Extract dominant colors from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        color_thief = ColorThief(img_byte_arr)
        palette = color_thief.get_palette(color_count=color_count)
        
        return [{'r': r, 'g': g, 'b': b, 'hex': '#{:02x}{:02x}{:02x}'.format(r, g, b)} for r, g, b in palette]

    def analyze_scene(self, frame):
        """Analyze scene content using CLIP"""
        try:
            scene_labels = [
                'indoor', 'outdoor', 'day', 'night', 'urban', 'nature',
                'happy', 'sad', 'dramatic', 'peaceful', 'warm', 'cold',
                'vintage', 'modern', 'minimalist', 'busy',
                'bright', 'dark', 'colorful', 'monochrome',
                'action', 'calm', 'energetic', 'relaxed'
            ]
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize image if too large
            if pil_image.size[0] > 640:
                pil_image.thumbnail((640, 640))
            
            inputs = self.clip_processor(
                text=scene_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to GPU if available
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Move results back to CPU for processing
            probs = probs.cpu()
            
            return {label: float(prob) for label, prob in zip(scene_labels, probs[0])}
        except Exception as e:
            print(f"Error analyzing scene: {e}")
            return {label: 0.0 for label in scene_labels}

video_analyzer = VideoAnalyzer()

@app.route('/')
def index():
    clean_old_files()  # Clean up old files
    return render_template('index.html')

def generate_analysis(video_path, frame_interval):
    try:
        frames, frame_paths = video_analyzer.extract_frames(video_path, interval=float(frame_interval))
        total_frames = len(frames)
        
        if total_frames == 0:
            yield f"data: {json.dumps({'error': 'No frames could be extracted from the video'})}\n\n"
            return
        
        results = []
        
        # Analyze frames in parallel
        futures = []
        for frame in frames:
            future = video_analyzer.executor.submit(video_analyzer.analyze_frame, frame)
            futures.append(future)
        
        # Process results as they complete
        for i, (future, frame_path) in enumerate(zip(futures, frame_paths)):
            if i % 2 == 0:
                progress = {
                    'status': 'processing',
                    'current': i + 1,
                    'total': total_frames,
                    'message': f'Analyzing frame {i + 1} of {total_frames}'
                }
                yield f"data: {json.dumps(progress)}\n\n"
            
            analysis = future.result()
            if analysis:
                frame_result = {
                    'frame_number': i + 1,
                    'frame_path': frame_path,
                    **analysis
                }
                results.append(frame_result)
        
        final_response = {
            'status': 'success',
            'frames_analyzed': len(results),
            'results': results
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        
    except Exception as e:
        error_response = {
            'status': 'error',
            'message': str(e)
        }
        yield f"data: {json.dumps(error_response)}\n\n"

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    if video.filename == '' or not allowed_file(video.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        frame_interval = float(request.form.get('frame_interval', 2.0))
    except ValueError:
        frame_interval = 2.0
        
    try:
        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{video.filename}")
        video.save(video_path)
        
        return Response(
            generate_analysis(video_path, frame_interval),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)