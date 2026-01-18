"""
RunPod Serverless Handler for Wan 2.2 Animate

Animates a character image using motion from a reference video.
Takes a character face image + performer video, outputs character
doing the performer's movements/expressions.
"""

import runpod
import torch
import os
import tempfile
import requests
import base64
from io import BytesIO
from PIL import Image

# Global model - loaded once at cold start
pipe = None


def download_file(url: str, suffix: str = "") -> str:
    """Download a file from URL to temp location."""
    print(f"[Handler] Downloading: {url[:80]}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()

    print(f"[Handler] Downloaded to: {temp_file.name}")
    return temp_file.name


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_base64(b64_string: str) -> Image.Image:
    """Load image from base64 string."""
    # Remove data URL prefix if present
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


def save_base64_to_file(b64_string: str, suffix: str) -> str:
    """Save base64 data to temp file."""
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    data = base64.b64decode(b64_string)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data)
    temp_file.close()
    return temp_file.name


def load_model():
    """Load Wan 2.2 Animate pipeline."""
    global pipe

    if pipe is not None:
        return pipe

    print("[Handler] Loading Wan 2.2 Animate model...")

    try:
        from diffusers import DiffusionPipeline

        model_id = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-Animate-14B-Diffusers")

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )

        # Move to GPU
        pipe = pipe.to("cuda")

        # Enable memory optimizations
        if hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()

        print("[Handler] Model loaded successfully")

    except Exception as e:
        print(f"[Handler] Error loading model: {e}")
        raise

    return pipe


def extract_frames_from_video(video_path: str, max_frames: int = 81):
    """Extract frames from video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    print(f"[Handler] Extracted {len(frames)} frames from video")
    return frames


def handler(job):
    """
    RunPod handler function.

    Input (supports both URL and base64):
    {
        "input": {
            // Image input (use one)
            "image_url": "https://...",
            "image_base64": "data:image/png;base64,...",

            // Video input (use one)
            "video_url": "https://...",
            "video_base64": "data:video/mp4;base64,...",

            // Generation parameters
            "prompt": "description of animation style",
            "negative_prompt": "what to avoid",
            "seed": 12345,
            "width": 832,
            "height": 480,
            "fps": 16,
            "num_frames": 81,
            "guidance_scale": 5.0,
            "num_inference_steps": 20
        }
    }

    Output:
    {
        "video": "data:video/mp4;base64,...",
        "duration": 5.0
    }
    """
    video_path = None
    output_path = None

    try:
        job_input = job.get("input", {})

        # Get image input
        image = None
        if job_input.get("image_url"):
            image = load_image_from_url(job_input["image_url"])
        elif job_input.get("image_base64"):
            image = load_image_from_base64(job_input["image_base64"])

        if image is None:
            return {"error": "image_url or image_base64 is required"}

        # Get video input
        if job_input.get("video_url"):
            video_path = download_file(job_input["video_url"], suffix=".mp4")
        elif job_input.get("video_base64"):
            video_path = save_base64_to_file(job_input["video_base64"], suffix=".mp4")

        if video_path is None:
            return {"error": "video_url or video_base64 is required"}

        # Get parameters
        prompt = job_input.get("prompt", "natural movement, high quality, realistic")
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, deformed")
        seed = job_input.get("seed", 12345)
        width = job_input.get("width", 832)
        height = job_input.get("height", 480)
        fps = job_input.get("fps", 16)
        num_frames = job_input.get("num_frames", 81)
        guidance_scale = job_input.get("guidance_scale", 5.0)
        num_inference_steps = job_input.get("num_inference_steps", 20)

        print(f"[Handler] Processing with params: {width}x{height}, {num_frames} frames, seed={seed}")

        # Load model
        pipe = load_model()

        # Set random seed
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Extract reference frames from video
        reference_frames = extract_frames_from_video(video_path, max_frames=num_frames)

        # Resize image to target dimensions
        image = image.resize((width, height), Image.LANCZOS)

        # Run the pipeline
        print("[Handler] Running inference...")

        output = pipe(
            image=image,
            video=reference_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=min(num_frames, len(reference_frames)),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Get output frames
        frames = output.frames[0] if hasattr(output, 'frames') else output.images

        # Export to video file
        from diffusers.utils import export_to_video

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        export_to_video(frames, output_path, fps=fps)

        print(f"[Handler] Generated video at {output_path}")

        # Read video and return as base64
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Calculate duration
        duration = len(frames) / fps

        return {
            "video": f"data:video/mp4;base64,{video_base64}",
            "duration": duration,
            "num_frames": len(frames),
            "resolution": f"{width}x{height}",
        }

    except Exception as e:
        print(f"[Handler] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Cleanup temp files
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass


# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
