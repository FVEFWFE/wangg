"""
RunPod Serverless Handler for Wan 2.2 Animate

Animates a character image using motion from a reference video.
Takes a character face image + performer video, outputs character
doing the performer's movements/expressions.
"""

import os
import sys

# Force CUDA and completely disable XPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_XPU"] = "0"
os.environ["USE_XPU"] = "0"
os.environ["INTEL_XPU_BACKEND"] = "0"
os.environ["SYCL_DEVICE_FILTER"] = ""
os.environ["ZE_AFFINITY_MASK"] = ""
# Disable Intel oneAPI
os.environ["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"

# Block XPU module loading by pre-registering empty module
import types
fake_xpu_module = types.ModuleType('torch.xpu')
fake_xpu_module.is_available = lambda: False
fake_xpu_module.device_count = lambda: 0
sys.modules['torch.xpu'] = fake_xpu_module

import torch

# Ensure torch.xpu points to our fake module
torch.xpu = fake_xpu_module

# Fix torch.distributed.device_mesh for PyTorch < 2.3
# Diffusers expects this module which was added in PyTorch 2.3
if not hasattr(torch.distributed, 'device_mesh'):
    # Create a mock device_mesh module
    import types
    device_mesh_module = types.ModuleType('torch.distributed.device_mesh')

    class DeviceMesh:
        """Mock DeviceMesh for compatibility."""
        def __init__(self, device_type="cuda", mesh=None):
            self.device_type = device_type
            self.mesh = mesh or [[0]]
        def get_rank(self): return 0
        def get_local_rank(self): return 0
        def size(self, dim=0): return 1
        def get_group(self, mesh_dim=None): return None

    device_mesh_module.DeviceMesh = DeviceMesh
    device_mesh_module.init_device_mesh = lambda *args, **kwargs: DeviceMesh()

    # Register the mock module
    import sys
    sys.modules['torch.distributed.device_mesh'] = device_mesh_module
    torch.distributed.device_mesh = device_mesh_module
    print("[Handler] Applied device_mesh compatibility patch")

# ALWAYS replace torch.xpu - RunPod base image has broken/incomplete FakeXPU
# that causes AttributeError: 'FakeXPU' object has no attribute 'empty_cache'

# Create FakeEvent class that inherits from appropriate base for PyTorch 2.4+
# This fixes: "DeviceInterface member Event should be inherit from _EventBase"
try:
    # PyTorch 2.4+ has torch.Event as the base class
    if hasattr(torch, 'Event'):
        class FakeXPUEvent(torch.Event):
            def __init__(self, enable_timing=False, blocking=False, interprocess=False):
                pass
            def record(self, stream=None): pass
            def wait(self, stream=None): pass
            def query(self): return True
            def elapsed_time(self, end_event): return 0.0
            def synchronize(self): pass
    elif hasattr(torch, '_EventBase'):
        class FakeXPUEvent(torch._EventBase):
            def __init__(self, enable_timing=False, blocking=False, interprocess=False):
                pass
            def record(self, stream=None): pass
            def wait(self, stream=None): pass
            def query(self): return True
            def elapsed_time(self, end_event): return 0.0
            def synchronize(self): pass
    else:
        # Fallback for older PyTorch
        class FakeXPUEvent:
            def __init__(self, enable_timing=False, blocking=False, interprocess=False):
                pass
            def record(self, stream=None): pass
            def wait(self, stream=None): pass
            def query(self): return True
            def elapsed_time(self, end_event): return 0.0
            def synchronize(self): pass
except Exception as e:
    print(f"[Handler] FakeXPUEvent fallback: {e}")
    class FakeXPUEvent:
        def __init__(self, enable_timing=False, blocking=False, interprocess=False):
            pass
        def record(self, stream=None): pass
        def wait(self, stream=None): pass
        def query(self): return True
        def elapsed_time(self, end_event): return 0.0
        def synchronize(self): pass

class CompleteFakeXPU:
    """Complete mock of torch.xpu for CUDA-only environments."""
    Event = FakeXPUEvent  # Required by PyTorch 2.4+ DeviceInterface

    def is_available(self): return False
    def device_count(self): return 0
    def empty_cache(self): pass
    def synchronize(self, device=None): pass
    def current_device(self): return 0
    def set_device(self, device): pass
    def manual_seed(self, seed): pass
    def manual_seed_all(self, seed): pass
    def get_rng_state(self, device=None): return torch.ByteTensor()
    def set_rng_state(self, new_state, device=None): pass
    def max_memory_allocated(self, device=None): return 0
    def memory_allocated(self, device=None): return 0
    def memory_reserved(self, device=None): return 0
    def reset_peak_memory_stats(self, device=None): pass
    def mem_get_info(self, device=None): return (0, 0)
    def memory_stats(self, device=None): return {}
    def memory_summary(self, device=None, abbreviated=False): return ""
    def get_device_name(self, device=None): return "FakeXPU"
    def __getattr__(self, name):
        """Catch-all for any other xpu methods diffusers might call."""
        def fake_method(*args, **kwargs): return None
        return fake_method

torch.xpu = CompleteFakeXPU()  # Unconditional replacement
print("[Handler] Applied FakeXPU with Event class")

# Fix PyTorch 2.3+ pytree compatibility - register_pytree_node was moved/removed
# Some older diffusers/transformers code still tries to access it
import torch.utils._pytree as _pytree
if not hasattr(_pytree, 'register_pytree_node'):
    # PyTorch 2.3+ removed this, add a compatibility shim
    def _register_pytree_node_compat(
        cls,
        flatten_fn,
        unflatten_fn,
        *,
        serialized_type_name=None,
        to_dumpable_context=None,
        from_dumpable_context=None,
    ):
        """Compatibility shim for removed register_pytree_node function."""
        # Try the internal _register_pytree_node if it exists
        if hasattr(_pytree, '_register_pytree_node'):
            try:
                _pytree._register_pytree_node(
                    cls,
                    flatten_fn,
                    unflatten_fn,
                    serialized_type_name=serialized_type_name,
                )
            except TypeError:
                # Signature mismatch, just skip
                pass
        # If no internal function exists, this becomes a no-op
        # The types are likely already registered via C++ in PyTorch 2.3+
    _pytree.register_pytree_node = _register_pytree_node_compat
    print("[Handler] Applied pytree compatibility patch")

import runpod
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

        # Explicitly set device to avoid XPU detection issues
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Handler] Using device: {device}")

        model_id = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-Animate-14B-Diffusers")

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        # Enable memory optimizations
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()

        print("[Handler] Model loaded successfully")

    except Exception as e:
        print(f"[Handler] Error loading model: {e}")
        import traceback
        traceback.print_exc()
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

        # Set random seed - use CPU generator to avoid device issues
        generator = torch.Generator().manual_seed(seed)

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
