# File: /Users/kazirasel/VideoAutomation/System/SysVisuals/Engines/image/imagen4_ai_engine.py
# UPDATED: Uses key_loader.py exclusively for credential path discovery.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imagen4_ai_engine.py

Google Imagen 4 Fast Provider for AIImageEngine.
Updated to use the Vertex AI Publisher Model API.

Usage:
  - Automatically obtains the Google Imagen credential JSON path
    through key_loader.get_imagen_json_path().
"""

from __future__ import annotations

import base64
import os
import json
import sys
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from ai_engine import AIImageEngine

# Import Vertex AI SDK
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic import PredictionServiceClient
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
except ImportError:
    aiplatform = None
    PredictionServiceClient = None

VA_ROOT = Path(os.environ.get("VA_ROOT", str(Path.home() / "VideoAutomation")))

SYS     = VA_ROOT / "System"
LOG_DIR = SYS / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "imagen4_ai_engine.log"

# Centralized key handling via key_loader.py
KEYLOADER_DIR = VA_ROOT / "System" / "AllKeys" / "KeyLoader"
if str(KEYLOADER_DIR) not in sys.path:
    sys.path.insert(0, str(KEYLOADER_DIR))

from key_loader import get_imagen_json_path

def _log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[IMAGEN {timestamp}] {msg}"
    # print to terminal
    print(line, flush=True)
    # append to central log file
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Provider-level config (AUTOMATICALLY UPDATED FROM JSON)
# ---------------------------------------------------------------------------

IMAGEN_LOCATION = os.environ.get("IMAGEN_LOCATION", "us-central1")

# UPDATED: Default to Imagen 4 Fast
IMAGEN_MODEL_NAME = os.environ.get("IMAGEN_MODEL_NAME", "imagen-4.0-fast-generate-001")

IMAGEN_API_ENDPOINT = f"{IMAGEN_LOCATION}-aiplatform.googleapis.com"

imagen_key_path = get_imagen_json_path()
IMAGEN_PROJECT_ID = os.environ.get("IMAGEN_PROJECT_ID")  # Start with Env Var or None

# 2. Extract Project ID from JSON (path provided by key_loader)
if imagen_key_path.exists():
    # Set Auth for Google SDK
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(imagen_key_path)
    
    try:
        with open(imagen_key_path, "r") as f:
            creds = json.load(f)
            if "project_id" in creds:
                IMAGEN_PROJECT_ID = creds["project_id"]
                _log(f"Auto-detected Project ID: {IMAGEN_PROJECT_ID}")
            else:
                _log("Warning: 'project_id' field missing in JSON key.")
    except Exception as e:
        _log(f"Error reading JSON key file: {e}")
else:
    _log(f"Warning: Key file not found at {imagen_key_path}")

# 3. Final Validation
if not IMAGEN_PROJECT_ID:
    raise RuntimeError(
        "CRITICAL: Could not determine Google Cloud Project ID. "
        "Please ensure the credential JSON returned by key_loader.get_imagen_json_path() exists and contains 'project_id'."
    )

# ---------------------------------------------------------------------------
# Low-level Imagen Client
# ---------------------------------------------------------------------------

_client: Optional[PredictionServiceClient] = None

def _ensure_client() -> PredictionServiceClient:
    global _client
    if _client is not None:
        return _client

    if aiplatform is None or PredictionServiceClient is None:
        raise RuntimeError("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")

    client_options = {"api_endpoint": IMAGEN_API_ENDPOINT}
    _client = PredictionServiceClient(client_options=client_options)
    return _client

def _imagen_predict(prompt: str, negative: Optional[str] = None, seed: Optional[int] = None) -> bytes:
    """
    Call Imagen via Vertex AI Publisher API using dynamic Project ID.
    """
    client = _ensure_client()

    # 1. Construct the Publisher Model Endpoint Path
    endpoint = (
        f"projects/{IMAGEN_PROJECT_ID}/locations/{IMAGEN_LOCATION}/"
        f"publishers/google/models/{IMAGEN_MODEL_NAME}"
    )

    # 2. Build Request Instances (The Prompt)
    # FIX: Using simple list of dicts avoids MapComposite/DESCRIPTOR errors
    instances = [
        {
            "prompt": prompt,
        }
    ]
    
    # 3. Build Request Parameters (Config)
    parameters = {
        "sampleCount": 1,
        "aspectRatio": "4:3",
        "addWatermark": False
    }

    if negative:
        parameters["negativePrompt"] = negative
    if seed is not None:
        parameters["seed"] = int(seed)

    _log(f"Generating with {IMAGEN_MODEL_NAME} on {IMAGEN_PROJECT_ID}...")
    
    # 4. Call the API
    try:
        # Passing python dicts directly is supported and safer across versions
        response = client.predict(
            endpoint=endpoint,
            instances=instances,
            parameters=parameters
        )
    except Exception as e:
        _log(f"API Call Failed: {e}")
        raise RuntimeError(f"Imagen generation failed: {e}")

    if not response.predictions:
        raise RuntimeError("Imagen: No predictions returned.")

    # 5. Extract Image safely
    try:
        prediction = response.predictions[0]
        
        # Safe conversion from Protobuf Map/Struct to Dict
        if hasattr(prediction, "items"):
            pred_dict = dict(prediction.items())
        else:
            pred_dict = prediction # Fallback
            
        if "bytesBase64Encoded" in pred_dict:
            img_b64 = pred_dict["bytesBase64Encoded"]
        else:
            raise RuntimeError(f"Missing 'bytesBase64Encoded' in response: {pred_dict.keys() if isinstance(pred_dict, dict) else pred_dict}")

    except Exception as e:
         raise RuntimeError(f"Failed to parse prediction response: {e}")

    return base64.b64decode(img_b64)

# ---------------------------------------------------------------------------
# Public interface implementation
# ---------------------------------------------------------------------------

@dataclass
class Imagen4Result:
    width: Optional[int]
    height: Optional[int]
    luma: Optional[float]

def generate_one(prompt: str, out_path: Path, seed: Optional[int] = None, negative: Optional[str] = None) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
    from PIL import Image
    import numpy as np

    try:
        img_bytes = _imagen_predict(prompt, negative=negative, seed=seed)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        # Measure dimensions and luma for system consistency
        with Image.open(str(out_path)) as im:
            im = im.convert("RGB")
            w, h = im.size
            arr = np.array(im).astype("float32")
            luma = float((0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]).mean())
            return (w, h), luma

    except Exception as e:
        _log(f"Error generating image: {e}")
        return None, None

def generate_batch(prompts: List[str], out_dir: Path, seed_base: Optional[int] = None) -> List[Tuple[Optional[Tuple[int, int]], Optional[float]]]:
    results = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, p in enumerate(prompts):
        seed = (seed_base + idx) if seed_base is not None else None
        out_path = out_dir / f"img_{idx:03d}.jpg"
        res = generate_one(p, out_path, seed=seed, negative=None)
        results.append(res)
    return results

class Imagen4AIEngine(AIImageEngine):
    def __init__(self):
        _log(f"Initialized Imagen Engine")

    def generate(
        self,
        prompt,
        out_path,
        seed=None,
        *,
        negative=None,
        topic=None,
        role=None,
        concept_id=None,
        **kwargs,
    ):
        return generate_one(prompt, Path(out_path), seed=seed, negative=negative)

    def generate_batch(
        self,
        prompts: List[str],
        out_dir: Path,
        seed_base: Optional[int] = None,
        *,
        negative=None,
        topic=None,
        role=None,
        concept_id=None,
        **kwargs,
    ):
        return generate_batch(prompts, Path(out_dir), seed_base=seed_base)