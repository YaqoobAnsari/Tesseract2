"""
Tesseract++ Web Application
Academic-grade web interface for floorplan to graph conversion
"""

import os
import sys
import json
import time
import uuid
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Add required paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

# Import app utilities
from utils.app_utils.api.models import ProcessingResponse, GraphVisualization, ExampleImage
from utils.app_utils.api.processing import ProcessingPipeline
from utils.app_utils.visualization.graph_converter import convert_to_cytoscape

# Initialize FastAPI app
app = FastAPI(
    title="Tesseract++ Floorplan Analyzer",
    description="Convert architectural floorplans to navigable graphs",
    version="1.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_LIMIT_MB = 10
PROCESSING_TIMEOUT = 180  # 3 minutes

# Resolve all paths relative to this file
_BASE_DIR = Path(__file__).parent
MODEL_WEIGHTS_DIR = _BASE_DIR / "Model_weights"
INPUT_IMAGES_DIR = _BASE_DIR / "Input_Images"
RESULTS_DIR = _BASE_DIR / "Results"

# Curated example images (order matters for display)
CURATED_EXAMPLES = [
    "FF part 1upE.png",
    "FF part 2up.png",
    "FF part 3upE.png",
    "SF part 1upE.png",
]

# Session storage (in-memory, cleared on restart)
active_sessions: Dict[str, Dict[str, Any]] = {}

# Processing pipeline instance
pipeline = None

class HealthCheck(BaseModel):
    status: str
    models_loaded: bool
    example_images: int
    message: str

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global pipeline

    print("=" * 50)
    print("Tesseract++ Web Application Starting...")
    print("=" * 50)

    # Check model weights
    model_checks = {
        "CRAFT Text Detector": "craft_mlt_25k.pth",
        "Text Interpreter": "None-VGG-BiLSTM-CTC.pth",
        "Door Detector": "door_mdl_32.pth"
    }

    missing_models = []
    for model_name, weight_file in model_checks.items():
        weight_path = MODEL_WEIGHTS_DIR / weight_file
        if not weight_path.exists():
            missing_models.append(f"{model_name} ({weight_file})")
        else:
            print(f"  {model_name} weights found")

    if missing_models:
        error_msg = f"Missing model weights in {MODEL_WEIGHTS_DIR}:\n" + "\n".join(missing_models)
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)

    # Initialize processing pipeline
    try:
        pipeline = ProcessingPipeline()
        print("  Processing pipeline initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize pipeline: {e}")
        raise

    # Check curated example images
    found = sum(1 for name in CURATED_EXAMPLES if (INPUT_IMAGES_DIR / name).exists())
    print(f"  Found {found}/{len(CURATED_EXAMPLES)} curated example images")

    print("=" * 50)
    print("Application ready!")
    print("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    for session_id, session_data in active_sessions.items():
        if "temp_file" in session_data and session_data["temp_file"] and os.path.exists(session_data["temp_file"]):
            os.remove(session_data["temp_file"])
    active_sessions.clear()

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    models_loaded = pipeline is not None
    found = sum(1 for name in CURATED_EXAMPLES if (INPUT_IMAGES_DIR / name).exists())

    return HealthCheck(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        example_images=found,
        message="System ready for processing" if models_loaded else "Models not loaded"
    )

@app.get("/api/examples", response_model=List[ExampleImage])
async def get_example_images():
    """Get list of curated example images"""
    examples = []

    for img_name in CURATED_EXAMPLES:
        img_path = INPUT_IMAGES_DIR / img_name
        if not img_path.exists():
            continue

        stat = img_path.stat()
        has_cached = pipeline.has_cached_result(img_name) if pipeline else False

        examples.append(ExampleImage(
            name=img_name,
            display_name=img_path.stem.replace("_", " "),
            size_kb=round(stat.st_size / 1024, 1),
            has_cached_result=has_cached
        ))

    return examples

@app.get("/api/example-image/{image_name}")
async def get_example_image(image_name: str):
    """Serve example image thumbnail"""
    image_path = INPUT_IMAGES_DIR / image_name
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Example image not found")

    return FileResponse(image_path, media_type="image/png")

@app.get("/api/floorplan-image/{image_name}")
async def get_floorplan_image(image_name: str):
    """Serve original floorplan image for background overlay"""
    # Check Input_Images for example images
    image_path = INPUT_IMAGES_DIR / image_name
    if image_path.exists() and image_path.is_file():
        return FileResponse(image_path, media_type="image/png")

    # Check temp storage for uploaded images
    if pipeline:
        temp_path = pipeline.temp_dir / image_name
        if temp_path.exists() and temp_path.is_file():
            return FileResponse(temp_path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Floorplan image not found")

@app.get("/api/cached-result/{image_name}")
async def get_cached_result(image_name: str):
    """Get pre-computed result for a cached example image"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    cached = pipeline.get_cached_result(image_name)
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached result for this image")

    cytoscape_data = convert_to_cytoscape(cached["graph_json"])
    pre_pruning_cytoscape = (
        convert_to_cytoscape(cached["pre_pruning_graph_json"])
        if cached.get("pre_pruning_graph_json") else None
    )

    return ProcessingResponse(
        session_id=str(uuid.uuid4()),
        status="success",
        image_name=image_name,
        processing_time=0.0,
        graph_data=cytoscape_data,
        pre_pruning_graph_data=pre_pruning_cytoscape,
        statistics={
            "total_nodes": cached["stats"]["total_nodes"],
            "total_edges": cached["stats"]["total_edges"],
            "node_types": cached["stats"]["node_types"],
            "pruning_reduction": cached["stats"].get("pruning_reduction", 0)
        },
        message=f"Loaded cached result for {image_name}"
    )

@app.post("/api/process")
async def process_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    example: Optional[str] = None
):
    """Process uploaded image or example image"""

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Check if user already has active processing
    client_ip = request.client.host
    for sid, data in active_sessions.items():
        if data.get("client_ip") == client_ip and data.get("status") == "processing":
            raise HTTPException(
                status_code=429,
                detail="Already processing an image. Please wait for completion."
            )

    try:
        # Determine image source
        if file and file.filename:
            # Check file size
            contents = await file.read()
            size_mb = len(contents) / (1024 * 1024)
            if size_mb > UPLOAD_LIMIT_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {UPLOAD_LIMIT_MB}MB"
                )

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_file.write(contents)
            temp_file.close()

            image_path = temp_file.name
            image_name = file.filename
            is_example = False

        elif example:
            # Check for cached result first
            if pipeline:
                cached = pipeline.get_cached_result(example)
                if cached is not None:
                    cytoscape_data = convert_to_cytoscape(cached["graph_json"])
                    pre_pruning_cytoscape = (
                        convert_to_cytoscape(cached["pre_pruning_graph_json"])
                        if cached.get("pre_pruning_graph_json") else None
                    )
                    return ProcessingResponse(
                        session_id=session_id,
                        status="success",
                        image_name=example,
                        processing_time=0.0,
                        graph_data=cytoscape_data,
                        pre_pruning_graph_data=pre_pruning_cytoscape,
                        statistics={
                            "total_nodes": cached["stats"]["total_nodes"],
                            "total_edges": cached["stats"]["total_edges"],
                            "node_types": cached["stats"]["node_types"],
                            "pruning_reduction": cached["stats"].get("pruning_reduction", 0)
                        },
                        message=f"Loaded cached result for {example}"
                    )

            # No cache — run full pipeline
            image_path = str(INPUT_IMAGES_DIR / example)
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Example image not found")

            image_name = example
            is_example = True

        else:
            raise HTTPException(status_code=400, detail="No image provided")

        # Store session info
        active_sessions[session_id] = {
            "client_ip": client_ip,
            "status": "processing",
            "start_time": time.time(),
            "image_name": image_name,
            "temp_file": image_path if not is_example else None
        }

        # Process image synchronously
        start_time = time.time()

        try:
            # Run processing pipeline
            result = await asyncio.to_thread(
                pipeline.process_image,
                image_path,
                image_name,
                timeout=PROCESSING_TIMEOUT
            )

            processing_time = time.time() - start_time

            # Convert graph to Cytoscape format
            cytoscape_data = convert_to_cytoscape(result["graph_json"])
            pre_pruning_cytoscape = (
                convert_to_cytoscape(result["pre_pruning_graph_json"])
                if result.get("pre_pruning_graph_json") else None
            )

            # Prepare response
            response = ProcessingResponse(
                session_id=session_id,
                status="success",
                image_name=image_name,
                processing_time=processing_time,
                graph_data=cytoscape_data,
                pre_pruning_graph_data=pre_pruning_cytoscape,
                statistics={
                    "total_nodes": result["stats"]["total_nodes"],
                    "total_edges": result["stats"]["total_edges"],
                    "node_types": result["stats"]["node_types"],
                    "pruning_reduction": result["stats"].get("pruning_reduction", 0)
                },
                message=f"Successfully processed {image_name}"
            )

            # Update session
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["result"] = response.dict()

            # Save uploaded image for floorplan overlay, then cleanup temp
            if not is_example and os.path.exists(image_path):
                if pipeline:
                    overlay_path = pipeline.temp_dir / image_name
                    shutil.copy2(image_path, str(overlay_path))
                os.remove(image_path)

            return response

        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Processing timeout exceeded ({PROCESSING_TIMEOUT}s)"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )

    finally:
        # Cleanup session after some time
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "completed"

@app.get("/api/session/{session_id}")
async def get_session_result(session_id: str):
    """Get result for a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    if session["status"] == "processing":
        return {"status": "processing", "message": "Still processing..."}

    return session.get("result", {"status": "error", "message": "No result available"})

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and its data"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if "temp_file" in session and session["temp_file"] and os.path.exists(session["temp_file"]):
            os.remove(session["temp_file"])
        del active_sessions[session_id]
        return {"message": "Session cleared"}

    return {"message": "Session not found"}

# Mount static files for React frontend (Vite build output goes to dist/)
frontend_dir = Path(__file__).parent / "utils" / "app_utils" / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "message": "Tesseract++ API is running. Frontend not built yet.",
            "docs": "/docs",
            "health": "/health"
        }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,  # Set to True for development
        log_level="info"
    )
