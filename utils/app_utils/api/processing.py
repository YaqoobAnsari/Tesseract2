"""
Processing pipeline wrapper for the Tesseract++ system
Wraps Main.py functionality for web API usage
"""

import os
import sys
import json
import signal
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import threading

# Add required paths
base_path = Path(__file__).parent.parent.parent.parent  # Back to Tesseract++ root
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path / "Models" / "Text_Models"))
sys.path.insert(0, str(base_path / "Models" / "Interpreter"))
sys.path.insert(0, str(base_path / "Models" / "Door_Models"))
sys.path.insert(0, str(base_path / "utils"))

# Import main processing functions
import Main

class TimeoutException(Exception):
    """Custom exception for processing timeout"""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def signal_handler(signum, frame):
        raise TimeoutException("Processing timeout")

    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm

class ProcessingPipeline:
    """
    Wrapper for the Tesseract++ processing pipeline
    """

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent.parent
        self.input_images_dir = self.base_path / "Input_Images"
        self.results_dir = self.base_path / "Results"
        self.temp_dir = self.base_path / "temp_processing"

        # Create temp directory if not exists
        self.temp_dir.mkdir(exist_ok=True)

        # Verify model weights exist
        self._verify_models()

    def _verify_models(self):
        """Verify all required model weights are present"""
        weights_dir = self.base_path / "Model_weights"
        required_weights = [
            "craft_mlt_25k.pth",
            "None-VGG-BiLSTM-CTC.pth",
            "door_mdl_32.pth"
        ]

        for weight_file in required_weights:
            weight_path = weights_dir / weight_file
            if not weight_path.exists():
                raise FileNotFoundError(f"Required model weight not found: {weight_path}")

    def process_image(self, image_path: str, image_name: str, timeout: int = 180) -> Dict[str, Any]:
        """
        Process a floorplan image through the Tesseract++ pipeline

        Args:
            image_path: Path to the image file
            image_name: Original image filename
            timeout: Processing timeout in seconds

        Returns:
            Dictionary containing processing results
        """
        # Create a unique temp folder for this processing session
        import uuid
        session_id = str(uuid.uuid4())
        session_dir = self.temp_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Copy image to Input_Images temporarily if not already there
        input_image_path = self.input_images_dir / image_name
        image_was_copied = False

        try:
            if not input_image_path.exists():
                shutil.copy2(image_path, input_image_path)
                image_was_copied = True

            # Redirect outputs to session directory
            original_cwd = os.getcwd()
            os.chdir(self.base_path)

            # Run the main processing pipeline with timeout
            def run_processing():
                try:
                    Main.make_graph(image_name)
                except Exception as e:
                    raise e

            # Use threading for timeout control
            thread = threading.Thread(target=run_processing)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutException(f"Processing exceeded {timeout} seconds")

            # Extract results
            image_name_no_ext = Path(image_name).stem

            # Find the generated JSON files
            json_dir = self.results_dir / "Json" / image_name_no_ext
            post_pruning_json = json_dir / f"{image_name_no_ext}_post_pruning.json"
            pre_pruning_json = json_dir / f"{image_name_no_ext}_pre_pruning.json"

            # Read the post-pruning graph
            if not post_pruning_json.exists():
                raise FileNotFoundError(f"Processing completed but output not found: {post_pruning_json}")

            with open(post_pruning_json, 'r') as f:
                graph_data = json.load(f)

            # Calculate statistics
            stats = self._calculate_statistics(graph_data)

            # Add pruning comparison if pre-pruning exists
            if pre_pruning_json.exists():
                with open(pre_pruning_json, 'r') as f:
                    pre_pruning_data = json.load(f)
                    pre_nodes = len(pre_pruning_data.get("nodes", []))
                    post_nodes = len(graph_data.get("nodes", []))
                    stats["pruning_reduction"] = round((1 - post_nodes / pre_nodes) * 100, 2) if pre_nodes > 0 else 0

            return {
                "graph_json": graph_data,
                "stats": stats,
                "session_id": session_id,
                "image_name": image_name
            }

        finally:
            # Cleanup
            os.chdir(original_cwd)

            # Remove copied image if it was temporary
            if image_was_copied and input_image_path.exists():
                input_image_path.unlink()

            # Clean up session directory
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)

    def _calculate_statistics(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate graph statistics"""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        # Count nodes by type
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": node_types
        }

    def get_example_images(self) -> list:
        """Get list of available example images"""
        images = []
        for img_path in sorted(self.input_images_dir.glob("*.png"))[:4]:
            images.append({
                "name": img_path.name,
                "path": str(img_path),
                "size_kb": img_path.stat().st_size / 1024
            })
        return images