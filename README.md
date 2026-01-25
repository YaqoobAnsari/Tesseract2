---
title: Tesseract++
emoji: 🏗️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Floorplan to navigable graph conversion
---

## Tesseract++: Floorplan Parsing to Navigable Graphs

This project converts annotated building floorplans into navigable graphs by combining text detection, semantic interpretation, room segmentation, and door detection. The pipeline produces both visual overlays and structured JSON suitable for routing, accessibility analysis, and downstream spatial reasoning.

### Key Capabilities
- **Text detection & interpretation**: CRAFT-based text detection (`Models/Text_Models`) followed by semantic parsing (`Models/Interpreter`) to label rooms, corridors, outdoors, and transition elements (stairs/elevators).
- **Graph construction**: Builds a multi-type `BuildingGraph` (`utils/graph.py`) with typed nodes (rooms, corridors, outside, doors, transitions) and geometric attributes.
- **Spatial refinement**: Flood/smart fill segmentation, subnode proposal, corridor/outdoor expansion, and edge creation/pruning for a well-conditioned graph.
- **Door pipeline**: Faster R-CNN–based door detection/refinement/classification (`Models/Door_Models`), with intelligent room-to-door connectivity:
  - **Type-aware door connections**: Different strategies for room-to-corridor (r2c), room-to-room (r2r), and exit doors
  - **Room family funneling**: Optimal pathfinding from subnodes through main room to doors
  - **Guaranteed connectivity**: Ensures all room subnodes can reach corridor doors via main room
- **Transition handling**: Automatic connection of stairs/elevators to corridor networks for multi-floor navigation support
- **Multi-floor connectivity**: Comprehensive support for processing multiple floorplan images and connecting them via transition nodes:
  - **Dynamic floor detection**: Automatically extracts floor numbers from image filenames (FF→1, SF→2, TF→3, numeric→N)
  - **Graph merging**: Combines individual floor graphs into a unified multi-floor graph with prefixed node IDs
  - **Inter-floor transition mapping**: Manual mapping system with comprehensive validation:
    - Image existence checks (all images must exist in `Input_Images/`)
    - Node type validation (only transition nodes can connect across floors)
    - One-to-one constraint enforcement (one transition per floor pair)
    - Floor order validation (warns on non-adjacent floor connections)
    - Spatial alignment verification (ensures vertical alignment of transitions)
  - **Auto-processing**: Automatically processes missing floor graphs before merging
  - **Robust error handling**: Detailed error messages and validation at every step
- **Outputs**: JSON graph exports and multiple plot variants (initial, thresholded, pre/post-pruning, blank overlays), plus timing/metadata summaries.

### Repository Layout
- `Main.py` — orchestrates the single-floor pipeline end-to-end.
- `MultiFloor.py` — standalone module for multi-floor connectivity processing.
- `app.py` — FastAPI web application entry point.
- `mappings/` — contains transition mapping files for multi-floor connections.
- `Models/Text_Models/` — text detection (CRAFT) and helpers.
- `Models/Interpreter/` — text interpretation and label parsing.
- `Models/Door_Models/` — door detection/classification models.
- `utils/` — graph utilities, connectivity, flood fill, timing analysis.
- `utils/app_utils/` — web application backend (API, visualization, graph conversion).
- `utils/app_utils/frontend/` — React + TypeScript frontend (Vite build).
- `Input_Images/` — sample/input floorplan images (included).
- `Results/` — single-floor generated plots, JSONs, and timing reports.
- `Multifloor_Results/` — multi-floor outputs (Jsons, Plots, Time&Meta per floor sequence).
- `Model_weights/` — **not tracked**; place model checkpoints here (`*.pth`, `*.ckpt`).
- `Dockerfile` / `docker-compose.yml` — container support for deployment and Hugging Face Spaces.

### Environment & Dependencies
Tested with Python 3.12 and CPU PyTorch 2.9.1. Minimal setup:
```bash
python -m venv tess
source tess/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Model Weights
Weights are not committed. Place required checkpoints under `Model_weights/`, e.g.:
- `Model_weights/craft_mlt_25k.pth` (CRAFT text detector)
- Door detector weights as expected by `Models/Door_Models/door_bboxer.py`
- Interpreter/text recognition weights as expected by `Models/Interpreter/text_interpreter.py`

### Running the Pipeline

#### Single Floor Processing
From the repo root:
```bash
source tess/bin/activate  # or your environment
python -u Main.py "FF part 1upE.png"
```
The script expects the image name to exist under `Input_Images/`. Outputs are written under `Results/`:
- `Results/Plots/...` — overlays for detection, graphs, doors, fills.
- `Results/Json/...` — graph JSONs (initial, pre-pruning, post-pruning).
- `Results/Time&Meta/...` — timing logs and correlation plots.

#### Multi-Floor Processing
Multi-floor connectivity is handled by the standalone `MultiFloor.py` module.

**Step 1: Prepare your images**

Place all floorplan images in `Input_Images/` with floor indicators in filenames:
- `FF part 1upE.png` → Floor 1 (First Floor)
- `SF part 1upE.png` → Floor 2 (Second Floor)
- `TF part 1upE.png` → Floor 3 (Third Floor)
- `B1 part 1upE.png` → Floor -1 (Basement 1)
- `4 part 1upE.png` → Floor 4 (numeric floors)

**Step 2: Create a mapping file**

Create a `.txt` file in the `mappings/` folder with transition connections:

```
# Mapping format: (floor_num, image_name, node_id):(floor_num, image_name, node_id)

# Connect stairs_1 from Floor 1 to Floor 2
(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)

# Connect elevator_1 from Floor 1 to Floor 2
(1, FF part 1upE.png, elevator_1):(2, SF part 1upE.png, elevator_1)
```

**Step 3: Run multi-floor processing**

```bash
source tess/bin/activate
python MultiFloor.py --mapping-file mappings/FF_SF.txt
```

Or use inline mapping:
```bash
python MultiFloor.py --mapping "(1, FF part 1upE.png, stairs_1):(2, SF part 1upE.png, stairs_1)"
```

**Output Structure:**
- `Multifloor_Results/Jsons/FF_SF/` — merged graph and floor backups
- `Multifloor_Results/Plots/FF_SF/` — visualization plots
- `Multifloor_Results/Time&Meta/FF_SF/` — timing and validation reports

**Validation Features:**
- ✅ Ensures all images exist before processing
- ✅ Validates that only transition nodes (stairs/elevators) are mapped
- ✅ Enforces one-to-one constraint (one transition per floor pair)
- ✅ **Enforces floor adjacency** (Floor N can only connect to N±1) — ERROR on violation
- ✅ Supports negative floors (basements: B1=-1, B2=-2)
- ✅ Auto-generates missing floor graphs before merging
- ✅ Provides detailed error messages for debugging

### Web Application

Tesseract++ includes an interactive web interface for uploading floorplans and exploring the generated graphs visually.

**Stack:** FastAPI backend + React/TypeScript frontend with Cytoscape.js for interactive graph rendering.

**Features:**
- Drag-and-drop image upload (PNG, max 10 MB)
- 4 pre-cached example floorplans for instant demos
- Interactive graph canvas with pan, zoom, and node hover tooltips
- Node type filtering (room, door, corridor, outside, floor transition)
- Toggleable floorplan background overlay behind the graph
- Statistics panel (node/edge counts, type breakdown, pruning reduction)
- Export to PNG (high-res) or JSON

#### Running the Web App

**Development (with hot reload):**
```bash
# Terminal 1: Backend
pip install -r requirements.txt
python app.py

# Terminal 2: Frontend
cd utils/app_utils/frontend
npm install
npm run dev
```
Frontend dev server runs at `http://localhost:3000` and proxies API calls to the backend at `http://localhost:8000`.

**Production (single server):**
```bash
cd utils/app_utils/frontend && npm install && npm run build && cd ../../..
python app.py
```
Open `http://localhost:8000` — FastAPI serves the built frontend and API from the same port.

#### Docker

```bash
docker-compose up --build
```
Requires `Model_weights/` and `Input_Images/` to be present locally (mounted as volumes). The app runs at `http://localhost:8000`.

#### Hugging Face Spaces

The Dockerfile supports deployment as a Docker Space on Hugging Face. Set the Space SDK to **Docker** and ensure model weights and input images are included in the repository or configured as persistent storage.

### Recent Improvements
- **Enhanced door connectivity**: Type-aware door-to-room edge creation ensures all door types (r2c, r2r, exit) are properly connected while maintaining optimal pathfinding structure
- **Transition node integration**: Stairs and elevators are now automatically connected to corridor networks, enabling multi-floor pathfinding
- **Robust room family funneling**: Improved algorithm guarantees connectivity for all room subnodes while preserving optimal door placement
- **Multi-floor module (`MultiFloor.py`)**: Complete standalone module for multi-floor connectivity with:
  - Text-based mapping file format
  - Comprehensive validation (floor adjacency, one-to-one constraints)
  - Auto-generation of missing floor graphs
  - Organized output structure under `Multifloor_Results/`
  - Visualization plots for merged graphs and inter-floor connections
- **OCR text in results**: Text detection results now include inferred text alongside bounding box coordinates
- **Improved flood fill robustness**: Enhanced seed finding algorithm prevents text annotation interference

### Notes and Good Practices
- Keep weights out of version control; `.gitignore` already excludes `Model_weights/` and `*.pth/*.ckpt`.
- If running on GPU, PyTorch will automatically use CUDA when available; otherwise CPU is used.
- Large artifacts (images/results) are included for reproducibility; consider Git LFS if you plan to grow the dataset.

### Citation / Academic Use
If you build upon this code for publications, please cite the CRAFT text detector (Baek et al., 2019) and Faster R-CNN (Ren et al., 2015) as appropriate, alongside your own work. No formal project-specific citation is provided here.

