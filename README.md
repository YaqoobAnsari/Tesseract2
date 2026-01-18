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
- `Main.py` — orchestrates the full pipeline end-to-end.
- `Models/Text_Models/` — text detection (CRAFT) and helpers.
- `Models/Interpreter/` — text interpretation and label parsing.
- `Models/Door_Models/` — door detection/classification models.
- `utils/` — graph utilities, connectivity, flood fill, timing analysis.
- `Input_Images/` — sample/input floorplan images (included).
- `Results/` — generated plots, JSONs, and timing reports (included).
- `Model_weights/` — **not tracked**; place model checkpoints here (`*.pth`, `*.ckpt`).

### Environment & Dependencies
Tested with Python 3.12 and CPU PyTorch 2.9.1. Minimal setup:
```bash
python -m venv tess
source tess/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless numpy pandas pillow matplotlib networkx lmdb natsort six scikit-image scipy tqdm fuzzywuzzy python-Levenshtein
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
Multi-floor connectivity is implemented and ready for use. To process multiple floors:

1. **Prepare your images**: Place all floorplan images in `Input_Images/` with floor indicators in filenames:
   - `FF part 1upE.png` → Floor 1 (First Floor)
   - `SF part 1upE.png` → Floor 2 (Second Floor)
   - `TF part 1upE.png` → Floor 3 (Third Floor)
   - `4 part 1upE.png` → Floor 4 (numeric floors)

2. **Define transition mapping**: Create a mapping dictionary specifying which transition nodes connect across floors:
   ```python
   transition_mapping = {
       (1, "FF part 1upE.png", "stairs_1"): [
           (2, "SF part 1upE.png", "stairs_1"),
           (3, "TF part 1upE.png", "stairs_1")
       ]
   }
   ```

3. **Uncomment and run**: In `Main.py`, uncomment the multi-floor example usage block and provide your mapping. The system will:
   - Validate the mapping (image existence, node types, floor constraints)
   - Auto-process any missing floor graphs
   - Merge all floor graphs into a unified graph
   - Connect transitions across floors with spatial alignment checks
   - Save the merged graph to `Results/Json/MULTI_FLOOR/merged_multi_floor_graph.json`

**Validation Features**:
- ✅ Ensures all images exist before processing
- ✅ Validates that only transition nodes (stairs/elevators) are mapped
- ✅ Enforces one-to-one constraint (one transition per floor pair)
- ✅ Checks floor order (warns on non-adjacent connections)
- ✅ Verifies spatial alignment of transitions across floors
- ✅ Provides detailed error messages for debugging

### Recent Improvements
- **Enhanced door connectivity**: Type-aware door-to-room edge creation ensures all door types (r2c, r2r, exit) are properly connected while maintaining optimal pathfinding structure
- **Transition node integration**: Stairs and elevators are now automatically connected to corridor networks, enabling multi-floor pathfinding
- **Robust room family funneling**: Improved algorithm guarantees connectivity for all room subnodes while preserving optimal door placement
- **Multi-floor support**: Complete implementation of multi-floor graph processing with validation, auto-processing, and inter-floor transition connectivity (ready for use, currently commented out in `Main.py` pending multi-floor test images)
- **Improved flood fill robustness**: Enhanced seed finding algorithm prevents text annotation interference, ensuring accurate room area calculations

### Notes and Good Practices
- Keep weights out of version control; `.gitignore` already excludes `Model_weights/` and `*.pth/*.ckpt`.
- If running on GPU, PyTorch will automatically use CUDA when available; otherwise CPU is used.
- Large artifacts (images/results) are included for reproducibility; consider Git LFS if you plan to grow the dataset.

### Citation / Academic Use
If you build upon this code for publications, please cite the CRAFT text detector (Baek et al., 2019) and Faster R-CNN (Ren et al., 2015) as appropriate, alongside your own work. No formal project-specific citation is provided here.

