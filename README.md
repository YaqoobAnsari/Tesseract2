## Tesseract++: Floorplan Parsing to Navigable Graphs

This project converts annotated building floorplans into navigable graphs by combining text detection, semantic interpretation, room segmentation, and door detection. The pipeline produces both visual overlays and structured JSON suitable for routing, accessibility analysis, and downstream spatial reasoning.

### Key Capabilities
- **Text detection & interpretation**: CRAFT-based text detection (`Models/Text_Models`) followed by semantic parsing (`Models/Interpreter`) to label rooms, corridors, outdoors, and transition elements (stairs/elevators).
- **Graph construction**: Builds a multi-type `BuildingGraph` (`utils/graph.py`) with typed nodes (rooms, corridors, outside, doors, transitions) and geometric attributes.
- **Spatial refinement**: Flood/smart fill segmentation, subnode proposal, corridor/outdoor expansion, and edge creation/pruning for a well-conditioned graph.
- **Door pipeline**: Faster R-CNN–based door detection/refinement/classification (`Models/Door_Models`), with room-to-door connectivity and corridor/outdoor integration.
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
From the repo root:
```bash
source tess/bin/activate  # or your environment
python -u Main.py "FF part 1upE.png"
```
The script expects the image name to exist under `Input_Images/`. Outputs are written under `Results/`:
- `Results/Plots/...` — overlays for detection, graphs, doors, fills.
- `Results/Json/...` — graph JSONs (initial, pre-pruning, post-pruning).
- `Results/Time&Meta/...` — timing logs and correlation plots.

### Notes and Good Practices
- Keep weights out of version control; `.gitignore` already excludes `Model_weights/` and `*.pth/*.ckpt`.
- If running on GPU, PyTorch will automatically use CUDA when available; otherwise CPU is used.
- Large artifacts (images/results) are included for reproducibility; consider Git LFS if you plan to grow the dataset.

### Citation / Academic Use
If you build upon this code for publications, please cite the CRAFT text detector (Baek et al., 2019) and Faster R-CNN (Ren et al., 2015) as appropriate, alongside your own work. No formal project-specific citation is provided here.

