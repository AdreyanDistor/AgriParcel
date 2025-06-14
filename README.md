# AgriParcel

AgriParcel is a framework for generating farmland polygons from satellite imagery. It combines zero-shot segmentation from **SAM2** with a fine-tuned **CLIP** classifier to identify agricultural parcels and produces ready-to-use GeoJSON files.

The pipeline works as follows:

1. **Segmentation** – RGB satellite tiles are processed by SAM2 to create binary masks for every visible object.
2. **Classification** – Each segmented image is classified as `farmland` or `non-farmland` using a CLIP model trained on farmland and park imagery.
3. **Delineation** – Masks identified as farmland are converted to geometric polygons.
4. **Simplification** – Polygons are optionally simplified to reduce the number of vertices.

The resulting GeoJSON files can be used for crop analysis, field boundary mapping and other agricultural applications.

## Setup

AgriParcel requires Python 3.8+ with the following packages:

- `torch`
- `clip`
- `geopandas`
- `rasterio`
- `shapely`
- `numpy`
- `opencv-python`
- `scikit-learn`
- `pandas`

Install the dependencies with pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install clip geopandas rasterio shapely numpy opencv-python scikit-learn pandas tqdm matplotlib seaborn
```

## Running the Pipeline

1. **Generate masks**
   ```bash
   python src/sam2/run_sam2.py
   ```
   The script expects a directory of `.tif` images and writes segmented masks to the output directory.

2. **Train or run the CLIP classifier**
   ```bash
   python src/clip/train.py        # for training
   python src/clip/test.py         # evaluate a saved model
   ```
   Training data should be arranged by class within a root folder.

3. **Delineate farmland polygons**
   ```bash
   python src/delineation/object_delineation.py
   ```
   This script reads mask images produced by SAM2 and outputs GeoJSON polygons.

4. **Simplify polygons (optional)**
   ```bash
   python src/simplify/simplify_eval.py
   ```
   Simplification reduces vertex count using the Douglas–Peucker algorithm.

Each stage can be run independently depending on your workflow.

## License

This repository is released under the MIT License.
