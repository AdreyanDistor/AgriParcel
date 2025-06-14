# AgriParcel
AgriParcel is a grameowrk for generating large-scale farmland vector datasets from sattelite imagery. By utilizing zero-shot segmentation from [**SAM2**](https://github.com/facebookresearch/sam2) and a fine-tuned [**CLIP**](https://github.com/openai/CLIP) model for farmland classification, we can generate ready to use vector datasets within a GeoJSON file. 

## AgriParcel Pipeline Steps
1. Segmentation ```bash (src/sam/run_sam2.py)```
2. Classification
3. Delineation
4. Simplification (optional)

## Setup
AgriParcel requires Python 3.8 at least

```bash
pip install -r requirements.txt
```
## Using AgriParcel
### Parameters
| Argument        | Type   | Default Value                              | Description                                                                                   |
|-----------------|--------|--------------------------------------------|-----------------------------------------------------------------------------------------------|
| `--input`       | str    | `data/input_data`                           | Path to the directory containing input `.tif` satellite imagery for mask generation.          |
| `--output`      | str    | `outputs/final_output`                     | File path prefix for simplified farmland polygons GeoJSON output.                             |
| `--unsimplified`| str    | `outputs/final_output_nonsimplified`      | File path prefix for raw (unsimplified) polygon GeoJSON output.                              |
| `--json`        | str    | `outputs/final_metadata`                  | File path for output `.json` metadata file listing classification confidence per mask.       |
| `--maskdir`     | str    | `outputs/image_segments`                  | Directory where intermediate RGBA `.tif` masks will be saved.                                |
| `--clip_model`  | str    | `models/clip_model.pt`                    | Path to the trained CLIP model `.pt` file for classifying masks.                             |
| `--sam_config`  | str    | `configs/sam2.1/sam2.1_hiera_l.yaml`      | Path to the YAML config file for initializing the SAM2 model.                                |
| `--sam_ckpt`    | str    | `models/sam2.1_hiera_large.pt`            | Path to the SAM2 checkpoint `.pt` file containing pretrained weights.                        |
| `--conf`        | float  | `0.5`                                      | CLIP classification confidence threshold (only masks with higher confidence are retained).    |

### Example Usage
```bash
python src/main.py \
  --input data/test_data \
  --output outputs/simplified_output \
  --unsimplified outputs/raw_output \
  --json outputs/metadata_log \
  --clip_model models/clip_model.pt \
  --sam_config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam_ckpt models/sam2.1_hiera_large.pt \
  --conf 0.75
```
# Sample Data Output
https://drive.google.com/drive/u/0/folders/1cWMypNDeFY1v_wBQ8WKSsBb96BdmDvnC

## License
This repository is released under the MIT License.

Email: adreyandistor@gmail.com
