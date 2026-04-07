# GDSR-Data-Preperation

Utilities to download, preprocess, and standardize public Guided Depth Super-Resolution (GDSR) benchmarks into a consistent NumPy format (.npy / memory-mapped), along with tools to validate and sanity-check outputs.

---

## Motivation
Reproducing GDSR results—especially on datasets like NYU Depth v2—is often unnecessarily difficult.

Many prior works:

- Reference secondary or unofficial repositories for dataset splits
- Provide incomplete or outdated instructions

- Depend on broken or inaccessible links, such as: http://gofile.me/3G5St/2lFq5R3TL

As a result, reproducing the exact train/test split used in published papers becomes unreliable and inconsistent.

## Our Solution
We provide a transparent, reproducible pipeline that:

- Reconstructs datasets using a clearly defined and consistent train–test split aligned with prior literature
- Converts raw data into efficient .npy and memory-mapped formats for scalable training
- Eliminates dependency on fragile external sources

## Citation

If you use this repository (scripts, directory layout, or processed tensors) in academic work, **please cite the companion paper** for the GDSR method or evaluation that this project supports.

**Publication (update when the camera-ready entry is fixed):**  
Tayyab Nasir *et al.*, title and venue to be added with the final PDF (see also the [project repository](https://github.com/tayyabnasir22/GDSR-Data-Preperation)).

BibTeX template—replace `title`, `booktitle` or `journal`, `year`, and `url` with the values from the published version:

```bibtex
@article{nasir2026naimasemanticsawarergb,
  title   = {NAIMA: Semantics Aware RGB Guided Depth Super-Resolution},
  author  = {Tayyab Nasir, Daochang Liu, Ajmal Mian},
  journal = {arXiv},
  year    = {2026},
  url     = {https://doi.org/10.48550/arXiv.2604.04407},
}
```

You may also cite this codebase directly:

```bibtex
@software{gdsr_data_preparation_2026,
  author = {Nasir, Tayyab},
  title  = {{GDSR-Data-Preperation}: Benchmark data preparation for guided depth super-resolution},
  year   = {2026},
  url    = {https://github.com/tayyabnasir22/GDSR-Data-Preperation}
}
```

## Requirements

- Python 3.12 recommended  
- Next, run the following to create a new environment:
```bash
python3.12 -m venv env
```
- Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

For the optional verification notebook, install Jupyter (or VS Code’s notebook support) and ensure `numpy` and `pillow` are available (they are pulled in transitively with the packages above in most environments).

---

## Layout and base path

Scripts resolve all inputs and outputs under a **base directory** (`PathManager.BASE_PATH`). Default is `./` (current working directory).

**Important:** Pass a path that ends with a path separator (e.g. `/data/gdsr/` or `./`) so that concatenated paths like `BASE + 'NYUV2/'` resolve correctly. If you omit the trailing slash, paths such as `.../NYUV2/` may break.

---

## 1. Download (`data_download.py`)

`data_download.py` uses [gdown](https://github.com/wkentaro/gdown) to fetch archives or files from Google Drive into the base directory, and optionally extracts ZIP files.

1. Open `data_download.py` and replace each placeholder `https://drive.google.com/uc?id=##########` with the real file IDs or URLs you are allowed to use.
2. **NYU Depth V2:** The script expects the **`nyu_depth_v2_labeled.mat`** file directly (`extract=False` for that entry). Do not point it at a zip unless you change the logic.
3. **RGB-D-D:** The dataset is not always public; obtain it from the authors and host or place it under your base path as required.
4. **TOF-DSR:** Use the official or author-provided Google Drive link once you have it.

Run:

```bash
# Default base path: ./
python3 data_download.py

# Explicit base path (note trailing slash)
python3 data_download.py /path/to/your/data/
```

After a successful run you should have, under the base directory, the raw assets expected by the processor (e.g. `nyu_depth_v2_labeled.mat`, `RGBDD-Full/…`, `TOFDSR/…` with `TOFDSR_Train.txt` / `TOFDSR_Test.txt`).

---

## 2. Process (`data_processor.py`)

`data_processor.py` runs all three converters in sequence:

| Benchmark | Input (under base path) | Output folder |
|-----------|-------------------------|---------------|
| **NYU V2** | `nyu_depth_v2_labeled.mat` (HDF5) | `NYUV2/` |
| **RGB-D-D** | `RGBDD-Full/` with `models/`, `plants/`, `portraits/` train/test trees containing `RGB.jpg` and `HR_gt.png` pairs | `RGBDD/` |
| **TOF-DSR** | `TOFDSR/` with split list files and images | `TOFDSRD/` |

Each split writes memory-mapped `.npy` files, including (per split prefix `train` / `test`):

- `*_images_split.npy`, `*_images_norm_split.npy`, `*_images_stand_split.npy`
- `*_depths_split.npy`, `*_depths_clipped_split.npy`, `*_depths_norm_split.npy`
- `*_mask_split.npy`, `*_minmax_split.npy`

Run:

```bash
python3 data_processor.py
python3 data_processor.py /path/to/your/data/
```

**Note:** `Utilities/DirectoryHelper.py` **deletes and recreates** each output benchmark folder before writing, so previous `.npy` outputs in those directories will be removed.

---

## 3. Test / verification scenarios (`Test.ipynb`)

The notebook performs lightweight **visual checks** that processed TOF-DSR tensors load and save as plausible RGB/depth PNGs.

1. Run processing so that `TOFDSRD/` exists under your base path (or open the notebook from the directory that already contains `TOFDSRD/`).
2. Optionally adjust paths in the first cell if your arrays are not in `./TOFDSRD/`.
3. Execute cells in order. Expected behaviors:
   - **RGB:** Load `test_images_split.npy` / `train_images_split.npy`, pick an index (e.g. 45), clip to `[0, 255]`, transpose from `CHW` to `HWC`, save `resources/sample_rgb.png` and `resources/train_sample_rgb.png`.
   - **Depth:** Load `test_depths_split.npy` / `train_depths_split.npy`, scale by `10000`, cast to `uint16`, save PNG previews under `resources/`.

If shapes print as in the checked-in outputs (e.g. test RGB `(560, 3, 384, 512)`), the pipeline is consistent with the TOF-DSR preprocessing in this repo.

---

## License

See [LICENSE](LICENSE).
