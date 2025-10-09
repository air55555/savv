## HSI ENVI tools

Command-line utilities and helpers for working with ENVI-format hyperspectral cubes: load, crop, and generate RGB previews (PPM/PNG) with optional overlays.

### Installation

- Python 3.8+
- Required: `numpy`
- Optional (for PNG output): `pillow`

Install with:

```bash
pip install -r requirements.txt
```

If you only need PPM output, Pillow is not required.

### How RGB previews are generated

The RGB preview is derived from the hyperspectral cube as follows:

- **Band selection**:
  - If the ENVI header contains `wavelengths` with the same length as the number of bands, the code picks the bands closest to approximately 650 nm (R), 550 nm (G), and 450 nm (B).
  - If wavelengths are not available or mismatched, it falls back to a default triplet of band indices.

- **Per-channel contrast stretch to 8-bit**:
  - Each selected band is independently converted to `uint8` by linear scaling between the 2nd and 98th percentiles of that band.
  - Values below the 2nd percentile map to 0; values above the 98th percentile map to 255.
  - The three channels are then stacked to form an RGB image.

This produces a visually balanced RGB rendering that is robust to outliers while preserving contrast in the bulk of the data.

Optionally, you can add:
- **Grid overlay**: red dots every 100 px.
- **Grid numbers**: yellow coordinate labels near grid points.
- **High-intensity highlight**: cyan overlay over pixels above a chosen percentile (computed from the preview image or full cube depending on mode).

### Usage

The `main.py` script provides two primary modes: preview-only and crop+preview.

```bash
# Preview-only (renders an RGB preview next to the input file)
python main.py --hdr data/25092025_cheese_0027.hdr --preview-only --png

# With overlays
python main.py --hdr data/25092025_cheese_0027.hdr --preview-only --png --grid --numbers --highlight --highlight-p 99.0

# Crop a region and save cropped ENVI + preview
python main.py --hdr data/25092025_cheese_0027.hdr --x 0 --y 0 --width 256 --height 256 --out data/crop_example --png
```

Arguments (selected):
- `--hdr`: Path to the ENVI `.hdr` file.
- `--preview-only`: Generate preview(s) without writing a cropped ENVI dataset.
- `--preview-out`: Custom output path for preview files (defaults next to input).
- `--png`: Write PNG (requires Pillow). If omitted, PPM is written.
- `--grid`: Overlay a red dot grid every 100 px.
- `--numbers`: Label grid points with coordinates.
- `--highlight`: Highlight high-intensity regions; control threshold with `--highlight-p` (default 99.0).
- `--x --y --width --height`: Crop rectangle (for crop mode).
- `--out`: Output base path (no extension) for cropped ENVI and preview.
- `--interleave`: ENVI interleave for output (`bsq`, `bil`, or `bip`).

### Spectra over highlighted regions

You can compute average spectra over regions highlighted by a percentile threshold (same logic as the `--highlight` preview) or by reusing a saved highlighted PNG as a mask.

```bash
# Recompute mask from cube using a percentile (recommended)
python spectra.py --hdr data/25092025_cheese_0027.hdr --percentile 95.5 --out-dir data --stem crop95

# Use an existing highlighted PNG as mask (must match spatial size)
python spectra.py --hdr data/25092025_cheese_0027.hdr --highlight-png data/crop95.png --out-dir data --stem crop95
```

Outputs (in `--out-dir`):
- `{stem}_spectra_all.png`: combined plot of all aggregations
- `{stem}_spectra_<name>.png`: individual plots per aggregation
- `{stem}_spectra.csv`: tabular spectra (first column is wavelength (nm) if available, otherwise band index)

Aggregation methods (10): mean, median, trimmed_mean_10, winsorized_mean_10, geometric_mean, harmonic_mean, p25, p75, max, min.

### Notes on Windows and open files

On Windows, attempting to overwrite a `.raw` file that is currently open via a memory map can fail. The helper functions avoid writing to the exact same base name as the input; if a conflict is detected, `_crop` is appended to the output base automatically.

### Library API overview

From `hsi_utils.py`:
- `load_envi(path) -> (cube, header, raw_path)`: Loads an ENVI dataset into a (lines, samples, bands) array.
- `crop_cube(cube, x, y, width, height, band_range=None)`: Returns a cropped subcube.
- `rgb_uint8_from_cube(cube, header=None, rgb_bands=None) -> np.ndarray`: Builds an 8-bit RGB image using the band selection and contrast stretch described above.
- `save_ppm_from_rgb(rgb, out_path)` and `save_png_from_rgb(rgb, out_path)`: Save previews.
- `load_crop_save_preview(...)`: Convenience to load, crop, save ENVI, and write a preview.


