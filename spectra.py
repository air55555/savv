import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from hsi_utils import load_envi, save_envi


def build_mask_from_cube(cube: np.ndarray, percentile: float) -> np.ndarray:
    inten = np.mean(cube.astype(np.float32), axis=2)
    thresh = float(np.percentile(inten, percentile))
    return inten >= thresh


def build_mask_from_highlight_png(png_path: Path) -> np.ndarray:
    # Heuristic: highlighted overlay used cyan (0,255,255) alpha blended over base.
    # Detect pixels with strong G and B, weaker R relative to G/B.
    from PIL import Image  # pillow is already a dependency
    img = Image.open(png_path).convert('RGB')
    arr = np.asarray(img, dtype=np.uint8)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    # Simple rules to isolate cyan-ish overlays
    mask = (g > 150) & (b > 150) & (r < (g + b) // 4)
    return mask


def aggregate_spectra(cube: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    masked = cube[mask]
    if masked.size == 0:
        raise ValueError('Mask selects zero pixels; cannot compute spectra.')
    # masked shape: (N, bands)
    bands = masked.shape[-1]
    out: Dict[str, np.ndarray] = {}

    # 1. Mean
    out['mean'] = masked.mean(axis=0)
    # 2. Median
    out['median'] = np.median(masked, axis=0)
    # 3. Trimmed mean (10%)
    k = max(1, int(0.1 * masked.shape[0]))
    sorted_vals = np.sort(masked, axis=0)
    trimmed = sorted_vals[k:-k] if sorted_vals.shape[0] > 2 * k else sorted_vals
    out['trimmed_mean_10'] = trimmed.mean(axis=0)
    # 4. Winsorized mean (10%)
    lo = sorted_vals[k] if sorted_vals.shape[0] > k else sorted_vals[0]
    hi = sorted_vals[-k - 1] if sorted_vals.shape[0] > k else sorted_vals[-1]
    wins = np.clip(masked, lo, hi)
    out['winsorized_mean_10'] = wins.mean(axis=0)
    # 5. Geometric mean (add small epsilon to avoid zeros)
    eps = 1e-6
    out['geometric_mean'] = np.exp(np.mean(np.log(masked + eps), axis=0))
    # 6. Harmonic mean
    out['harmonic_mean'] = 1.0 / (np.mean(1.0 / (masked + eps), axis=0))
    # 7. 25th percentile
    out['p25'] = np.percentile(masked, 25, axis=0)
    # 8. 75th percentile
    out['p75'] = np.percentile(masked, 75, axis=0)
    # 9. Max
    out['max'] = masked.max(axis=0)
    # 10. Min
    out['min'] = masked.min(axis=0)

    return out


def save_spectra_plots(spectra: Dict[str, np.ndarray], wavelengths: Optional[List[float]], out_dir: Path, stem: str) -> List[Path]:
    import matplotlib.pyplot as plt
    out_paths: List[Path] = []
    x = np.arange(len(next(iter(spectra.values())))) if not wavelengths else np.asarray(wavelengths)

    # Combined plot
    plt.figure(figsize=(10, 6))
    for name, spec in spectra.items():
        plt.plot(x, spec, label=name)
    plt.xlabel('Wavelength (nm)' if wavelengths else 'Band index')
    plt.ylabel('Intensity')
    plt.title('Masked region spectra (multiple aggregations)')
    plt.legend(ncol=2, fontsize=8)
    combined = out_dir / f"{stem}_spectra_all.png"
    plt.tight_layout()
    plt.savefig(combined, dpi=150)
    plt.close()
    out_paths.append(combined)

    # Individual plots
    for name, spec in spectra.items():
        plt.figure(figsize=(8, 4))
        plt.plot(x, spec, label=name, color='tab:blue')
        plt.xlabel('Wavelength (nm)' if wavelengths else 'Band index')
        plt.ylabel('Intensity')
        plt.title(name)
        plt.tight_layout()
        p = out_dir / f"{stem}_spectra_{name}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        out_paths.append(p)
    return out_paths


def save_spectra_csv(spectra: Dict[str, np.ndarray], wavelengths: Optional[List[float]], out_dir: Path, stem: str) -> Path:
    # Columns: wavelength_or_band, then each aggregation
    names = list(spectra.keys())
    n = len(next(iter(spectra.values())))
    x = np.arange(n) if wavelengths is None else np.asarray(wavelengths)
    arr = np.zeros((n, 1 + len(names)), dtype=float)
    arr[:, 0] = x
    for i, name in enumerate(names, start=1):
        arr[:, i] = spectra[name]
    header_cols = ['wavelength_nm' if wavelengths is not None else 'band'] + names
    out_path = out_dir / f"{stem}_spectra.csv"
    np.savetxt(out_path, arr, delimiter=',', header=','.join(header_cols), comments='')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Compute spectra over highlighted region.')
    parser.add_argument('--hdr', type=str, required=True, help='Path to ENVI .hdr file')
    parser.add_argument('--highlight-png', type=str, default=None, help='Optional path to highlighted PNG to derive mask')
    parser.add_argument('--percentile', type=float, default=95.5, help='Percentile for highlight mask if recomputing from cube')
    parser.add_argument('--out-dir', type=str, default='data', help='Directory to save spectra outputs')
    parser.add_argument('--stem', type=str, default='crop', help='Output file name stem prefix')
    parser.add_argument('--subtract-csv', type=str, default=None, help='Path to spectra CSV to subtract and save residual cubes')
    args = parser.parse_args()

    cube, header, _ = load_envi(args.hdr)
    if args.highlight_png:
        mask = build_mask_from_highlight_png(Path(args.highlight_png))
        # Ensure mask size matches cube spatial dims
        if mask.shape != cube.shape[:2]:
            raise ValueError(f'Mask size {mask.shape} does not match cube size {cube.shape[:2]}')
    else:
        mask = build_mask_from_cube(cube, percentile=args.percentile)

    # Print simple statistics about mask coverage
    total_pixels = int(mask.shape[0] * mask.shape[1])
    masked_pixels = int(mask.sum())
    coverage_pct = (masked_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    print(f"Mask stats: {masked_pixels} masked / {total_pixels} total ({coverage_pct:.3f}%)")

    spectra = aggregate_spectra(cube, mask)
    wavelengths = header.get('wavelengths') if isinstance(header.get('wavelengths'), list) else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = save_spectra_plots(spectra, wavelengths, out_dir, args.stem)
    csv_path = save_spectra_csv(spectra, wavelengths, out_dir, args.stem)

    print('Saved:')
    for p in plots:
        print(f'  {p}')
    print(f'  {csv_path}')

    # Optional: subtract provided CSV spectra from the cube and save residuals
    if args.subtract_csv:
        csv_file = Path(args.subtract_csv)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        # Load CSV; first row is header
        with csv_file.open('r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        names = header[1:]  # skip wavelength/band column
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        # Handle case where CSV might be a single row
        if data.ndim == 1:
            data = data[None, :]
        spectra_mat = data[:, 1:]  # drop first column (x)
        # Validate dimensions
        if spectra_mat.shape[0] != len(np.unique(np.arange(spectra_mat.shape[0]))):
            pass  # no-op, just clarity
        if spectra_mat.shape[0] != data.shape[0]:
            raise ValueError('Unexpected CSV shape after parsing.')
        # Transpose if needed to get shape (bands,) per method
        # Here each column corresponds to a method; rows correspond to x-axis values
        # So for a given method i, spectra_mat[:, i] is length==bands
        bands = cube.shape[2]
        if spectra_mat.shape[0] != bands:
            raise ValueError(f'CSV spectra length {spectra_mat.shape[0]} does not match cube bands {bands}')

        # Map names to safe suffixes
        def safe_suffix(name: str) -> str:
            return name.lower().replace(' ', '_')

        # Subtract and save residual cubes
        for i, name in enumerate(names):
            spec = spectra_mat[:, i].astype(cube.dtype, copy=False)
            # Broadcast subtract over H and W: (H,W,B) - (B,) -> (H,W,B)
            residual = (cube.astype(np.float32) / spec.astype(np.float32) )*100
            out_base = out_dir / f"{args.stem}_residual_{safe_suffix(name)}"
            save_envi(residual.astype(cube.dtype, copy=False), out_base_path=str(out_base), template_header={'wavelengths': wavelengths} if wavelengths else None)
            # Ensure ENVI header contains a default RGB triplet for viewers
            hdr_path = out_base.with_suffix('.hdr')
            try:
                txt = hdr_path.read_text(encoding='utf-8', errors='ignore')
                if 'default bands' not in txt.lower():
                    with hdr_path.open('a', encoding='utf-8') as f:
                        f.write('default bands = {165, 104, 44}\n')
            except OSError:
                # If header is not accessible, skip silently
                pass
            print(f"Residual saved: {out_base.with_suffix('.hdr')} / {out_base.with_suffix('.raw')}")


if __name__ == '__main__':
    main()


