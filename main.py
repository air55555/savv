import argparse
import os
import subprocess
from pathlib import Path

from hsi_utils import (
    load_crop_save_preview,
    load_envi,
    save_rgb_preview_ppm,
    rgb_uint8_from_cube,
    overlay_grid_dots,
    save_ppm_from_rgb,
    save_png_from_rgb,
)


def open_with_default_viewer(path: Path):
    try:
        os.startfile(path)  # Windows
        return
    except AttributeError:
        pass
    try:
        subprocess.run(["xdg-open", str(path)], check=False)
        return
    except Exception:
        pass
    try:
        subprocess.run(["open", str(path)], check=False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="HSI ENVI tools: preview or crop and preview")
    parser.add_argument("--hdr", type=str, default="data/25092025_cheese_0027.hdr", help="Path to ENVI .hdr file")
    parser.add_argument("--preview-only", action="store_true", help="Only generate and display RGB preview of full image")
    parser.add_argument("--preview-out", type=str, default=None, help="Path to save preview (defaults to alongside input)")
    parser.add_argument("--png", action="store_true", help="Save preview as PNG in addition to or instead of PPM")
    parser.add_argument("--grid", action="store_true", help="Overlay 100px grid of dots on preview")
    parser.add_argument("--numbers", action="store_true", help="Overlay index numbers at 100px grid points")
    parser.add_argument("--highlight", action="store_true", help="Highlight high-intensity regions")
    parser.add_argument("--highlight-p", type=float, default=99.0, help="Percentile for highlight threshold (e.g., 99.0)")
    parser.add_argument("--x", type=int, default=0, help="Crop x (column)")
    parser.add_argument("--y", type=int, default=0, help="Crop y (row)")
    parser.add_argument("--w", "--width", dest="width", type=int, default=256, help="Crop width")
    parser.add_argument("--h", "--height", dest="height", type=int, default=256, help="Crop height")
    parser.add_argument("--out", type=str, default="data/crop_example", help="Output base path (without extension)")
    parser.add_argument("--interleave", type=str, default="bsq", choices=["bsq", "bil", "bip"], help="Output interleave")
    args = parser.parse_args()

    if args.preview_only:
        cube, header, _ = load_envi(args.hdr)
        # Build RGB, optional overlays, then save as requested
        rgb = rgb_uint8_from_cube(cube, header=header)
        out_base = Path(args.preview_out) if args.preview_out else Path(args.hdr).with_suffix('')
        saved_paths = []
        # Primary output (respect --png vs PPM)
        rgb_main = rgb
        if args.grid:
            rgb_main = overlay_grid_dots(rgb_main, step=100, color=(255, 0, 0), dot_size=1)
        if args.numbers:
            from hsi_utils import overlay_grid_numbers  # local import to avoid unused when not needed
            rgb_main = overlay_grid_numbers(rgb_main, step=100, color=(255, 255, 0), offset=(2, 2), mode='xy')
        if args.png:
            png_path = out_base.with_suffix('.png')
            saved_paths.append(save_png_from_rgb(rgb_main, png_path))
        else:
            ppm_path = out_base.with_suffix('.ppm')
            saved_paths.append(save_ppm_from_rgb(rgb_main, ppm_path))

        # Special grid outputs: always save an extra with `_grid` suffix
        rgb_grid = overlay_grid_dots(rgb, step=100, color=(255, 0, 0), dot_size=1)
        if args.numbers:
            from hsi_utils import overlay_grid_numbers
            rgb_grid = overlay_grid_numbers(rgb_grid, step=100, color=(255, 255, 0), offset=(2, 2), mode='xy')
        if args.highlight:
            from hsi_utils import overlay_high_intensity
            # For preview-only, compute highlight from the full cube
            rgb_grid = overlay_high_intensity(rgb_grid, cube=cube, header=header, percentile=args.highlight_p)
        grid_saved = []
        try:
            grid_saved.append(save_png_from_rgb(rgb_grid, out_base.with_name(out_base.name + '_grid').with_suffix('.png')))
        except Exception:
            # PNG optional (requires Pillow); continue even if unavailable
            pass
        grid_saved.append(save_ppm_from_rgb(rgb_grid, out_base.with_name(out_base.name + '_grid').with_suffix('.ppm')))
        saved_paths.extend(grid_saved)

        for p in saved_paths:
            print(f"Opening preview: {p}")
            open_with_default_viewer(Path(p))
        return

    hdr_out, raw_out, preview = load_crop_save_preview(
        hdr_path=args.hdr,
        x=args.x,
        y=args.y,
        width=args.width,
        height=args.height,
        out_base_path=args.out,
        rgb_preview_path=None,
        band_range=None,
        interleave=args.interleave,
    )

    print(f"Cropped ENVI saved:\n  {hdr_out}\n  {raw_out}")
    # Regenerate preview(s) with requested options
    cube, header, _ = load_envi(str(hdr_out))
    rgb = rgb_uint8_from_cube(cube, header=header)
    out_base = Path(args.out)
    paths = []
    # Primary output (respect --png vs PPM); grid if requested
    rgb_main = rgb
    if args.grid:
        rgb_main = overlay_grid_dots(rgb_main, step=100, color=(255, 0, 0), dot_size=1)
    if args.numbers:
        from hsi_utils import overlay_grid_numbers
        rgb_main = overlay_grid_numbers(rgb_main, step=100, color=(255, 255, 0), offset=(2, 2), mode='xy')
    if args.highlight:
        from hsi_utils import overlay_high_intensity
        rgb_main = overlay_high_intensity(rgb_main, cube=cube, header=header, percentile=args.highlight_p)
    if args.png:
        paths.append(save_png_from_rgb(rgb_main, out_base.with_suffix('.png')))
    else:
        paths.append(save_ppm_from_rgb(rgb_main, out_base.with_suffix('.ppm')))

    # Special grid outputs: always save an extra with `_grid` suffix
    rgb_grid = overlay_grid_dots(rgb, step=100, color=(255, 0, 0), dot_size=1)
    if args.numbers:
        from hsi_utils import overlay_grid_numbers
        rgb_grid = overlay_grid_numbers(rgb_grid, step=100, color=(255, 255, 0), offset=(2, 2), mode='xy')
    if args.highlight:
        from hsi_utils import overlay_high_intensity
        rgb_grid = overlay_high_intensity(rgb_grid, cube=cube, header=header, percentile=args.highlight_p)
    try:
        paths.append(save_png_from_rgb(rgb_grid, out_base.with_name(out_base.name + '_grid').with_suffix('.png')))
    except Exception:
        pass
    paths.append(save_ppm_from_rgb(rgb_grid, out_base.with_name(out_base.name + '_grid').with_suffix('.ppm')))

    for p in paths:
        print(f"Opening preview: {p}")
        open_with_default_viewer(Path(p))


if __name__ == "__main__":
    main()
