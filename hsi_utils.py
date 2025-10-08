from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image

import numpy as np


# --- ENVI header parsing ---

_ENVI_DTYPE_MAP: Dict[int, np.dtype] = {
    1: np.dtype('uint8'),
    2: np.dtype('int16'),
    3: np.dtype('int32'),
    4: np.dtype('float32'),
    5: np.dtype('float64'),
    12: np.dtype('uint16'),
    13: np.dtype('uint32'),
    14: np.dtype('int64'),
    15: np.dtype('uint64'),
}

_ENVI_BYTE_ORDER_MAP: Dict[int, str] = {
    0: '<',  # little-endian
    1: '>',  # big-endian
}


def _parse_envi_header(hdr_path: Path) -> Dict[str, object]:
    """Parse a minimal ENVI .hdr file into a dict.

    Supported fields: samples, lines, bands, data type, interleave, byte order,
    header offset, wavelengths (optional).
    """
    text = hdr_path.read_text(encoding='utf-8', errors='ignore')
    header: Dict[str, object] = {}

    # Handle multi-line braces blocks like wavelengths = {a, b, c}
    def _collect_block(name: str, start_idx: int) -> Tuple[str, int]:
        i = text.find('{', start_idx)
        if i == -1:
            return '', start_idx
        depth = 1
        j = i + 1
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        return text[i + 1:j - 1], j

    # Normalize to simple key = value lines where possible
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().lower().startswith('envi')]
    i = 0
    while i < len(lines):
        line = lines[i]
        if '=' not in line:
            i += 1
            continue
        key, value = [s.strip() for s in line.split('=', 1)]
        key_l = key.lower()
        if value.endswith('{') or value.endswith('{\n'):
            # Reconstruct block from original text to capture across newlines
            # Find this line occurrence in the raw text
            start = text.lower().find(f"{key_l}")
            if start == -1:
                i += 1
                continue
            block, _ = _collect_block(key_l, start)
            if key_l == 'wavelengths':
                # Split by commas or whitespace
                vals = [v.strip() for v in block.replace('\n', ' ').split(',') if v.strip()]
                try:
                    header['wavelengths'] = [float(v.split()[0]) for v in vals]
                except ValueError:
                    header['wavelengths'] = None
        else:
            value = value.strip()
            if key_l in ('samples', 'lines', 'bands', 'header offset', 'data type', 'byte order'):
                try:
                    header[key_l] = int(value)
                except ValueError:
                    header[key_l] = int(float(value))
            elif key_l == 'interleave':
                header[key_l] = value.lower()
            elif key_l == 'file type':
                header[key_l] = value
            else:
                header[key_l] = value
        i += 1

    # Basic validation and defaults
    for required in ('samples', 'lines', 'bands', 'data type', 'interleave'):
        if required not in header:
            raise ValueError(f"Missing '{required}' in ENVI header: {hdr_path}")

    header.setdefault('byte order', 0)
    header.setdefault('header offset', 0)
    return header


def _dtype_from_envi(data_type_code: int, byte_order: int) -> np.dtype:
    base = _ENVI_DTYPE_MAP.get(data_type_code)
    if base is None:
        raise ValueError(f"Unsupported ENVI data type code: {data_type_code}")
    endian = _ENVI_BYTE_ORDER_MAP.get(byte_order, '<')
    return np.dtype(endian + base.str[1:])


def load_envi(hdr_path: str | os.PathLike) -> Tuple[np.ndarray, Dict[str, object], Path]:
    """Load an ENVI image cube -> (cube, header, raw_path).

    Returns a NumPy memmap array shaped (lines, samples, bands) in BSQ-like layout
    in memory (regardless of file interleave, data is arranged in that order in the
    returned array). Also returns the parsed header and the associated .raw path.
    """
    hdrp = Path(hdr_path)
    if hdrp.suffix.lower() != '.hdr':
        hdrp = hdrp.with_suffix('.hdr')
    header = _parse_envi_header(hdrp)

    samples = int(header['samples'])
    lines = int(header['lines'])
    bands = int(header['bands'])
    interleave = str(header['interleave']).lower()
    byte_order = int(header.get('byte order', 0))
    header_offset = int(header.get('header offset', 0))
    dtype = _dtype_from_envi(int(header['data type']), byte_order)

    # RAW path: try same stem with .raw next to .hdr
    raw_path = hdrp.with_suffix('.raw')
    if not raw_path.exists():
        # Sometimes data file is named in header as 'data file'
        data_file = header.get('data file') or header.get('data filename') or header.get('file')
        if data_file:
            candidate = Path(data_file)
            raw_path = candidate if candidate.is_absolute() else hdrp.parent / candidate
    if not raw_path.exists():
        raise FileNotFoundError(f"ENVI data file not found for header: {hdrp}")

    # Create memmap according to interleave
    total_elems = lines * samples * bands
    mm = np.memmap(raw_path, mode='r', dtype=dtype, offset=header_offset, shape=(total_elems,))

    if interleave == 'bsq':
        arr = mm.reshape((bands, lines, samples))
        cube = np.transpose(arr, (1, 2, 0))  # -> (lines, samples, bands)
    elif interleave == 'bil':
        arr = mm.reshape((lines, bands, samples))
        cube = np.transpose(arr, (0, 2, 1))
    elif interleave == 'bip':
        cube = mm.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")

    # Return as read-only view to avoid accidental writes on memmap
    cube.flags.writeable = False
    return cube, header, raw_path


def crop_cube(
    cube: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    band_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Crop spatial region and optional band range from cube.

    - x, y: top-left column/row (0-based)
    - width, height: spatial size
    - band_range: (start_band_inclusive, end_band_exclusive) or None
    Returns a new ndarray (not a view) shaped (height, width, bands_subset).
    """
    if width <= 0 or height <= 0:
        raise ValueError('width/height must be positive')
    if x < 0 or y < 0 or x + width > cube.shape[1] or y + height > cube.shape[0]:
        raise ValueError('Requested crop is out of bounds')
    if band_range is not None:
        b0, b1 = band_range
        if b0 < 0 or b1 <= b0 or b1 > cube.shape[2]:
            raise ValueError('Invalid band_range')
        subset = cube[y:y + height, x:x + width, b0:b1]
    else:
        subset = cube[y:y + height, x:x + width, :]
    return np.ascontiguousarray(subset)


def _envi_header_text(
    samples: int,
    lines: int,
    bands: int,
    data_type_code: int,
    interleave: str,
    byte_order: int,
    header_offset: int,
    wavelengths: Optional[Sequence[float]] = None,
) -> str:
    parts: List[str] = [
        'ENVI',
        f'samples = {samples}',
        f'lines   = {lines}',
        f'bands   = {bands}',
        f'header offset = {header_offset}',
        'file type = ENVI Standard',
        f'data type = {data_type_code}',
        f'interleave = {interleave}',
        f'byte order = {byte_order}',
    ]
    if wavelengths is not None:
        wl_items = ',\n  '.join(f'{w:.6f}' for w in wavelengths)
        parts.append('wavelengths = {\n  ' + wl_items + '\n}')
    return '\n'.join(parts) + '\n'


def _envi_data_type_code(dtype: np.dtype) -> int:
    # Strip endianness and get base kind/size mapping to ENVI codes
    dt = np.dtype(dtype)
    base = np.dtype(dt.str[-1] + dt.str[-2]) if dt.itemsize >= 10 else dt
    for code, npdt in _ENVI_DTYPE_MAP.items():
        if np.dtype(npdt).kind == dt.kind and np.dtype(npdt).itemsize == dt.itemsize:
            return code
    # Fallback for common types
    if dt == np.uint16:
        return 12
    if dt == np.float32:
        return 4
    if dt == np.float64:
        return 5
    raise ValueError(f'Cannot map dtype to ENVI code: {dtype}')


def save_envi(
    cube: np.ndarray,
    out_base_path: str | os.PathLike,
    template_header: Optional[Dict[str, object]] = None,
    interleave: str = 'bsq',
) -> Tuple[Path, Path]:
    """Save cube to ENVI .hdr/.raw next to out_base_path.

    Returns (hdr_path, raw_path).
    """
    out_base = Path(out_base_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    lines, samples, bands = cube.shape
    dt = np.dtype(cube.dtype)
    byte_order = 0  # write little-endian
    data_type_code = _envi_data_type_code(dt)
    header_offset = 0
    wavelengths = None
    if template_header and template_header.get('wavelengths'):
        wavelengths = list(template_header['wavelengths'])
        if len(wavelengths) != bands:
            wavelengths = None

    hdr_txt = _envi_header_text(
        samples=samples,
        lines=lines,
        bands=bands,
        data_type_code=data_type_code,
        interleave=interleave,
        byte_order=byte_order,
        header_offset=header_offset,
        wavelengths=wavelengths,
    )

    hdr_path = out_base.with_suffix('.hdr')
    raw_path = out_base.with_suffix('.raw')
    hdr_path.write_text(hdr_txt, encoding='utf-8')

    # Write data according to chosen interleave
    arr = cube
    if interleave == 'bsq':
        to_write = np.transpose(arr, (2, 0, 1))  # (bands, lines, samples)
    elif interleave == 'bil':
        to_write = np.transpose(arr, (0, 2, 1))  # (lines, bands, samples)
    elif interleave == 'bip':
        to_write = arr  # (lines, samples, bands)
    else:
        raise ValueError('interleave must be one of bsq/bil/bip')

    with open(raw_path, 'wb') as f:
        to_write.astype(dt.newbyteorder('<'), copy=False).tofile(f)

    return hdr_path, raw_path


def _auto_rgb_band_indices(
    wavelengths: Optional[Sequence[float]],
    bands: int,
    default_triplet: Tuple[int, int, int] = (60, 40, 20),
) -> Tuple[int, int, int]:
    if not wavelengths or len(wavelengths) != bands:
        r, g, b = default_triplet
        r = min(max(r, 0), bands - 1)
        g = min(max(g, 0), bands - 1)
        b = min(max(b, 0), bands - 1)
        return r, g, b
    targets = (650.0, 550.0, 450.0)  # nm
    idxs: List[int] = []
    w = np.array(wavelengths, dtype=float)
    for t in targets:
        idxs.append(int(np.argmin(np.abs(w - t))))
    return tuple(idxs)  # type: ignore[return-value]


def _contrast_stretch_uint8(channel: np.ndarray) -> np.ndarray:
    c = channel.astype(np.float64, copy=False)
    p2, p98 = np.percentile(c, [2, 98])
    if p98 <= p2:
        p2, p98 = float(c.min()), float(c.max())
        if p98 <= p2:
            return np.zeros_like(channel, dtype=np.uint8)
    out = (np.clip(c, p2, p98) - p2) / (p98 - p2)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return out


def save_rgb_preview_ppm(
    cube: np.ndarray,
    out_path: str | os.PathLike,
    header: Optional[Dict[str, object]] = None,
    rgb_bands: Optional[Tuple[int, int, int]] = None,
) -> Path:
    """Save an RGB preview as binary PPM (P6) without external deps.

    If rgb_bands is None, attempts to pick bands near 650/550/450 nm when
    wavelengths are available; otherwise uses a simple default triplet.
    """
    h, w, b = cube.shape
    wavelengths = header.get('wavelengths') if header else None
    if rgb_bands is None:
        r_idx, g_idx, b_idx = _auto_rgb_band_indices(wavelengths, b)
    else:
        r_idx, g_idx, b_idx = rgb_bands
    r = _contrast_stretch_uint8(cube[:, :, r_idx])
    g = _contrast_stretch_uint8(cube[:, :, g_idx])
    bl = _contrast_stretch_uint8(cube[:, :, b_idx])

    # Write binary PPM (P6)
    outp = Path(out_path)
    with open(outp, 'wb') as f:
        header_text = f"P6\n{w} {h}\n255\n".encode('ascii')
        f.write(header_text)
        rgb = np.dstack([r, g, bl])
        f.write(rgb.tobytes(order='C'))
    return outp


def rgb_uint8_from_cube(
    cube: np.ndarray,
    header: Optional[Dict[str, object]] = None,
    rgb_bands: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Create an 8-bit RGB image from a cube with contrast stretch per channel."""
    h, w, b = cube.shape
    wavelengths = header.get('wavelengths') if header else None
    if rgb_bands is None:
        r_idx, g_idx, b_idx = _auto_rgb_band_indices(wavelengths, b)
    else:
        r_idx, g_idx, b_idx = rgb_bands
    r = _contrast_stretch_uint8(cube[:, :, r_idx])
    g = _contrast_stretch_uint8(cube[:, :, g_idx])
    bl = _contrast_stretch_uint8(cube[:, :, b_idx])
    return np.dstack([r, g, bl])


def overlay_grid_dots(
    rgb: np.ndarray,
    step: int = 100,
    color: Tuple[int, int, int] = (255, 0, 0),
    dot_size: int = 1,
) -> np.ndarray:
    """Overlay grid of dots every `step` pixels on a copy of the RGB image.

    dot_size is the half-size of a square dot: 1 -> 3x3, 0 -> single pixel.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB array')
    out = rgb.copy()
    h, w, _ = out.shape
    rr = range(0, h, max(step, 1))
    cc = range(0, w, max(step, 1))
    r0, g0, b0 = color
    d = max(dot_size, 0)
    for y in rr:
        y0 = max(0, y - d)
        y1 = min(h, y + d + 1)
        for x in cc:
            x0 = max(0, x - d)
            x1 = min(w, x + d + 1)
            out[y0:y1, x0:x1, 0] = r0
            out[y0:y1, x0:x1, 1] = g0
            out[y0:y1, x0:x1, 2] = b0
    return out


def _digit_font_5x3() -> Dict[str, np.ndarray]:
    # 5 rows x 3 cols bitmap font for digits 0-9 and comma, space
    patterns = {
        '0': [
            '111',
            '101',
            '101',
            '101',
            '111',
        ],
        '1': [
            '010',
            '110',
            '010',
            '010',
            '111',
        ],
        '2': [
            '111',
            '001',
            '111',
            '100',
            '111',
        ],
        '3': [
            '111',
            '001',
            '111',
            '001',
            '111',
        ],
        '4': [
            '101',
            '101',
            '111',
            '001',
            '001',
        ],
        '5': [
            '111',
            '100',
            '111',
            '001',
            '111',
        ],
        '6': [
            '111',
            '100',
            '111',
            '101',
            '111',
        ],
        '7': [
            '111',
            '001',
            '010',
            '100',
            '100',
        ],
        '8': [
            '111',
            '101',
            '111',
            '101',
            '111',
        ],
        '9': [
            '111',
            '101',
            '111',
            '001',
            '111',
        ],
        ',': [
            '000',
            '000',
            '000',
            '010',
            '100',
        ],
        ' ': [
            '000',
            '000',
            '000',
            '000',
            '000',
        ],
        'x': [
            '101',
            '010',
            '010',
            '010',
            '101',
        ],
    }
    font: Dict[str, np.ndarray] = {}
    for ch, rows in patterns.items():
        font[ch] = np.array([[1 if c == '1' else 0 for c in row] for row in rows], dtype=np.uint8)
    return font


_FONT_5x3 = _digit_font_5x3()


def _draw_text(rgb: np.ndarray, text: str, top: int, left: int, color: Tuple[int, int, int]) -> None:
    # Draw text using 5x3 font with 1px spacing horizontally and vertically
    r0, g0, b0 = color
    cursor_x = left
    cursor_y = top
    char_spacing = 1
    for ch in text:
        glyph = _FONT_5x3.get(ch)
        if glyph is None:
            glyph = _FONT_5x3[' ']
        h, w = glyph.shape
        y0 = cursor_y
        y1 = min(rgb.shape[0], y0 + h)
        x0 = cursor_x
        x1 = min(rgb.shape[1], x0 + w)
        gy1 = y1 - y0
        gx1 = x1 - x0
        if gy1 > 0 and gx1 > 0:
            m = glyph[:gy1, :gx1].astype(bool)
            sub = rgb[y0:y1, x0:x1]
            sub[m, 0] = r0
            sub[m, 1] = g0
            sub[m, 2] = b0
        cursor_x += w + char_spacing


def overlay_grid_numbers(
    rgb: np.ndarray,
    step: int = 100,
    color: Tuple[int, int, int] = (255, 255, 0),
    offset: Tuple[int, int] = (2, 2),
    mode: str = 'xy',
) -> np.ndarray:
    """Overlay index numbers near each grid point.

    mode: 'xy' -> label as "x,y" (column,row), 'i' -> sequential index per point.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 RGB array')
    out = rgb.copy()
    h, w, _ = out.shape
    oy, ox = offset
    rr = list(range(0, h, max(step, 1)))
    cc = list(range(0, w, max(step, 1)))
    idx = 0
    for y in rr:
        for x in cc:
            if mode == 'i':
                label = str(idx)
            else:
                label = f"{x},{y}"
            y_text = min(max(0, y + oy), h - 5)
            x_text = min(max(0, x + ox), w - 3)
            _draw_text(out, label, top=y_text, left=x_text, color=color)
            idx += 1
    return out


def save_ppm_from_rgb(rgb: np.ndarray, out_path: str | os.PathLike) -> Path:
    if rgb.dtype != np.uint8:
        raise ValueError('RGB array must be uint8')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB array must be HxWx3')
    h, w, _ = rgb.shape
    outp = Path(out_path)
    with open(outp, 'wb') as f:
        header_text = f"P6\n{w} {h}\n255\n".encode('ascii')
        f.write(header_text)
        f.write(rgb.tobytes(order='C'))
    return outp


def save_png_from_rgb(rgb: np.ndarray, out_path: str | os.PathLike) -> Path:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise ImportError('Pillow is required to save PNG previews. Install with: pip install pillow') from exc
    if rgb.dtype != np.uint8:
        raise ValueError('RGB array must be uint8')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB array must be HxWx3')
    img = Image.fromarray(rgb, mode='RGB')
    outp = Path(out_path)
    img.save(outp, format='PNG')
    return outp


def load_crop_save_preview(
    hdr_path: str | os.PathLike,
    x: int,
    y: int,
    width: int,
    height: int,
    out_base_path: str | os.PathLike,
    rgb_preview_path: Optional[str | os.PathLike] = None,
    band_range: Optional[Tuple[int, int]] = None,
    interleave: str = 'bsq',
) -> Tuple[Path, Path, Path]:
    """Convenience: load HSI, crop, save ENVI, save RGB preview.

    Returns (cropped_hdr_path, cropped_raw_path, preview_path).
    """
    cube, header, _ = load_envi(hdr_path)
    cropped = crop_cube(cube, x=x, y=y, width=width, height=height, band_range=band_range)
    hdr_out, raw_out = save_envi(cropped, out_base_path=out_base_path, template_header=header, interleave=interleave)
    if rgb_preview_path is None:
        rgb_preview_path = str(Path(out_base_path).with_suffix('.ppm'))
    preview_out = save_rgb_preview_ppm(cropped, out_path=rgb_preview_path, header=header)
    return hdr_out, raw_out, preview_out


__all__ = [
    'load_envi',
    'crop_cube',
    'save_envi',
    'save_rgb_preview_ppm',
    'rgb_uint8_from_cube',
    'overlay_grid_dots',
    'overlay_grid_numbers',
    'save_ppm_from_rgb',
    'save_png_from_rgb',
    'load_crop_save_preview',
]


