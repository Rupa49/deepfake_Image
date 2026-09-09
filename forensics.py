"""
forensics.py  –  Multi-Signal Forensic Analysis Module  v2.0
─────────────────────────────────────────────────────────────────────
Signals:
  1. ELA          – double-compression / paste artifacts
  2. Noise Map    – sensor noise inconsistency (Laplacian)
  3. Metadata     – EXIF / software tag inspection
  4. DCT Analysis – AI images are suspiciously smooth in frequency domain [NEW]
  5. Texture      – AI images have unnaturally uniform textures [NEW]
  6. Edge Profile – AI images have globally uniform sharpness [NEW]

Why signals 4-6 specifically catch Midjourney / DALL-E / Stable Diffusion:
  • Diffusion / GAN images have extremely smooth pixel transitions
    → very low high-frequency DCT energy
  • AI images have "plastic" textures with no micro-detail variation
    → low LBP variance across image blocks
  • Real camera images have depth-of-field / motion blur variation
    → AI images are uniformly sharp everywhere (no blur falloff)
  • AI images almost NEVER contain real camera EXIF metadata
    → metadata suspicion score is very high
"""
import io
import numpy as np
import cv2
from PIL import Image, ExifTags


# ─────────────────────────────────────────────────────────────────────────────
# 1. ELA – Error Level Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_ela(pil_img: Image.Image, quality: int = 90):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    orig_arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    reco_arr = np.array(recompressed).astype(np.float32)
    diff     = np.abs(orig_arr - reco_arr)
    ela_img  = np.clip(diff * 15, 0, 255).astype(np.uint8)
    ela_score = float(np.mean(ela_img > 20))
    return ela_img, ela_score


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise Map – sensor noise inconsistency
# ─────────────────────────────────────────────────────────────────────────────

def compute_noise_map(pil_img: Image.Image):
    gray    = np.array(pil_img.convert("L")).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise   = gray - blurred
    block   = 8
    h_b     = gray.shape[0] // block
    w_b     = gray.shape[1] // block
    stds    = []
    for r in range(h_b):
        for c in range(w_b):
            patch = noise[r*block:(r+1)*block, c*block:(c+1)*block]
            stds.append(float(np.std(patch)))
    stds          = np.array(stds)
    inconsistency = float(np.std(stds) / (np.mean(stds) + 1e-8))
    inconsistency = min(inconsistency, 1.0)
    norm       = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    noise_heat = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_INFERNO)
    noise_img  = cv2.cvtColor(noise_heat, cv2.COLOR_BGR2RGB)
    return noise_img, inconsistency


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metadata Inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_metadata(pil_img: Image.Image) -> dict:
    flags     = []
    suspicion = 0.0
    software  = None
    has_exif  = False

    exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
    if exif_data:
        has_exif = True
        readable = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
        software = readable.get("Software", None)
        if software:
            ai_keywords = ["stable diffusion", "midjourney", "dall", "gan", "deepfake",
                           "faceswap", "roop", "insightface", "diffusion", "generated",
                           "adobe firefly", "bing image", "canva", "nightcafe", "runway",
                           "leonardo", "playground", "bluewillow"]
            if any(kw in str(software).lower() for kw in ai_keywords):
                flags.append(f"⚠ AI generation software detected: '{software}'")
                suspicion += 0.80
        for expected in ["Make", "Model", "LensModel"]:
            if expected not in readable:
                flags.append(f"Missing EXIF field: {expected}")
                suspicion += 0.08
        if not readable.get("GPSInfo"):
            flags.append("No GPS data (inconclusive)")
    else:
        has_exif = False
        # AI images almost NEVER have real EXIF – strong signal
        flags.append("⚠ No EXIF metadata – highly associated with AI-generated images")
        suspicion += 0.55
    if pil_img.mode not in ("RGB", "L"):
        flags.append(f"Unusual image mode: {pil_img.mode}")
        suspicion += 0.1

    suspicion = min(suspicion, 1.0)
    return {"has_exif": has_exif, "software": software,
            "flags": flags, "suspicion": suspicion}


# ─────────────────────────────────────────────────────────────────────────────
# 4. DCT Coefficient Analysis 
#    AI images are too smooth → very low high-frequency DCT energy
# ─────────────────────────────────────────────────────────────────────────────

def compute_dct_score(pil_img: Image.Image):
    """
    Split image into 8x8 blocks, compute 2D DCT per block.
    Measure ratio of high-frequency to total energy.
    AI images: suspiciously LOW HF ratio (too smooth / no grain).
    Score: higher = more AI-like (more suspicious).
    """
    gray  = np.array(pil_img.convert("L")).astype(np.float32)
    h, w  = gray.shape
    block = 8
    h_b, w_b = h // block, w // block
    hf_ratios = np.zeros((h_b, w_b), dtype=np.float32)

    for r in range(h_b):
        for c in range(w_b):
            patch = gray[r*block:(r+1)*block, c*block:(c+1)*block]
            dct   = cv2.dct(patch)
            total = np.sum(dct ** 2) + 1e-8
            hf    = np.sum(dct[4:, 4:] ** 2)
            hf_ratios[r, c] = hf / total

    mean_hf  = float(np.mean(hf_ratios))
    # Baseline: real photos ~0.10. AI images often < 0.04
    baseline  = 0.10
    dct_score = float(np.clip((baseline - mean_hf) / baseline, 0.0, 1.0))

    norm     = (hf_ratios - hf_ratios.min()) / (hf_ratios.max() - hf_ratios.min() + 1e-8)
    vis      = cv2.resize(np.uint8(norm * 255), (w, h), interpolation=cv2.INTER_NEAREST)
    dct_heat = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    dct_img  = cv2.cvtColor(dct_heat, cv2.COLOR_BGR2RGB)
    return dct_img, dct_score, mean_hf


# ─────────────────────────────────────────────────────────────────────────────
# 5. Texture Uniformity Score [NEW]
#    AI images have unnaturally uniform, "plastic" textures.
# ─────────────────────────────────────────────────────────────────────────────

def compute_texture_score(pil_img: Image.Image):
    """
    Local Binary Pattern (LBP) texture analysis.
    Low LBP variance → image is too smooth → suspicious.
    """
    gray = np.array(pil_img.convert("L"))
    lbp  = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        rolled = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp   += (gray >= rolled).astype(np.uint8)

    h, w   = gray.shape
    block  = 16
    h_b    = h // block
    w_b    = w // block
    bvars  = []
    for r in range(h_b):
        for c in range(w_b):
            patch = lbp[r*block:(r+1)*block, c*block:(c+1)*block]
            bvars.append(float(np.std(patch.astype(float))))

    mean_var  = float(np.mean(bvars)) if bvars else 0.0
    # Real photos: ~1.5-3.0. AI images: often < 1.0
    baseline  = 1.8
    tex_score = float(np.clip((baseline - mean_var) / baseline, 0.0, 1.0))

    lbp_norm = ((lbp.astype(float) / 8.0) * 255).astype(np.uint8)
    tex_heat = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_PLASMA)
    tex_img  = cv2.cvtColor(tex_heat, cv2.COLOR_BGR2RGB)
    return tex_img, tex_score, mean_var


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge Sharpness Profile [NEW]
#    Real photos have varying sharpness (blur falloff). AI = uniformly sharp.
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_score(pil_img: Image.Image):
    """
    Laplacian variance per 32x32 block.
    Real photos: HIGH block-to-block variance.
    AI images  : LOW variance (everything equally sharp).
    """
    gray     = np.array(pil_img.convert("L")).astype(np.float32)
    lap      = cv2.Laplacian(gray, cv2.CV_32F)
    h, w     = gray.shape
    block    = 32
    h_b, w_b = h // block, w // block
    sv       = []
    for r in range(h_b):
        for c in range(w_b):
            patch = lap[r*block:(r+1)*block, c*block:(c+1)*block]
            sv.append(float(np.var(patch)))

    if not sv:
        return np.zeros((h, w, 3), dtype=np.uint8), 0.0, 0.0

    sv         = np.array(sv)
    cov        = float(np.std(sv) / (np.mean(sv) + 1e-8))
    baseline   = 1.5
    edge_score = float(np.clip((baseline - cov) / baseline, 0.0, 1.0))
    edge_score = max(0.0, edge_score)

    lap_vis   = np.abs(lap)
    lap_norm  = np.clip(lap_vis / (lap_vis.max() + 1e-8), 0, 1)
    edge_heat = cv2.applyColorMap(np.uint8(lap_norm * 255), cv2.COLORMAP_BONE)
    edge_img  = cv2.cvtColor(edge_heat, cv2.COLOR_BGR2RGB)
    return edge_img, edge_score, cov


# ─────────────────────────────────────────────────────────────────────────────
# Master
# ─────────────────────────────────────────────────────────────────────────────

def run_full_forensics(pil_img: Image.Image) -> dict:
    """
    Weights:
      Metadata     0.25  (strongest single signal – AI images have no EXIF)
      DCT          0.20  (frequency smoothness – catches diffusion models)
      Texture      0.15  (LBP uniformity)
      ELA          0.15
      Noise        0.15
      Edge         0.10
    """
    ela_img,  ela_score          = compute_ela(pil_img)
    noise_img,noise_score        = compute_noise_map(pil_img)
    meta                         = inspect_metadata(pil_img)
    dct_img,  dct_score, hf_val  = compute_dct_score(pil_img)
    tex_img,  tex_score, lbp_val = compute_texture_score(pil_img)
    edge_img, edge_score, cov    = compute_edge_score(pil_img)

    forensic_score = (
        0.25 * meta["suspicion"] +
        0.20 * dct_score         +
        0.15 * ela_score         +
        0.15 * noise_score       +
        0.15 * tex_score         +
        0.10 * edge_score
    )
    forensic_score = min(forensic_score, 1.0)

    return {
        "ela_img": ela_img, "ela_score": ela_score,
        "noise_img": noise_img, "noise_score": noise_score,
        "metadata": meta,
        "dct_img": dct_img, "dct_score": dct_score, "hf_energy": hf_val,
        "tex_img": tex_img, "tex_score": tex_score, "lbp_variance": lbp_val,
        "edge_img": edge_img, "edge_score": edge_score, "sharpness_cov": cov,
        "forensic_score": forensic_score,
    }