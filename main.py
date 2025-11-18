import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageStat
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdjustmentSuggestion(BaseModel):
    name: str
    value: int = Field(..., ge=-100, le=100)
    rationale: str


class AnalysisResult(BaseModel):
    pros: List[str]
    cons: List[str]
    suggestions: List[AdjustmentSuggestion]


@app.get("/")
def read_root():
    return {"message": "Image Analysis API is running"}


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Convert to numpy for stats
    np_img = np.asarray(image).astype(np.float32) / 255.0

    # Global brightness and contrast
    luminance = 0.2126 * np_img[..., 0] + 0.7152 * np_img[..., 1] + 0.0722 * np_img[..., 2]
    mean_lum = float(np.mean(luminance))
    std_lum = float(np.std(luminance))

    # Highlights and shadows estimation
    highlights = float(np.mean(luminance[luminance > 0.85])) if np.any(luminance > 0.85) else 0.0
    shadows = float(np.mean(luminance[luminance < 0.15])) if np.any(luminance < 0.15) else 0.0

    # Clipping detection
    pct_white_clip = float(np.mean(luminance >= 0.98)) * 100.0
    pct_black_clip = float(np.mean(luminance <= 0.02)) * 100.0

    # Color cast: compare channel means to neutral gray
    means = np.mean(np_img, axis=(0, 1))
    r_m, g_m, b_m = [float(x) for x in means]
    rg_diff = r_m - g_m
    gb_diff = g_m - b_m

    pros = []
    cons = []

    # Pros
    if std_lum > 0.12:
        pros.append("Good overall contrast")
    if 0.35 <= mean_lum <= 0.65:
        pros.append("Balanced exposure")
    if pct_white_clip < 1.0 and pct_black_clip < 1.0:
        pros.append("No significant clipping")

    # Cons
    if std_lum < 0.08:
        cons.append("Image looks a bit flat (low contrast)")
    if mean_lum < 0.35:
        cons.append("Overall underexposed")
    if mean_lum > 0.65:
        cons.append("Overall overexposed")
    if pct_white_clip >= 1.0:
        cons.append("Highlights are clipping")
    if pct_black_clip >= 1.0:
        cons.append("Shadows are clipping")
    if abs(rg_diff) > 0.03 or abs(gb_diff) > 0.03:
        cons.append("Slight color cast detected")

    suggestions: List[AdjustmentSuggestion] = []

    # iPhone-style adjustments heuristic mapping to [-100, 100]
    # Exposure: target mean ~0.5
    exposure_adj = int(np.clip((0.5 - mean_lum) * 200, -100, 100))
    if abs(exposure_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Exposure", value=exposure_adj,
                          rationale="Align overall brightness toward a balanced midtone"))

    # Brilliance: combo of exposure and contrast to bring detail
    brilliance_adj = int(np.clip((0.5 - mean_lum) * 120 + (0.15 - std_lum) * 250, -100, 100))
    if abs(brilliance_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Brilliance", value=brilliance_adj,
                          rationale="Recover details in highlights and shadows while boosting clarity"))

    # Highlights: reduce if highlight mean high or white clipping
    highlights_adj = int(np.clip((0.80 - highlights) * 300 - (pct_white_clip * 0.8), -100, 100))
    if abs(highlights_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Highlights", value=highlights_adj,
                          rationale="Protect bright areas from clipping and restore detail"))

    # Shadows: lift if deep shadows or black clipping
    shadows_adj = int(np.clip(((0.12 - shadows) * 500) + (pct_black_clip * 0.8), -100, 100))
    if abs(shadows_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Shadows", value=shadows_adj,
                          rationale="Recover details in dark regions without washing out contrast"))

    # Contrast: based on luminance std
    contrast_adj = int(np.clip((0.15 - std_lum) * 600, -80, 80))
    if abs(contrast_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Contrast", value=contrast_adj,
                          rationale="Add separation between tones to avoid a flat look"))

    # Brightness: small offset to midtones separate from exposure
    brightness_adj = int(np.clip((0.52 - mean_lum) * 140, -60, 60))
    if abs(brightness_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Brightness", value=brightness_adj,
                          rationale="Fine-tune midtones after setting exposure"))

    # Black Point: raise if blacks too lifted; lower if clipped
    # Aim for ~0.02 blacks; positive value in iPhone UI increases black point (deepens blacks)
    target_black = 0.02
    black_bias = (target_black - shadows) * 2000 - (pct_black_clip * 0.5)
    black_point_adj = int(np.clip(black_bias, -100, 100))
    if abs(black_point_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Black Point", value=black_point_adj,
                          rationale="Set a clean shadow anchor without crushing detail"))

    # Saturation/Vibrance via color variance
    sat_proxy = float(np.std(np_img, axis=(0, 1)).mean())  # crude proxy
    saturation_adj = int(np.clip((0.12 - sat_proxy) * 500, -60, 60))
    if abs(saturation_adj) >= 5:
        suggestions.append(AdjustmentSuggestion(name="Saturation", value=saturation_adj,
                          rationale="Enhance color richness while keeping skin tones natural"))

    # Warmth/Tint suggestion from color cast
    if rg_diff > 0.03:
        cons.append("Image leans warm (red over green)")
    elif rg_diff < -0.03:
        cons.append("Image leans cool/green")
    if gb_diff > 0.03:
        cons.append("Image leans green over blue")
    elif gb_diff < -0.03:
        cons.append("Image leans magenta/blue")

    result = AnalysisResult(pros=pros or ["Nice composition and clean tones"],
                            cons=cons or ["Minor tweaks can further enhance pop"],
                            suggestions=suggestions)
    return result


@app.get("/test")
def test_database():
    return {"backend": "✅ Running"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
