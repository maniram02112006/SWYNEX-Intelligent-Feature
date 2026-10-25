"""
AI Design Insight Assistant - Rule-based Heuristics Engine
SWYNEX Internship Task 3: Intelligent Feature

This module provides transparent, deterministic design insights and
recommendations based on graphic design principles and classifier probability
distributions. It does NOT call external LLMs or third-party APIs.
"""

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

CLASS_NAMES: List[str] = ["cinematic", "minimalist", "typography", "sports", "vintage"]
LOW_CONFIDENCE_THRESHOLD: float = 0.50
AMBIGUITY_MARGIN_THRESHOLD: float = 0.15

# Visual design characteristics and classifier rationale for each style
STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cinematic": {
        "title": "Cinematic Style",
        "summary": "Atmospheric, dramatic lighting with narrative visual focus.",
        "rationale": (
            "Cinematic designs rely on photographic depth of field, dramatic directional lighting, "
            "mood-driven color palettes (such as teal and orange or rich shadow tones), and "
            "storytelling composition reminiscent of film keyframes and movie posters."
        ),
        "suggestions": [
            "Depth & Atmosphere: Enhance three-dimensional depth using subtle atmospheric fog, vignetting, or rim lighting.",
            "Color Harmony: Unify lighting color temperatures across subject and background to maintain a coherent theatrical mood.",
            "Text Integration: Position movie-title typography in natural negative space using wide-tracked sans-serif or elegant serif letterforms.",
        ],
    },
    "minimalist": {
        "title": "Minimalist Style",
        "summary": "Restrained composition prioritizing negative space and focal clarity.",
        "rationale": (
            "Minimalist designs prioritize generous breathing room, stripped-down geometric or "
            "isolated focal elements, and disciplined monochromatic or duotone color schemes. "
            "Visual hierarchy is established through stark scale rather than decorative clutter."
        ),
        "suggestions": [
            "Negative Space: Preserve at least 35-50% uncluttered breathing room around your primary focal point.",
            "Scale Contrast: Create visual drama by pairing one small hero element against an expansive background.",
            "Color Discipline: Restrict your color palette to 1-2 intentional accent colors against neutral tones.",
        ],
    },
    "typography": {
        "title": "Typography Style",
        "summary": "Letterforms, layout structure, and text hierarchy as the central artwork.",
        "rationale": (
            "Typography-focused designs treat fonts, letterforms, kerning, and editorial grids as the "
            "primary expressive visual subject rather than mere informational captions. Strong contrast "
            "and distinct font hierarchies drive viewer engagement."
        ),
        "suggestions": [
            "Typographic Hierarchy: Establish unmistakable size and weight ratios (e.g., 2.5x to 3x) between headline, subhead, and body.",
            "Grid Alignment: Align all text elements to a consistent baseline grid to maintain editorial balance and rhythm.",
            "Kerning & Readability: Manually adjust optical kerning on display titles and maintain generous line-height (1.4-1.6x) on body text.",
        ],
    },
    "sports": {
        "title": "Sports Style",
        "summary": "High-energy diagonal vectors, athletic motion, and dynamic contrast.",
        "rationale": (
            "Sports graphics emphasize kinetic action, dynamic diagonal framing, high-contrast saturated "
            "palettes, and forceful athletic postures. Compositional lines pull the eye along axes of speed and impact."
        ),
        "suggestions": [
            "Kinetic Energy: Introduce 15-30 degree diagonal angles, speed streaks, or dynamic particle effects to convey momentum.",
            "High-Impact Contrast: Combine bold, saturated accent hues (electric lime, fiery orange) against dark textured backdrops.",
            "Powerful Type: Use heavy, slanted, or condensed display sans-serif fonts with bold drop shadows or cutout treatments.",
        ],
    },
    "vintage": {
        "title": "Vintage Style",
        "summary": "Nostalgic warm palettes, analog textures, and heritage ornamentation.",
        "rationale": (
            "Vintage designs evoke mid-century print, retro advertising, and heritage branding. "
            "They utilize warm, sun-faded palettes (sepia, ochre, muted sage), distressed halftone "
            "textures, and classic decorative borders or badge stamps."
        ),
        "suggestions": [
            "Analog Texture: Apply subtle paper grain, halftone dot screens, or worn-edge distressing to soften digital sharpness.",
            "Heritage Typography: Incorporate arched banner text, slab-serifs, or hand-drawn script fonts suited to the historical era.",
            "Aged Color Palette: Tone down neon digital saturation in favor of muted, archival pigments and warm off-white backgrounds.",
        ],
    },
}


def validate_probabilities(probabilities: Sequence[float], expected_count: int = 5) -> np.ndarray:
    """Validate probability vector, ensuring proper length, numeric validity, and finite values."""
    if probabilities is None:
        raise ValueError("Probabilities array cannot be None.")
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim != 1 or len(arr) != expected_count:
        raise ValueError(
            f"Expected {expected_count} class probabilities, but received shape {arr.shape}."
        )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Probabilities array contains NaN or infinite values.")
    if (arr < 0.0).any():
        raise ValueError("Probabilities cannot contain negative numbers.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("Sum of probabilities must be greater than zero.")
    # Normalize to ensure strictly sum = 1.0
    return arr / total


def analyze_design(
    probabilities: Sequence[float],
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Generate an intelligent design insight analysis based on model outputs.

    Returns a structured dictionary containing:
      - predicted_class: name of highest-scoring class
      - confidence: highest probability (0.0 to 1.0)
      - ranked_classes: list of (class_name, score) sorted descending
      - is_uncertain: boolean indicating low confidence or ambiguous top predictions
      - uncertainty_reasons: list of explanatory strings if uncertain
      - style_summary: high-level description of predicted style
      - visual_rationale: detailed explanation of the predicted style's design grammar
      - suggestions: 3 actionable design improvement tips
      - hybrid_blend_note: guidance when runner-up style is prominent (>15%)
      - disclaimer: factual note regarding softmax scores vs calibrated probabilities
    """
    arr = validate_probabilities(probabilities, expected_count=len(class_names))

    # Rank classes
    ranked_indices = np.argsort(arr)[::-1]
    ranked = [(class_names[idx], float(arr[idx])) for idx in ranked_indices]

    best_class, best_score = ranked[0]
    second_class, second_score = ranked[1]
    margin = best_score - second_score

    # Uncertainty evaluation
    uncertainty_reasons: List[str] = []
    is_low_confidence = best_score < LOW_CONFIDENCE_THRESHOLD
    is_ambiguous = margin < AMBIGUITY_MARGIN_THRESHOLD

    if is_low_confidence:
        uncertainty_reasons.append(
            f"The top prediction score ({best_score * 100:.1f}%) is below the {LOW_CONFIDENCE_THRESHOLD * 100:.0f}% confidence threshold."
        )
    if is_ambiguous:
        uncertainty_reasons.append(
            f"The prediction is ambiguous: runner-up style '{second_class.title()}' ({second_score * 100:.1f}%) "
            f"is only {margin * 100:.1f}% behind '{best_class.title()}'."
        )

    is_uncertain = len(uncertainty_reasons) > 0

    # Profile content
    profile = STYLE_PROFILES.get(
        best_class,
        {
            "title": f"{best_class.title()} Style",
            "summary": "Design style predicted by the model.",
            "rationale": "Visual patterns match the learned style profile.",
            "suggestions": ["Refine visual hierarchy.", "Check color contrast.", "Ensure clear focal points."],
        },
    )

    # Hybrid blend observation if second class is significant
    hybrid_blend_note = ""
    if second_score >= 0.15:
        hybrid_blend_note = (
            f"**Hybrid Style Detected:** This image exhibits characteristics of both "
            f"**{best_class.title()}** ({best_score * 100:.1f}%) and **{second_class.title()}** ({second_score * 100:.1f}%). "
            f"If your goal is a distinct {best_class.title()} aesthetic, amplify its core traits "
            f"and tone down {second_class.title()} cues. If a fusion is intended, ensure both elements harmonize intentionally."
        )

    disclaimer = (
        "Model scores represent raw softmax distribution across five closed categories, not calibrated "
        "real-world certainty. Because the underlying model was trained on a small illustrative dataset, "
        "predictions should serve as directional guidance and be paired with human creative judgment."
    )

    return {
        "predicted_class": best_class,
        "confidence": best_score,
        "runner_up_class": second_class,
        "runner_up_confidence": second_score,
        "margin": margin,
        "ranked_classes": ranked,
        "is_uncertain": is_uncertain,
        "uncertainty_reasons": uncertainty_reasons,
        "style_title": profile["title"],
        "style_summary": profile["summary"],
        "visual_rationale": profile["rationale"],
        "suggestions": profile["suggestions"],
        "hybrid_blend_note": hybrid_blend_note,
        "disclaimer": disclaimer,
    }
