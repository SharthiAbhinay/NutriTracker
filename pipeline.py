"""
pipeline.py
-----------
Orchestrates the full NutriTrack AI pipeline from parsed food items to
scaled macro totals and RDI comparisons.

Shared by both the text path (parser → pipeline) and
the photo path (vision → pipeline).

RDI targets (standard 2,000 kcal reference values):
  Calories : 2000 kcal
  Protein  : 50 g
  Carbs    : 275 g
  Fat      : 78 g
"""

import time

from parser     import parse_meal
from vision     import analyze_image_bytes
from retrieval  import get_food_macros
from normalise  import estimate_grams

# Recommended Daily Intake targets used for progress bar calculations
RDI = {
    "calories": 2000,
    "protein":  50,
    "carbs":    275,
    "fat":      78,
}


def _compute_rdi(totals: dict) -> dict:
    """
    Compute RDI-relative metrics for each macro.

    Returns: dict mapping macro → {consumed, target, percent, remaining}
    """
    rdi_result = {}
    for macro, target in RDI.items():
        consumed  = totals[macro]
        percent   = round((consumed / target) * 100)
        remaining = max(0.0, round((target - consumed) * 10) / 10)
        rdi_result[macro] = {
            "consumed":  consumed,
            "target":    target,
            "percent":   percent,
            "remaining": remaining,
        }
    return rdi_result


def run_pipeline(items: list) -> dict:
    """
    Core pipeline: for each parsed food item, retrieve USDA macros,
    normalise units to grams, scale macros, and accumulate totals.

    Stages per item:
      1. LLM search query generation  (retrieval.py)
      2. USDA candidate search         (usda.py)
      3. Noise filtering               (retrieval.py)
      4. LLM candidate selection       (retrieval.py)
      5. USDA detail fetch             (usda.py)
      6. Unit normalisation            (normalise.py)
      7. Macro scaling + accumulation  (here)

    Args:
        items: List of {name, quantity, unit} dicts from parser or vision

    Returns:
        dict with keys:
          parsed_items — original item list
          food_results — per-item lookup results and scaled macros
          totals — summed meal macros
          rdi — RDI comparison metrics
    """
    food_results = []
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    for item in items:
        name     = item["name"]
        quantity = item["quantity"]
        unit     = item["unit"]

        usda = get_food_macros(name)

        if not usda:
            # Food not found in USDA — record failure but continue
            food_results.append({
                "name": name, "quantity": quantity, "unit": unit,
                "found": False, "scaled": None,
            })
            continue

        # Convert quantity + unit to grams, then scale macros from per-100g
        grams  = estimate_grams(usda["food_data"], quantity, unit)
        factor = grams / 100.0

        scaled = {
            "calories": round(usda["calories"] * factor),
            "protein":  round(usda["protein"]  * factor * 10) / 10,
            "carbs":    round(usda["carbs"]     * factor * 10) / 10,
            "fat":      round(usda["fat"]       * factor * 10) / 10,
        }

        # Accumulate into meal totals
        for macro in totals:
            totals[macro] += scaled[macro]

        food_results.append({
            "name":                name,
            "quantity":            quantity,
            "unit":                unit,
            "estimated_grams":     round(grams),
            "matched_description": usda["description"],
            "found":               True,
            "scaled":              scaled,
        })

        time.sleep(1.0)

    # Round totals
    totals["calories"] = round(totals["calories"])
    totals["protein"]  = round(totals["protein"]  * 10) / 10
    totals["carbs"]    = round(totals["carbs"]     * 10) / 10
    totals["fat"]      = round(totals["fat"]       * 10) / 10

    return {
        "parsed_items": items,
        "food_results": food_results,
        "totals":       totals,
        "rdi":          _compute_rdi(totals),
    }


def analyze_meal(meal_text: str) -> dict:
    """
    Full text-to-macros pipeline.

    Returns: Pipeline result dict
    """
    parsed = parse_meal(meal_text)
    if not parsed.get("items"):
        raise ValueError("Could not parse any food items from that description.")
    return run_pipeline(parsed["items"])


def analyze_meal_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Full image-to-macros pipeline.

    Returns: Pipeline result dict 
    """
    parsed = analyze_image_bytes(image_bytes, mime_type)
    if not parsed.get("items"):
        raise ValueError("Could not identify any food items in the image.")
    return run_pipeline(parsed["items"])