"""
normalise.py
------------
Converts food quantities from user-specified units (cups, tbsp, slices, etc.)
into gram weights for macro scaling.
  1. Physical units (g, kg, oz, lb) — always deterministic
  2. USDA foodPortions data — exact match on portion description/modifier
  3. Standard generic fallback weights (USDA serving size guidelines)
"""

# Physical unit conversion factors (to grams)
PHYSICAL = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "pound": 453.59, "pounds": 453.59,
}

# Aliases for common household units — used to match USDA portionDescription
UNIT_ALIASES = {
    "cup":   ["cup", "c."],
    "tbsp":  ["tablespoon", "tbsp", "tbs"],
    "tsp":   ["teaspoon", "tsp"],
    "slice": ["slice", "slc"],
    "piece": ["piece", "pce"],
    "unit":  ["medium", "large", "small", "each"],
}

# Based on USDA serving size guidelines; used when USDA foodPortions data is absent or does not contain an exact unit match.
UNIT_GRAMS = {
    "cup":   240.0,
    "tbsp":  15.0,
    "tsp":   5.0,
    "slice": 30.0,
    "piece": 100.0,
    "unit":  100.0,
}


def estimate_grams(food_data: dict, quantity: float, unit: str) -> float:
    """
    Convert a food quantity + unit into grams.

    Returns: Estimated gram weight as a float
    """
    u = unit.lower().strip()

    if u in PHYSICAL:
        return quantity * PHYSICAL[u]

    # consult USDA foodPortions if available
    # Only use a USDA portion if it exactly matches our unit alias
    # and has a plausible gram weight (5g–500g).
    portions = (food_data or {}).get("foodPortions", [])
    aliases  = UNIT_ALIASES.get(u, [u])

    for portion in portions:
        desc     = (portion.get("portionDescription") or "").lower()
        modifier = (portion.get("modifier") or "").lower()
        grams    = portion.get("gramWeight")
        if not grams:
            continue
        if any(a == desc or a == modifier for a in aliases):
            if 5 <= grams <= 500:
                return quantity * grams

    # ---reliable generic fallback ---
    return quantity * UNIT_GRAMS.get(u, 100.0)