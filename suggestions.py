"""
suggestions.py
Planning & Optimisation subdomain: gap analysis engine.

Given a user's accumulated daily macro totals vs RDI targets, the engine:
  1. Identifies which macros are most deficient (> 20% of target remaining)
  2. Uses LLM to generate 6 targeted food search queries
  3. Retrieves USDA macro profiles for each candidate food
  4. Uses LLM to select the 3 best gap-closing foods from the candidate pool
"""

import json
import re
import time
import requests
import os

from usda import search_food_candidates, get_food_details, extract_macros
from parser import groq_request  

TEXT_MODEL = "llama-3.3-70b-versatile"

# Noise terms to filter from suggestion candidates
NOISY_TERMS = [
    "sandwich", "salad", "cracker", "crackers", "wrap", "restaurant",
    "mcdonald", "wendy", "kfc", "burger", "beverage", "drink", "mix",
    "cookie", "cake", "dessert", "spread", "baby food", "infant", "formula",
]


def _filter_noise(candidates: list) -> list:
    """Remove noisy USDA candidates from suggestion pool."""
    filtered = [
        c for c in candidates
        if not any(t in (c.get("description", "")).lower() for t in NOISY_TERMS)
    ]
    return filtered if filtered else candidates


def _get_deficient_macros(rdi: dict) -> list:
    """
    Identify macros with > 20% of daily target still remaining,
    sorted by largest deficit first.

    Returns:
        List of macro names (e.g. ["protein", "carbs"])
    """
    deficient = [
        (macro, vals)
        for macro, vals in rdi.items()
        if vals["remaining"] > 0 and (vals["remaining"] / vals["target"]) > 0.2
    ]
    # Sort by largest proportional gap
    deficient.sort(key=lambda x: x[1]["remaining"] / x[1]["target"], reverse=True)
    return [macro for macro, _ in deficient]


def _fetch_food_macros(query: str) -> dict | None:
    """
    Simple USDA lookup for a suggestion candidate (no LLM query generation —
    suggestion queries are already LLM-generated and optimised).

    Returns: dict with description, calories, protein, carbs, fat, or None
    """
    candidates = search_food_candidates(query, top_k=10)
    if not candidates:
        return None
    filtered = _filter_noise(candidates)
    best = filtered[0]
    try:
        details = get_food_details(best["fdcId"])
        macros  = extract_macros(details)
        return {"description": details.get("description"), **macros}
    except Exception:
        return None


def _get_food_search_queries(deficient_macros: list, rdi: dict) -> list:
    """
    Use LLM to generate 6 food search queries targeting the identified
    macro gaps. Queries are plain ingredient names suitable for USDA search.

    Returns: List of up to 6 USDA-ready search query strings
    """
    gaps = ", ".join(
        f"{m}: {rdi[m]['remaining']}{'kcal' if m == 'calories' else 'g'} remaining"
        for m in deficient_macros
    )

    prompt = (
        f"A user needs to close these daily nutritional gaps: {gaps}\n\n"
        "Suggest 6 simple whole foods that would help. Return ONLY a JSON array "
        'of search strings suitable for a food database, e.g. '
        '["chicken breast cooked", "brown rice cooked", "almonds"].\n'
        "Rules:\n"
        "- Use plain ingredient names, not dishes\n"
        "- Include preparation state (raw, cooked, etc.)\n"
        "- No branded products\n"
        "- Variety across food groups"
    )

    try:
        data = groq_request({
            "model": TEXT_MODEL,
            "temperature": 0.3,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
        })
        raw   = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[[\s\S]*?\]", raw)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [str(q) for q in queries[:6]]
    except Exception:
        pass

    # Fallback: generic high-nutrient foods that cover common gaps
    return [
        "chicken breast cooked", "brown rice cooked", "eggs cooked",
        "avocado raw", "lentils cooked", "almonds",
    ]


def _pick_suggestions_with_llm(pool: list, rdi: dict) -> list:
    """
    Use LLM to select the 3 best foods from the candidate pool
    based on the user's specific macro gaps and dietary variety.

    Returns: List of up to 3 selected food dicts
    """
    gaps = ", ".join(
        f"{m}: need {vals['remaining']}{'kcal' if m == 'calories' else 'g'} more"
        for m, vals in rdi.items()
        if vals["remaining"] > 0
    )

    food_list = "\n".join(
        f"{i+1}. {f['description']} — "
        f"{f['calories']}kcal, {f['protein']}g protein, "
        f"{f['carbs']}g carbs, {f['fat']}g fat (per 100g)"
        for i, f in enumerate(pool)
    )

    prompt = (
        "You are a nutrition assistant helping a user close their daily macro gaps.\n\n"
        f"Current gaps: {gaps}\n\n"
        f"Available foods (per 100g):\n{food_list}\n\n"
        "Pick the 3 best foods to help close these gaps. "
        "Prefer variety — don't pick the same food type twice.\n"
        "Reply with ONLY a JSON array of the candidate numbers, e.g. [1, 3, 5]"
    )

    try:
        data  = groq_request({
            "model": TEXT_MODEL,
            "temperature": 0,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": prompt}],
        })
        raw   = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\[[\d,\s]+\]", raw)
        if match:
            indices = [int(n) - 1 for n in json.loads(match.group())]
            selected = [pool[i] for i in indices if 0 <= i < len(pool)]
            if selected:
                return selected
    except Exception:
        pass

    return pool[:3]  # fallback: first 3 in pool


def get_suggestions(rdi: dict) -> dict:
    """
    Main entry point: compute gap analysis and return food suggestions.
    """
    deficient = _get_deficient_macros(rdi)

    if not deficient:
        return {"all_met": True, "deficient": [], "suggestions": []}

    # Build candidate pool: LLM queries → USDA lookup → macro profiles
    queries = _get_food_search_queries(deficient, rdi)
    seen    = set()
    pool    = []

    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        try:
            food = _fetch_food_macros(query)
            if food:
                pool.append({"query": query, **food})
        except Exception:
            pass
        time.sleep(0.5) 

    if not pool:
        return {"all_met": False, "deficient": deficient, "suggestions": []}

    suggestions = _pick_suggestions_with_llm(pool, rdi)
    return {"all_met": False, "deficient": deficient, "suggestions": suggestions}