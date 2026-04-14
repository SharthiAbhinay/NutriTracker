"""
retrieval.py

Retrieval Augmented Generation (RAG) pipeline for food macro lookup.

For each food item parsed from a meal description or photo:
  1. LLM generates an optimal USDA search query (e.g. "milk" → "milk whole fluid")
  2. USDA FoodData Central is searched for up to 15 Foundation/SR Legacy candidates
  3. A noise filter removes fast food, branded, dried/powdered, and processed items
  4. LLM selects the best match from the top 8 filtered candidates
  5. USDA detail fetch extracts macros per 100g; falls back on 404 errors
"""

import re
import time
import requests

from usda import search_food_candidates, get_food_details, extract_macros
from parser import groq_request  

TEXT_MODEL = "llama-3.3-70b-versatile"

# Noise filter — removes irrelevant USDA candidates that would produce
# incorrect macro estimates (fast food, branded products, dried forms, etc.)
NOISY_TERMS = [
    "sandwich", "cracker", "crackers", "wrap", "restaurant",
    "mcdonald", "wendy", "kfc", "burger king", "subway",
    "beverage", "cookie", "cake", "dessert bar", "cereal bar",
    "spread", "dip", "baby food", "infant", "formula",
    "protein powder", "supplement", "seasoned", "flavored",
    "dried", "powder", "condensed", "evaporated", "dehydrated",
]


def _filter_noise(candidates: list, food_name: str) -> list:
    """
    Remove noisy USDA candidates that are unlikely to match a plain
    whole-ingredient query. Special case: suppress oil results when
    the food name mentions "olive" without "oil" (to avoid matching
    olive oil when the user just said "olive").

    Returns the filtered list, or the original list if filtering
    removes all candidates (fail-safe).
    """
    name = food_name.lower()

    # Suppress oil entries when user meant the fruit, not the oil
    if "olive" in name and "oil" not in name:
        candidates = [
            c for c in candidates
            if "oil" not in (c.get("description", "")).lower()
        ]

    filtered = [
        c for c in candidates
        if not any(t in (c.get("description", "")).lower() for t in NOISY_TERMS)
    ]
    return filtered if filtered else candidates  


def _build_search_query(food_name: str) -> str:
    """
    Use LLM to generate an optimised USDA search query from a food name.
    This step is critical: naive queries like "milk" return condensed milk
    (986 kcal/cup); LLM-generated "milk whole fluid" returns the correct entry.

    Returns: Optimised USDA query string, e.g. "milk whole fluid", "oil olive"
    """
    prompt = (
        f'Convert this food name into the best USDA FoodData Central search query.\n'
        f'Food: "{food_name}"\n'
        f'Rules: include preparation state (cooked, raw, fluid, dry heat), '
        f'use simplest plain form, for oils start with "oil", avoid brand names, '
        f'2-5 words max.\nReply with ONLY the search query.'
    )
    try:
        data = groq_request({
            "model": TEXT_MODEL,
            "temperature": 0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
        })
        query = data["choices"][0]["message"]["content"].strip().lower()
        return query.strip("'\"") or food_name
    except Exception:
        return food_name  # fall back to raw name on any error


def _choose_best_candidate(food_name: str, candidates: list) -> dict:
    """
    Use LLM to select the best USDA candidate for a given food name.
    Instructs the model to prefer plain whole ingredients in their simplest
    common form and avoid dried, powdered, fast food, or branded entries.

    Args:
        food_name:  Original food name (used as context for the LLM)
        candidates: Filtered list of USDA candidate dicts

    Returns:
        Best candidate dict; falls back to candidates[0] on any error
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    lines = "\n".join(
        f"{i+1}. {c['description']} ({c['dataType']})"
        for i, c in enumerate(candidates[:8])
    )
    prompt = (
        f'Select the best USDA food match for: "{food_name}"\n'
        f'Candidates:\n{lines}\n'
        f'Rules: prefer plain whole ingredients, simplest common form, '
        f'avoid fast food/branded/mixed dishes, avoid dried or powdered '
        f'forms unless specified.\nReply with ONLY the candidate number.'
    )
    try:
        data = groq_request({
            "model": TEXT_MODEL,
            "temperature": 0,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": prompt}],
        })
        raw = data["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
    except Exception:
        pass
    return candidates[0]


def get_food_macros(food_name: str) -> dict | None:
    """
    Full RAG pipeline for a single food item. Combines LLM query generation,
    USDA search, noise filtering, LLM candidate selection, and macro extraction.

    Args:
        food_name: Food name to look up, e.g. "grilled chicken", "olive oil"

    Returns:
        dict with keys: description, food_data, calories, protein, carbs, fat
        Returns None if no valid USDA match is found.
    """
    query = _build_search_query(food_name)

    candidates = search_food_candidates(query, top_k=15)
    if not candidates:
        candidates = search_food_candidates(food_name, top_k=15)
    if not candidates:
        return None

    filtered = _filter_noise(candidates, food_name)
    if not filtered:
        return None

    best = _choose_best_candidate(food_name, filtered)
    if not best:
        return None

    ordered = [best] + [c for c in filtered if c["fdcId"] != best["fdcId"]]
    for candidate in ordered[:5]:
        try:
            details = get_food_details(candidate["fdcId"])
            macros  = extract_macros(details)
            return {
                "description": details.get("description"),
                "food_data": details,
                **macros,
            }
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue  # stale USDA record — try next candidate
            raise

    return None