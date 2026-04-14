"""
usda.py
-------
Handles all communication with the USDA FoodData Central API.
Provides search, detail fetch, and macro extraction helpers used
by the retrieval pipeline.

API docs: https://fdc.nal.usda.gov/api-guide.html
"""

import os
import requests

USDA_API_KEY = os.getenv("USDA_API_KEY", "")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def search_food_candidates(query: str, top_k: int = 15) -> list:
    """
    Search USDA FoodData Central for food candidates matching `query`.
    Restricts results to Foundation Foods and SR Legacy datasets,
    which contain plain whole-ingredient entries with reliable macro data.

    Args:
        query:  Search string, e.g. "chicken breast cooked"
        top_k:  Maximum number of candidates to return (default 15)

    Returns:
        List of food candidate dicts (fdcId, description, dataType, ...)
    """
    res = requests.post(
        f"{BASE_URL}/foods/search?api_key={USDA_API_KEY}",
        json={
            "query": query,
            "pageSize": top_k,
            "dataType": ["Foundation", "SR Legacy"],
        },
    )
    res.raise_for_status()
    return res.json().get("foods", [])


def get_food_details(fdc_id: int) -> dict:
    """
    Fetch full nutrient and portion details for a specific USDA food entry.

    Returns: Full food detail dict including foodNutrients and foodPortions
    """
    res = requests.get(f"{BASE_URL}/food/{fdc_id}?api_key={USDA_API_KEY}")
    res.raise_for_status()
    return res.json()


def extract_macros(food_data: dict) -> dict:
    """
    Extract the four key macronutrients from a USDA food detail record.
    Maps standardised USDA nutrient IDs to calories, protein, carbs, fat.
    All values are per 100g of the food.

    Falls back to Atwater calculation (4/4/9 kcal per g protein/carb/fat)
    if the USDA record does not include an energy entry.

    Returns:
        dict: {calories, protein, carbs, fat} all as floats, per 100g
    """
    macros = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

    for n in food_data.get("foodNutrients", []):
        meta = n.get("nutrient", {})
        num  = str(meta.get("number", ""))
        name = str(meta.get("name", "")).lower()
        unit = str(meta.get("unitName", "")).lower()
        val  = float(n.get("amount") or 0)

        if num == "1003" or name == "protein":
            macros["protein"] = val
        elif num == "1004" or "total lipid" in name or "total fat" in name:
            macros["fat"] = val
        elif num == "1005" or "carbohydrate" in name:
            macros["carbs"] = val
        elif num == "1008" or ("energy" in name and unit == "kcal"):
            if val > 0:
                macros["calories"] = val

    # Atwater fallback 
    if macros["calories"] == 0:
        macros["calories"] = round(
            4 * macros["protein"] + 4 * macros["carbs"] + 9 * macros["fat"], 2
        )

    return macros