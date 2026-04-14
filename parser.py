"""
parser.py
---------
Parses free-text meal descriptions into structured food items using
LLaMA 3.3 70B via the Groq API. Handles both simple whole foods and
composite dishes via few-shot prompting.
"""

import json
import re
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TEXT_MODEL = "llama-3.3-70b-versatile"

# Few-shot system prompt — teaches the model to decompose composite dishes
# into ingredients AND handle simple whole foods naturally.
SYSTEM_PROMPT = """You are a meal parser and recipe decomposer. Extract food items from the user's meal description.

If the user mentions a composite dish (e.g. tiramisu, pizza, burger, pasta, curry, stew, sandwich), decompose it into its main ingredients with estimated gram weights.

If the user mentions simple whole foods (e.g. eggs, rice, chicken, banana), extract them as-is.

Return ONLY valid JSON in this exact format, no markdown, no explanation:
{"items": [{"name": "food name", "quantity": number, "unit": "unit"}]}

Examples:

Input: "2 eggs and 1 slice toast"
Output: {"items": [{"name": "egg", "quantity": 2, "unit": "unit"}, {"name": "bread", "quantity": 1, "unit": "slice"}]}

Input: "1 bowl of pasta carbonara"
Output: {"items": [{"name": "spaghetti cooked", "quantity": 180, "unit": "g"}, {"name": "bacon", "quantity": 40, "unit": "g"}, {"name": "egg", "quantity": 1, "unit": "unit"}, {"name": "parmesan cheese", "quantity": 20, "unit": "g"}, {"name": "black pepper", "quantity": 2, "unit": "g"}]}

Input: "1 slice pepperoni pizza"
Output: {"items": [{"name": "pizza dough", "quantity": 60, "unit": "g"}, {"name": "tomato sauce", "quantity": 30, "unit": "g"}, {"name": "mozzarella cheese", "quantity": 40, "unit": "g"}, {"name": "pepperoni", "quantity": 20, "unit": "g"}]}

Input: "1 cup oats with banana"
Output: {"items": [{"name": "oats", "quantity": 1, "unit": "cup"}, {"name": "banana", "quantity": 1, "unit": "unit"}]}

Input: "1 tiramisu"
Output: {"items": [{"name": "mascarpone cheese", "quantity": 80, "unit": "g"}, {"name": "ladyfinger biscuit", "quantity": 30, "unit": "g"}, {"name": "egg yolk", "quantity": 20, "unit": "g"}, {"name": "sugar", "quantity": 15, "unit": "g"}, {"name": "espresso", "quantity": 30, "unit": "g"}, {"name": "heavy cream", "quantity": 25, "unit": "g"}]}

Rules:
- One object per distinct food item or ingredient
- Use singular food names (egg not eggs)
- Units: use g, kg, oz, lb, cup, tbsp, tsp, slice, piece, unit
- For decomposed dishes, always use grams
- For simple whole foods, use natural units (cup, piece, unit, slice)
- quantity must be a number"""


def groq_request(payload: dict, retries: int = 6) -> dict:
    """
    POST to Groq chat completions with exponential backoff on 429 rate limits.
    Returns the parsed JSON response dict.
    """
    for attempt in range(retries):
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if res.status_code == 429:
            wait = min(2 ** attempt, 60)
            print(f"  [Rate limit] waiting {wait}s...")
            time.sleep(wait)
            continue
        res.raise_for_status()
        return res.json()
    raise RuntimeError("Groq rate limit exceeded after retries.")


def _parse_json(raw: str, fallback_text: str) -> dict:
    """
    Try to extract a valid items JSON from the LLM response.
    Falls back to regex extraction, then a simple regex parse of the
    original text if the model output is unusable.
    """
    # direct JSON parse
    try:
        parsed = json.loads(raw)
        if parsed.get("items"):
            return parsed
    except json.JSONDecodeError:
        pass

    # extract first {...} block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if parsed.get("items"):
                return parsed
        except json.JSONDecodeError:
            pass

    # extract individual name/quantity/unit triples via regex
    triples = re.findall(
        r'"name"\s*:\s*"([^"]+)"\s*,\s*"quantity"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"unit"\s*:\s*"([^"]+)"',
        raw,
    )
    if triples:
        return {
            "items": [
                {"name": n, "quantity": float(q), "unit": u}
                for n, q, u in triples
            ]
        }

    # fallback — simple regex on the original meal text
    return _fallback_parse(fallback_text)


def _fallback_parse(text: str) -> dict:
    """
    Last-resort parser: splits on commas/and and extracts
    (quantity, unit, food_name) triples with a simple regex.
    """
    parts = re.split(r",|\band\b", text.lower())
    items = []
    for part in parts:
        m = re.match(
            r"^(\d+(?:\.\d+)?)\s*(cup|slice|tbsp|tsp|g|oz|piece|unit)?\s+(.+)$",
            part.strip(),
        )
        if m:
            items.append({
                "name": m.group(3).strip().rstrip("s"),
                "quantity": float(m.group(1)),
                "unit": m.group(2) or "unit",
            })
    return {"items": items}


def parse_meal(meal_text: str) -> dict:
    """
    Parse a free-text meal description into structured items.

    Returns: dict with key "items": list of {name, quantity, unit}
    """
    data = groq_request({
        "model": TEXT_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this meal: {meal_text}"},
        ],
    })
    raw = data["choices"][0]["message"]["content"].strip()
    return _parse_json(raw, meal_text)