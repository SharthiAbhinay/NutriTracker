"""
vision.py
---------
Analyses meal photographs using LLaMA 4 Scout 17B (multimodal) via Groq.
The model receives a base64-encoded image and returns a structured JSON
list of identified food items with gram-weight estimates.

Few-shot examples and a size reference guide are included in the prompt
to calibrate portion size estimates from overhead RGB images.
"""

import base64
import json
import re
import time
import requests
import os

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# Vision prompt — few-shot examples calibrate the model's gram estimates.
# The size reference guide anchors common food categories to expected weights.

VISION_PROMPT = """You are a meal analysis assistant with expertise in food portion estimation.
Look at this overhead food image carefully and estimate the weight of each ingredient in grams.

Return ONLY valid JSON in this exact format, no markdown, no explanation:
{"items": [{"name": "food name", "quantity": number, "unit": "g"}]}

Use these reference examples to calibrate your estimates:

Example 1 — Rice bowl with chicken and vegetables:
{"items": [{"name": "white rice", "quantity": 150, "unit": "g"}, {"name": "chicken breast", "quantity": 120, "unit": "g"}, {"name": "broccoli", "quantity": 80, "unit": "g"}, {"name": "soy sauce", "quantity": 10, "unit": "g"}]}

Example 2 — Salad plate:
{"items": [{"name": "mixed greens", "quantity": 60, "unit": "g"}, {"name": "cherry tomato", "quantity": 50, "unit": "g"}, {"name": "cucumber", "quantity": 40, "unit": "g"}, {"name": "olive oil dressing", "quantity": 15, "unit": "g"}, {"name": "feta cheese", "quantity": 20, "unit": "g"}]}

Example 3 — Protein plate with grains:
{"items": [{"name": "grilled salmon", "quantity": 140, "unit": "g"}, {"name": "brown rice", "quantity": 130, "unit": "g"}, {"name": "steamed spinach", "quantity": 70, "unit": "g"}, {"name": "lemon juice", "quantity": 5, "unit": "g"}]}

Size reference guide:
- Full protein portion (meat/fish): 100-180g
- Grain or starch side: 100-200g
- Cooked vegetables: 50-120g
- Raw leafy greens: 30-80g
- Sauce or dressing: 5-20g
- Cheese or nuts: 15-30g
- A typical full meal plate weighs 250-550g total

Rules:
- List every distinct food item visible, including sauces, dressings, and garnishes
- Use specific food names (e.g. "black olive" not "olive", "chicken breast" not "chicken")
- Estimate ALL quantities in grams
- quantity must be a number greater than 0"""


def _groq_request(payload: dict, retries: int = 6) -> dict:
    
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


def _parse_vision_output(raw: str) -> dict:
    """
    Parse the vision model's text output into a structured items dict.
    Tries direct JSON parse first, then falls back to regex extraction.

    Raises:
        ValueError: If no valid food items can be parsed from the output
    """
    try:
        parsed = json.loads(raw)
        if parsed.get("items"):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if parsed.get("items"):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse food items from image. Try a clearer photo.")


def analyze_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Analyse a meal image and return identified food items with gram weights.
    """
    # Encode image to base64 for inline API transmission
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    data = _groq_request({
        "model": VISION_MODEL,
        "temperature": 0.2,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                },
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    })

    raw = data["choices"][0]["message"]["content"].strip()
    return _parse_vision_output(raw)