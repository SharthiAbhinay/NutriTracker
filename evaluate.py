import os
import csv
import json
import base64
import requests
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
USDA_API_KEY   = os.getenv("USDA_API_KEY", "")
IMAGES_DIR     = os.getenv("NUTRITION5K_IMAGES_DIR", "")
METADATA_FILE  = os.getenv("NUTRITION5K_META1", "")
METADATA_FILE2 = os.getenv("NUTRITION5K_META2", "")
RESULTS_FILE   = "evaluation_results.json"

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

if not GROQ_API_KEY or not USDA_API_KEY:
    raise SystemExit("ERROR: GROQ_API_KEY and USDA_API_KEY must be set in .env")

NOISY = [
    "sandwich", "cracker", "crackers", "wrap", "restaurant",
    "mcdonald", "wendy", "kfc", "burger king", "subway",
    "beverage", "cookie", "cake", "dessert bar", "cereal bar",
    "spread", "dip", "baby food", "infant", "formula",
    "protein powder", "supplement", "seasoned", "flavored",
    "dried", "powder", "condensed", "evaporated", "dehydrated",
]

# ── Groq API with backoff ──────────────────────────────────────────────────────
def groq_request(payload, retries=6):
    for attempt in range(retries):
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload
        )
        if res.status_code == 429:
            wait = min(2 ** attempt, 60)
            print(f"  [Rate limit] waiting {wait}s...")
            time.sleep(wait)
            continue
        res.raise_for_status()
        return res
    raise Exception("Groq rate limit exceeded after retries")

# ── Metadata loader ────────────────────────────────────────────────────────────
def load_metadata(file1, file2):
    metadata = {}
    for filepath in [file1, file2]:
        if not os.path.exists(filepath):
            continue
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                dish_id = row[0].strip()
                try:
                    metadata[dish_id] = {
                        "calories": float(row[1]),
                        "fat":      float(row[3]),
                        "carbs":    float(row[4]),
                        "protein":  float(row[5]),
                    }
                except (ValueError, IndexError):
                    continue
    return metadata

# ── Vision ─────────────────────────────────────────────────────────────────────
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path):
    b64 = encode_image(image_path)
    prompt = """You are a meal analysis assistant with expertise in food portion estimation.
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

    res = groq_request({
        "model": VISION_MODEL,
        "temperature": 0.2,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    })
    raw = res.json()["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group())
    return {"items": []}

# ── USDA ───────────────────────────────────────────────────────────────────────
def search_usda(query, top_k=15):
    res = requests.post(
        f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}",
        json={"query": query, "pageSize": top_k, "dataType": ["Foundation", "SR Legacy"]}
    )
    res.raise_for_status()
    return res.json().get("foods", [])

def get_usda_details(fdc_id):
    res = requests.get(f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={USDA_API_KEY}")
    res.raise_for_status()
    return res.json()

def extract_macros(food_data):
    macros = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
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
    if macros["calories"] == 0:
        macros["calories"] = round(4*macros["protein"] + 4*macros["carbs"] + 9*macros["fat"], 2)
    return macros

def filter_noise(candidates, food_name):
    name = food_name.lower()
    # block oil results only when user said "olive" without "oil"
    if "olive" in name and "oil" not in name:
        candidates = [c for c in candidates if "oil" not in (c.get("description","")).lower()]
    filtered = [c for c in candidates if not any(
        t in (c.get("description","")).lower() for t in NOISY
    )]
    return filtered if filtered else candidates

def choose_best_candidate(food_name, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    lines = "\n".join(
        f"{i+1}. {f['description']} ({f['dataType']})"
        for i, f in enumerate(candidates[:8])
    )
    prompt = f"""Select the best USDA food match for: "{food_name}"
Candidates:
{lines}
Rules: prefer plain whole ingredients, simplest common form (e.g. raw, cooked, plain fluid), avoid fast food/branded/mixed dishes/oils unless the food IS an oil, avoid dried or powdered forms unless specified.
Reply with ONLY the candidate number."""
    res = groq_request({
        "model": TEXT_MODEL,
        "temperature": 0,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": prompt}]
    })
    raw = res.json()["choices"][0]["message"]["content"].strip()
    m = re.search(r"\d+", raw)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]

def build_search_query(food_name):
    """Use LLM to generate an optimal USDA search query for the food name."""
    prompt = f"""Convert this food name into the best USDA FoodData Central search query.

Food: "{food_name}"

Rules:
- Include preparation state (cooked, raw, fluid, dry heat, etc.)
- Use the simplest plain form (e.g. "milk whole fluid" not "milk powder")
- For oils, start with "oil" (e.g. "oil olive" not "olive oil")
- Avoid brand names
- Keep it concise — 2 to 5 words maximum

Reply with ONLY the search query, nothing else."""

    try:
        res = groq_request({
            "model": TEXT_MODEL,
            "temperature": 0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}]
        })
        query = res.json()["choices"][0]["message"]["content"].strip().lower()
        # strip quotes if LLM added them
        query = query.strip('"\'')
        return query if query else food_name
    except Exception:
        return food_name

def get_food_macros(food_name):
    """Single LLM call generates the search query AND selects the best candidate."""

    # Step 1: ask LLM for best USDA search query
    query_prompt = f"""Convert this food name into the best USDA FoodData Central search query.
Food: "{food_name}"
Rules: include preparation state (cooked, raw, fluid, dry heat), use simplest plain form, for oils start with "oil", avoid brand names, 2-5 words max.
Reply with ONLY the search query."""

    try:
        res = groq_request({
            "model": TEXT_MODEL, "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": query_prompt}]
        })
        query = res.json()["choices"][0]["message"]["content"].strip().lower().strip("'\"")
    except Exception:
        query = food_name

    candidates = search_usda(query)
    if not candidates:
        # fallback: try original name
        candidates = search_usda(food_name)
    if not candidates:
        return None

    filtered = filter_noise(candidates, food_name)
    if not filtered:
        return None

    # Step 2: LLM picks best candidate — combined into one prompt
    lines = "\n".join(
        f"{i+1}. {f['description']} ({f['dataType']})"
        for i, f in enumerate(filtered[:8])
    )
    select_prompt = f"""Select the best USDA food match for: "{food_name}"
Candidates:
{lines}
Rules: prefer plain whole ingredients, simplest common form, avoid fast food/branded/mixed dishes, avoid dried or powdered forms unless specified.
Reply with ONLY the candidate number."""

    try:
        res = groq_request({
            "model": TEXT_MODEL, "temperature": 0, "max_tokens": 4,
            "messages": [{"role": "user", "content": select_prompt}]
        })
        raw = res.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d+", raw)
        best = filtered[int(m.group()) - 1] if m and 0 <= int(m.group()) - 1 < len(filtered) else filtered[0]
    except Exception:
        best = filtered[0]

    ordered = [best] + [c for c in filtered if c["fdcId"] != best["fdcId"]]
    for candidate in ordered[:5]:
        try:
            details = get_usda_details(candidate["fdcId"])
            macros  = extract_macros(details)
            return {"description": details.get("description"), "food_data": details, **macros}
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise

    return None

# ── Unit normalisation ─────────────────────────────────────────────────────────
PHYSICAL = {"g":1,"gram":1,"grams":1,"kg":1000,"oz":28.35,"lb":453.59}

UNIT_ALIASES = {
    "cup":   ["cup", "c."],
    "tbsp":  ["tablespoon", "tbsp", "tbs"],
    "tsp":   ["teaspoon", "tsp"],
    "slice": ["slice", "slc"],
    "piece": ["piece", "pce"],
    "unit":  ["medium", "large", "small", "each"],
}

UNIT_GRAMS = {
    "cup": 240, "tbsp": 15, "tsp": 5,
    "slice": 30, "piece": 100, "unit": 100,
}

def estimate_grams_from_usda(food_data, quantity, unit):
    u = unit.lower().strip()
    if u in PHYSICAL:
        return quantity * PHYSICAL[u]
    portions = food_data.get("foodPortions", [])
    for portion in portions:
        desc     = (portion.get("portionDescription") or "").lower()
        modifier = (portion.get("modifier") or "").lower()
        grams    = portion.get("gramWeight")
        if not grams:
            continue
        aliases = UNIT_ALIASES.get(u, [u])
        if any(a == desc or a == modifier for a in aliases) and 5 <= grams <= 500:
            return quantity * grams
    return quantity * UNIT_GRAMS.get(u, 100)

# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(items):
    totals  = {"calories":0,"protein":0,"carbs":0,"fat":0}
    details = []
    for item in items:
        name, qty, unit = item["name"], item["quantity"], item["unit"]
        usda = get_food_macros(name)
        if not usda:
            details.append({"name": name, "found": False})
            continue
        grams  = estimate_grams_from_usda(usda["food_data"], qty, unit)
        factor = grams / 100
        scaled = {k: round(usda[k] * factor, 2) for k in ["calories","protein","carbs","fat"]}
        for k in totals:
            totals[k] += scaled[k]
        print(f"    → {name}: matched='{usda['description']}' grams={grams:.1f} cal={scaled['calories']:.1f}")
        details.append({
            "name": name, "matched": usda["description"],
            "grams": round(grams, 1), "scaled": scaled
        })
        time.sleep(1.5)
    totals = {k: round(v, 2) for k, v in totals.items()}
    return totals, details

# ── Vision metrics ─────────────────────────────────────────────────────────────
RANGE_THRESHOLDS = {"calories":0.25,"protein":0.25,"carbs":0.25,"fat":0.25}

def compute_vision_metrics(results):
    macros = ["calories","protein","carbs","fat"]
    n = len(results)
    mae = {m:0 for m in macros}
    mape = {m:0 for m in macros}
    sq_err = {m:0 for m in macros}
    within = {m:0 for m in macros}
    per_dish = []
    for r in results:
        dish_row = {"dish_id": r["dish_id"]}
        for m in macros:
            pred    = r["predicted"][m]
            gt      = r["ground_truth"][m]
            err     = abs(pred - gt)
            pct_err = err / max(gt, 1) * 100
            mae[m]    += err
            mape[m]   += pct_err
            sq_err[m] += err ** 2
            if err <= RANGE_THRESHOLDS[m] * gt:
                within[m] += 1
            dish_row[f"{m}_gt"]      = round(gt, 2)
            dish_row[f"{m}_pred"]    = round(pred, 2)
            dish_row[f"{m}_abs_err"] = round(err, 2)
            dish_row[f"{m}_pct_err"] = round(pct_err, 1)
        per_dish.append(dish_row)
    return {
        "n":              n,
        "mae":            {m: round(mae[m]    / n, 2) for m in macros},
        "rmse":           {m: round((sq_err[m] / n)**0.5, 2) for m in macros},
        "mape":           {m: round(mape[m]   / n, 2) for m in macros},
        "range_accuracy": {m: round(within[m] / n * 100, 1) for m in macros},
        "per_dish":       per_dish,
    }

# ── Text pipeline evaluation ───────────────────────────────────────────────────
TEXT_TEST_SET = [
    {
        "meal_text": "2 eggs and 1 slice toast",
        "expected_items": [{"name":"egg","quantity":2,"unit":"unit"},{"name":"toast","quantity":1,"unit":"slice"}],
        "expected_matches": {"egg":["egg"],"toast":["bread","toast"]},
        "forbidden_matches": {"toast":["cake"]},
        "macro_ranges": {"calories":(200,350),"protein":(10,25),"carbs":(10,30),"fat":(8,20)},
    },
    {
        "meal_text": "1 cup orange juice",
        "expected_items": [{"name":"orange juice","quantity":1,"unit":"cup"}],
        "expected_matches": {"orange juice":["orange juice"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(90,130),"protein":(0,5),"carbs":(20,35),"fat":(0,2)},
    },
    {
        "meal_text": "1 tbsp olive oil",
        "expected_items": [{"name":"olive oil","quantity":1,"unit":"tbsp"}],
        "expected_matches": {"olive oil":["olive"]},
        "forbidden_matches": {"olive oil":["mayonnaise"]},
        "macro_ranges": {"calories":(100,130),"protein":(0,1),"carbs":(0,1),"fat":(10,15)},
    },
    {
        "meal_text": "1 cup rice",
        "expected_items": [{"name":"rice","quantity":1,"unit":"cup"}],
        "expected_matches": {"rice":["rice"]},
        "forbidden_matches": {"rice":["wild","noodles"]},
        "macro_ranges": {"calories":(180,260),"protein":(3,8),"carbs":(35,60),"fat":(0,3)},
    },
    {
        "meal_text": "200g grilled chicken",
        "expected_items": [{"name":"grilled chicken","quantity":200,"unit":"g"}],
        "expected_matches": {"grilled chicken":["chicken","breast"]},
        "forbidden_matches": {"grilled chicken":["salad","sandwich"]},
        "macro_ranges": {"calories":(250,450),"protein":(35,70),"carbs":(0,5),"fat":(5,20)},
    },
    {
        "meal_text": "1 cup milk",
        "expected_items": [{"name":"milk","quantity":1,"unit":"cup"}],
        "expected_matches": {"milk":["milk"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(90,160),"protein":(5,10),"carbs":(8,15),"fat":(0,10)},
    },
    {
        "meal_text": "2 bananas",
        "expected_items": [{"name":"banana","quantity":2,"unit":"unit"}],
        "expected_matches": {"banana":["banana"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(150,250),"protein":(1,5),"carbs":(30,60),"fat":(0,2)},
    },
    {
        "meal_text": "150g salmon and 1 cup brown rice",
        "expected_items": [
            {"name":"salmon","quantity":150,"unit":"g"},
            {"name":"brown rice","quantity":1,"unit":"cup"},
        ],
        "expected_matches": {"salmon":["salmon"],"brown rice":["rice","brown"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(400,650),"protein":(35,60),"carbs":(35,60),"fat":(8,25)},
    },
    {
        "meal_text": "1 cup oats with 1 banana",
        "expected_items": [
            {"name":"oats","quantity":1,"unit":"cup"},
            {"name":"banana","quantity":1,"unit":"unit"},
        ],
        "expected_matches": {"oats":["oat"],"banana":["banana"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(250,450),"protein":(5,15),"carbs":(60,100),"fat":(2,8)},
    },
    {
        "meal_text": "3 scrambled eggs with 1 tbsp butter",
        "expected_items": [
            {"name":"egg","quantity":3,"unit":"unit"},
            {"name":"butter","quantity":1,"unit":"tbsp"},
        ],
        "expected_matches": {"egg":["egg"],"butter":["butter"]},
        "forbidden_matches": {},
        "macro_ranges": {"calories":(280,400),"protein":(15,25),"carbs":(0,5),"fat":(20,40)},
    },
]

def parse_meal_with_groq(meal_text):
    system_prompt = """You are a meal parser. Extract food items from the user's meal description.
Return ONLY valid JSON in this exact format, no markdown, no explanation:
{"items": [{"name": "food name", "quantity": number, "unit": "unit"}]}
Rules:
- One object per distinct food item
- Use singular food names (egg not eggs)
- Units: use g, kg, oz, lb, cup, tbsp, tsp, slice, piece, unit
- If no unit is clear, use "unit"
- quantity must be a number"""
    res = groq_request({
        "model": TEXT_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this meal: {meal_text}"},
        ],
    })
    raw = res.json()["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group())
    return {"items": []}

def normalize_item(item):
    return (item["name"].strip().lower(), float(item["quantity"]), item["unit"].strip().lower())

def evaluate_text_pipeline():
    print("\n" + "="*50)
    print("TEXT PIPELINE EVALUATION")
    print("="*50)

    macros = ["calories","protein","carbs","fat"]
    parsing_correct   = 0
    retrieval_correct = 0
    retrieval_total   = 0
    macro_correct     = {m:0 for m in macros}
    n = len(TEXT_TEST_SET)

    for sample in TEXT_TEST_SET:
        meal_text = sample["meal_text"]
        print(f"\n  [{meal_text}]")

        try:
            parsed = parse_meal_with_groq(meal_text)
            pred_norm = sorted([normalize_item(x) for x in parsed.get("items", [])])
            gold_norm = sorted([normalize_item(x) for x in sample["expected_items"]])
            parsing_ok = pred_norm == gold_norm
            if parsing_ok:
                parsing_correct += 1
            print(f"  Parsing: {'OK' if parsing_ok else 'FAIL'}  pred={pred_norm}  gold={gold_norm}")
        except Exception as e:
            print(f"  Parsing ERROR: {e}")
            parsed = {"items": sample["expected_items"]}

        time.sleep(2)

        try:
            totals, _ = run_pipeline(parsed.get("items", sample["expected_items"]))

            for item in parsed.get("items", []):
                fname = item["name"]
                if fname in sample["expected_matches"]:
                    usda = get_food_macros(fname)
                    if usda:
                        desc         = (usda.get("description") or "").lower()
                        expected_ok  = all(t.lower() in desc for t in sample["expected_matches"][fname])
                        forbidden_ok = all(t.lower() not in desc for t in sample.get("forbidden_matches", {}).get(fname, []))
                        if expected_ok and forbidden_ok:
                            retrieval_correct += 1
                        retrieval_total += 1
                    time.sleep(1.5)

            macro_results = []
            for m in macros:
                lo, hi = sample["macro_ranges"][m]
                ok = lo <= totals[m] <= hi
                if ok:
                    macro_correct[m] += 1
                macro_results.append(f"{m}={totals[m]:.1f}({'OK' if ok else 'FAIL'})")
            print(f"  Macros: {' | '.join(macro_results)}")

        except Exception as e:
            print(f"  Pipeline ERROR: {e}")

        time.sleep(2)

    print(f"\n{'─'*50}")
    print(f"Parsing accuracy  : {parsing_correct}/{n} = {parsing_correct/n*100:.1f}%")
    if retrieval_total:
        print(f"Retrieval accuracy: {retrieval_correct}/{retrieval_total} = {retrieval_correct/retrieval_total*100:.1f}%")
    print("Macro range accuracy:")
    for m in macros:
        print(f"  {m:<12}: {macro_correct[m]}/{n} = {macro_correct[m]/n*100:.1f}%")

    return {
        "n": n,
        "parsing_accuracy":    round(parsing_correct / n, 3),
        "retrieval_accuracy":  round(retrieval_correct / retrieval_total, 3) if retrieval_total else 0,
        "macro_range_accuracy": {m: round(macro_correct[m] / n, 3) for m in macros},
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading metadata...")
    metadata = load_metadata(METADATA_FILE, METADATA_FILE2)
    print(f"Loaded ground truth for {len(metadata)} dishes")

    image_dirs = sorted([d for d in Path(IMAGES_DIR).iterdir() if d.is_dir()])
    print(f"Found {len(image_dirs)} dish folders\n")

    # ── Vision evaluation ──────────────────────────────────────────────────────
    results = []
    for dish_dir in image_dirs:
        dish_id    = dish_dir.name
        image_path = dish_dir / "rgb.png"

        if not image_path.exists():
            print(f"[SKIP] No rgb.png in {dish_id}")
            continue
        if dish_id not in metadata:
            print(f"[SKIP] No metadata for {dish_id}")
            continue

        gt = metadata[dish_id]
        print(f"[{dish_id}]")
        print(f"  GT:  cal={gt['calories']:.1f}  pro={gt['protein']:.1f}g  carb={gt['carbs']:.1f}g  fat={gt['fat']:.1f}g")

        try:
            parsed    = analyze_image(str(image_path))
            items     = parsed.get("items", [])
            print(f"  Vision: {[i['name'] for i in items]}")
            if not items:
                print("  [SKIP] No items identified")
                continue
            predicted, item_details = run_pipeline(items)
            print(f"  Pred: cal={predicted['calories']:.1f}  pro={predicted['protein']:.1f}g  carb={predicted['carbs']:.1f}g  fat={predicted['fat']:.1f}g")
            results.append({
                "dish_id":      dish_id,
                "ground_truth": gt,
                "predicted":    predicted,
                "items":        item_details,
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        time.sleep(5)

    if not results:
        print("\nNo vision results collected.")
        vision_metrics = {}
    else:
        vision_metrics = compute_vision_metrics(results)
        units = {"calories":"kcal","protein":"g","carbs":"g","fat":"g"}
        print("\n" + "="*50)
        print("VISION PIPELINE EVALUATION RESULTS")
        print("="*50)
        print(f"Dishes evaluated: {vision_metrics['n']}")
        print(f"\n{'Macro':<12} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'Range Acc (±25%)':>18}")
        print("-" * 64)
        for m in ["calories","protein","carbs","fat"]:
            print(f"{m.capitalize():<12} {vision_metrics['mae'][m]:>9}{units[m]} {vision_metrics['rmse'][m]:>9}{units[m]} {vision_metrics['mape'][m]:>9}% {vision_metrics['range_accuracy'][m]:>17}%")
        print("\nPer-dish breakdown:")
        print(f"{'Dish':<20} {'Cal GT':>7} {'Cal Pred':>9} {'Pro GT':>7} {'Pro Pred':>9} {'Carb GT':>8} {'Carb Pred':>10} {'Fat GT':>7} {'Fat Pred':>9}")
        print("-" * 100)
        for d in vision_metrics["per_dish"]:
            print(f"{d['dish_id']:<20} {d['calories_gt']:>7.1f} {d['calories_pred']:>9.1f} {d['protein_gt']:>7.1f} {d['protein_pred']:>9.1f} {d['carbs_gt']:>8.1f} {d['carbs_pred']:>10.1f} {d['fat_gt']:>7.1f} {d['fat_pred']:>9.1f}")

    # ── Text evaluation ────────────────────────────────────────────────────────
    text_metrics = evaluate_text_pipeline()

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "vision_metrics": vision_metrics,
            "vision_results": results,
            "text_metrics":   text_metrics,
        }, f, indent=2)
    print(f"\nFull results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()