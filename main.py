import os
import re
import json
from pathlib import Path

import requests
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
DATABASE = os.environ.get("DATABASE")

client = OpenAI(api_key=OPENAI_API_KEY)

client_mongo = None

def get_client_mongo():
    global client_mongo
    if client_mongo is None:
        client_mongo = MongoClient(MONGODB_URI)
    return client_mongo

def safe_parse_list(raw):
    raw = raw.strip()

    # Remove outer single or double quotes
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()

    # Remove ```json or ``` markers
    raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()

    # Remove trailing commas before } or ]
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    return json.loads(raw)


def extract_text(response):
    txts = []
    for res in response['responses']:
        if 'fullTextAnnotation' in res:
            text = res['fullTextAnnotation']['text'].lower()
            if "facts" in text:
                txts.append(res['fullTextAnnotation']['text'])
    return txts


def texts_from_image_url(url):
    request_body = {"requests": [{"image": {"source": {"imageUri": url}}, "features": [{"type": "TEXT_DETECTION"}]}]}
    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(request_body),
        timeout=60
    )
    tx = extract_text(response.json())
    return tx


def get_functional_ingredients(mess):
    text = mess.get('text', '')
    image_url = mess.get('image_url', '')

    if not text and not image_url:
        return []
    
    if text:
        ingredients_text = text
    elif image_url:
        ingredients_text = texts_from_image_url(image_url)
    else:
        return []

    system_prompt = """

**Task:**  
Extract a unique, normalized list of functional ingredient names from the provided text. For each, include any associated dose information if available.

---

### Instructions

**1. Identify and Normalize Functional Ingredients**
   - Scan the text for all ingredients that provide nutritional function or benefit.
   - For each ingredient:
     - Normalize the name:
       - Remove descriptive words, percentages, adjectives, and modifiers (e.g. "natural", "organic", "enhanced").
       - If the ingredient includes an "L-" prefix (e.g. "L-Citrulline") and the underlying ingredient is the same with or without it, omit the prefix unless a branded ingredient is specified.
       - Use title case (e.g. "Citrulline Malate").
     - If a branded ingredient name is supplied (e.g., “Cognizin Citicoline”), use its simplest distinct form as `branded_name`. If there is no branded name, set `branded_name: false`.
       - If a branded name is present, also add it to the `branded_names` array.
   - Ignore ingredients that are specifically stated to be absent (e.g., “Caffeine Free”, “Contains No Stevia”).

**2. Extract Dose Information**
   - For each ingredient, check if the dose (quantity and unit, such as mg, g) is specified.
   - If present, record the dose as a number and unit (as shown in text).
   - If dose or unit is not mentioned, set both fields to `false`.

**3. Merge Duplicates**
   - Ensure the final list contains no duplicate or near-duplicate ingredient entries.
   - Apply normalization rules consistently for merging.

**4. Output JSON Structure**
   - The output must strictly follow this JSON format:
     ```
     {
         "functional_ingredients_with_dose": [
             {"name": ..., "branded_name": ..., "dose": ..., "unit": ...},
             ...
         ],
         "branded_names": [
             ...
         ]
     }
     ```
     - Use `false` for `"branded_name"` where there is no branded name.
     - If no branded ingredients are found, `"branded_names"` should be an empty array.
     - Do not include any keys or arrays beyond those specified.

---

**Example Output:**
```
{
    "functional_ingredients_with_dose": [
        {"name": "Caffeine", "branded_name": "RocketCaf", "dose": 500, "unit": "mg"},
        {"name": "Citrulline", "branded_name": false, "dose": 200, "unit": "mg"}
    ],
    "branded_names": ["RocketCaf"]
}
```

---

**Reminders:**  
- Merge duplicate and near-duplicate ingredient names using normalization.
- Do not list any ingredients that are explicitly omitted from the product.
- Adhere strictly to the specified output format.

    """

    result = client.chat.completions.create(
        model="gpt-6-luna",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ingredients_text}
        ]
    )
    json_string = result.choices[0].message.content
    parsed_data = json.loads(json_string)

    if "error" in parsed_data:
      raise Exception(f"OpenAI API error: {parsed_data['error']['message']}")
    
    return parsed_data


def get_gpt_product(raw_amz, target_brand):
    prompt_text = """
**You are a JSON generator. Your task is to output a single-line JSON object describing a dietary supplement or functional food/beverage. Carefully follow the steps and rules below:**

---

### 1. Extract & Include Core Product Fields

From the provided data, extract and populate the following top-level fields:
- `name`: string  
- `description`: string  
- `brand`: string
- `price`: string  
- `date_first_available`: string (use "Unavailable" if this information is missing)
- `rating`: number (e.g., 4.1)  
- `number_of_ratings`: number (e.g., 155)  
- `top_reviews`: array  
- `asin`: string (or "Unavailable" if not present)

---

### 2. Product Format

Determine the product’s format using this list:
```
["Tablets", "Capsules", "Dissolving Strips", "Softgels", "Powders", "Gummies", "Liquids", "Chewables", "Lozenges", "Effervescent Tablets", "Patches", "Tinctures", "Sprays", "Bars"]
```
If the format cannot be determined, set `format`: false. 

NOTE: "Oral Strips" may also be referred to as "Oral Strips". 

---

### 3. Category & Tags

Assign one best-fit category and relevant tags from the dictionary below:
```
categories_with_tags = {
  "Weight Management": [
    "Weight Management",
    "Metabolic Rate Enhancement",
    "Fat Oxidation",
    "Appetite Control",
    "Thermogenesis",
    "Portion Control"
  ],
  "Digestive & Gut Health": [
    "Digestive Comfort",
    "Bloating Relief",
    "Bowel Regularity",
    "Gut Flora Balance",
    "Prebiotic Support",
    "Probiotic Support",
    "Nutrient Absorption",
    "Colon Health",
    "Microbiome Diversity"
  ],
  "Energy & Performance": [
    "Natural Energy Boost",
    "Fatigue Reduction",
    "Mental Stamina",
    "Physical Stamina",
    "Electrolyte Balance",
    "Rehydration",
    "Strength Enhancement",
    "Power Output",
    "Agility Enhancement",
    "Endurance Enhancement",
    "VO2 Max Support",
    "Performance Hydration"
  ],
  "Joint & Mobility Support": [
    "Joint Flexibility",
    "Joint Comfort",
    "Cartilage Support",
    "Mobility Support",
    "Flexibility Support"
  ],
  "Cognitive Health": [
    "Memory Support",
    "Focus Enhancement",
    "Cognitive Clarity",
    "Brain Health",
    "Neuroprotection",
    "Brain Fog Reduction",
    "Learning Support",
    "Cognitive Development"
  ],
  "Hormonal Health": [
    "Hormone Balance",
    "Testosterone Support",
    "Cortisol Regulation"
  ],
  "Muscle & Strength": [
    "Muscle Recovery",
    "Strength Enhancement",
    "Power Output"
  ],
  "Immune Health": [
    "Immune System Support",
    "Immune Defense",
    "Cellular Immunity",
    "Cold & Flu Protection",
    "Allergy Relief"
  ],
  "Beauty & Personal Care": [
    "Radiant Skin",
    "Anti-Aging Support",
    "Skin Hydration",
    "Collagen Production",
    "Wrinkle Reduction",
    "Even Skin Tone",
    "Hair Growth",
    "Hair Strength",
    "Nail Strength",
    "Fresh Breath",
    "Oral Microbiome Balance"
  ],
  "Men's Wellness": [
    "Prostate Health",
    "Male Vitality",
    "Testosterone Support",
    "Libido Support"
  ],
  "Women's Wellness": [
    "Hormone Balance",
    "Menstrual Comfort",
    "Menopause Support",
    "Breast Health",
    "Urinary Tract Health",
    "Vaginal Health"
  ],
  "Maternal & Prenatal Health": [
    "Prenatal Nutrition",
    "Fetal Development Support",
    "Morning Sickness Relief",
    "Lactation Support",
    "Postnatal Recovery"
  ],
  "Detoxification & Organ Support": [
    "Liver Health",
    "Detoxification Support",
    "Kidney Health",
    "Toxin Elimination",
    "Cellular Renewal"
  ],
  "Sleep & Stress Support": [
    "Sleep Quality",
    "Restful Sleep",
    "Circadian Rhythm Support",
    "Mood Support",
    "Stress Relief",
    "Anxiety Relief",
    "Emotional Balance"
  ],
  "Bone & Skeletal Health": [
    "Bone Strength",
    "Bone Density",
    "Calcium Absorption",
    "Osteoporosis Prevention",
    "Fracture Resistance"
  ],
  "Cardiovascular Health": [
    "Cholesterol Balance",
    "Blood Pressure Support",
    "Circulation Support",
    "Heart Function",
    "Vascular Health",
    "Homocysteine Regulation"
  ],
  "Metabolic Health": [
    "Blood Sugar Balance",
    "Insulin Sensitivity"
  ],
  "Aging & Longevity": [
    "Anti-Aging Support",
    "Cellular Renewal",
    "DNA Protection",
    "Longevity Support"
  ],
  "Vision & Eye Health": [
    "Vision Clarity",
    "Macular Health",
    "Eye Strain Relief",
    "Night Vision Support",
    "Dry Eye Relief"
  ],
  "Respiratory Health": [
    "Sinus Health",
    "Lung Function Support",
    "Breathing Ease",
    "Mucus Clearance"
  ]
}
```
- `category`: One key from `categories_with_tags` (choose the closest match, or use "Other" if none fit).
- `tags`: Array of one or more relevant tags from the selected category.

---

### 4. Claims Extraction

Identify all claims in the product name, description, or other relevant text. For each claim, add an object in the `claims` array with:
- `verbatim`: Exact claim text as stated.
- `category`: A relevant category from `categories_with_tags` keys that best fits this claim.
- `tags`: All matching tags from the corresponding category values of `categories_with_tags`.
- `supported_by`: Array of ingredient names from the claim text that genuinely substantiate the claim (empty array if none).
---

### 5. Output Requirements

- Produce a valid JSON object, **all on a single line**.
- Do **not** include any extra formatting, markdown, or line breaks.

---

###  6. If the "target_brand" provided in your input is specified (i.e., not 'false') and matches the brand detected in the product, make sure the "brand" field in your output exactly matches the capitalization, spacing, and formatting of the input "target_brand".

---

### 7. Example Output (single-line):

```
{"name":"Example Product","description":"A powerful cognitive supplement...","brand":"Example Brand","price":"$27.94","date_first_available":"February 3, 2007","rating":4.1,"number_of_ratings":155,"top_reviews":["Great value!","Really works."],"asin":"B001CN08OW","format":"Capsules","category":"Cognitive Health","tags":["Memory Improvement","Focus"],"claims":[{"verbatim":"Helps You Think More Clearly","category":"Cognitive Health","tags":["Memory Improvement","Focus","Neuroprotection"],"supported_by":["Caffeine","Citrulline"]}]}
```

---

**Key reminders:**
- Every field must be present and correctly filled or set to the default/fallback value as required.
- If information is unavailable, use the specified fallback (e.g., `"Unavailable"`, `false`, or empty array).
- Follow the structure, naming, and format **exactly**.
- Do not invent new Tags and Categories. Draw exclusively from `categories_with_tags`


    """
    
    payload = {}
    payload['raw_product_data'] = raw_amz
    payload['target_brand'] = target_brand
    payload = json.dumps(payload)
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[
            {"role": "system", "content": prompt_text}, 
            {"role": "user", "content": payload}
        ],
        response_format={"type": "json_object"}
    )
    json_string = response.choices[0].message.content
    parsed_data = json.loads(json_string)
    return parsed_data


def fetch_brand_products(brand_name: str):
    prompt = f"""
You are a data collection agent.

Use web search to find products that are explicitly sold by the brand: "{brand_name}".

Use ONLY:
- Official brand website
- Official brand store on Amazon

Return ONLY a valid JSON array of strings.

Rules:
- No duplicates
- No commentary, no notes

Return format:
[
  "Product 1 name",
  "Product 2 name",
  "Product 3 name"
]

If none found, return [].

**Key reminders:**
- Follow the the format exactly and return only a json object.
"""

    response = client.chat.completions.create(
        model="gpt-5-search-api",
        web_search_options={},
        messages=[{"role": "system", "content": prompt}]
    )

    raw = response.choices[0].message.content

    return safe_parse_list(raw)


def fetch_product_details(brand_name: str, product_name: str):
    prompt = f'''
You are an expert data extraction agent.

Using web search ONLY, find verifiable information for the following product:

Brand: "{brand_name}"
Product name: "{product_name}"

Minor variations in wording, abbreviations, hyphenation, or order (e.g., "caps" vs "capsules", "20bn" vs "20 billion") should still be treated as the same product.
Proceed if the product is sold by or clearly associated with the brand "{brand_name}" on any trusted source, including the official brand website.
Proceed if the product is a dietary supplement, functional food/beverage, or cosmetic. For example:
  1. Dietary Supplement:
    • Products intended to supplement the diet with vitamins, minerals, herbs, amino acids, probiotics, or other bioactive ingredients.
    • Common forms: capsules, tablets, powders, liquids, gummies.
    • Examples: Creatine capsules, Berberine pills, Fiber powders, Probiotic capsules.
    • Typical language: “Support”, “Boost”, “Daily”, “+”, “Complex”, “Formula”.

  2. Functional Food / Beverage:
    • Products consumed like food or drink but provide additional health benefits.
    • Common forms: drinks, bars, powders meant to mix into shakes, fortified foods.
    • Examples: Protein bars with added vitamins, fortified beverages, smoothie mixes.
    • Language hints: “Pack”, “Drink mix”, “Shake”, “Bar”, “Functional beverage”.

  3. Cosmetic:
    • Products intended to cleanse, protect, beautify, or enhance the appearance of skin, hair, nails, lips, or body (not primarily ingested).
    • Common forms: creams, serums, lotions, oils, masks, scrubs, balms, shampoos, conditioners.
    • Examples: Face serums, collagen creams, lip balms, hair growth oils, body lotions.
    • Language hints: “Glow”, “Brightening”, “Anti-aging”, “Hydrating”, “Repair”, “Firming”, “Skin”, “Hair”, “Beauty”.

Your goal is to gather as much accurate detail as possible from trusted, verifiable sources such as:
- Official brand website
- Official brand store on Amazon

If a value cannot be verified, use false.
If no trustworthy source mentions this product in relation to the brand, return an empty array: []

Return ONE valid JSON object using this exact schema:

Field descriptions (outside the JSON)
	•	name → Full product name or false
	•	brand → Use "{brand_name}" exactly
	•	description → Detailed description (features, benefits, form, usage), or false
	•	price → Current retail price with currency symbol or false
	•	date_first_available → Format: YYYY-MM-DD or false
	•	asin → ASIN or false
	•	format → As stated in the web search result or false.
	•	category → Product category/type or false
	•	ingredients → List of ingredients or []
	•	directions → How to use the product or false
	•	warnings → Safety warnings or false
	•	supplement_label_text → Key label text lines or []
  • nutritional_facts → Parsed supplement facts / amount per serving (from the Supplement Facts panel for supplements or Nutrition Facts panel for foods), image url if the image is available. Otherwise, set text and image_url to blank('').
	•	product_specifications.net_weight → e.g. "500g", "60 capsules" or false
	•	product_specifications.serving_size → e.g. "1 scoop", "2 capsules" or false
	•	product_specifications.servings_per_container → Number or false
	•	tags → Relevant keywords or []

{{
  "name": "",
  "brand": "{brand_name}",
  "description": "",
  "price": "",
  "date_first_available": "",
  "asin": "",
  "format": "",
  "category": "",
  "ingredients": [],
  "directions": "",
  "warnings": "",
  "supplement_label_text": "",
  "nutritional_facts": {{
    "text": "Per serving: 500mg Vitamin C, 10mg Zinc...",
    "image_url": "https://example.com/image.jpg"
  }}
  "product_specifications": {{
    "net_weight": "",
    "serving_size": "",
    "servings_per_container": ""
  }},
  "tags": []
}}

FINAL OUTPUT RULES
- No markdown
- No explanations
- No comments
- No backticks
- No headings
- No trailing text
- No surrounding quotes outside the JSON
- All keys must match exactly as provided
- All string values must use double quotes
- Use false, [] or "" when data is unavailable
- NEVER include links in parentheses
- NEVER include new fields
- NEVER change field names
- NEVER add text outside the JSON object
'''

    response = client.chat.completions.create(
        model="gpt-5-search-api",
        web_search_options={},
        messages=[{"role": "system", "content": prompt}]
    )

    raw = response.choices[0].message.content

    return safe_parse_list(raw)


# ----- Main -----
mongo = get_client_mongo()

db = mongo[DATABASE]
products_collection = db["test_products"]

brand = "Happy Being"
products = fetch_brand_products(brand)

print(f"Found {len(products)} products for {brand}...")

for product in products:
    print(f'Processing {product}...')

    data = fetch_product_details(brand, product)
    if not data:
        print(f'{product} data is empty. Skipped!')
        continue
    
    nutritional_facts = data.get('nutritional_facts')
    functional_ingredients_info = get_functional_ingredients(nutritional_facts)

    better = get_gpt_product(data, brand)
    
    if isinstance(functional_ingredients_info, dict):
      better["v3_functional_ingredients_info"] = functional_ingredients_info.get("functional_ingredients_with_dose", [])
      better["branded_names"] = functional_ingredients_info.get("branded_names", [])
    else:
        better["v3_functional_ingredients_info"] = []
        better["branded_names"] = []

    better["asin"] = data.get("asin", '')
    better['supplement_label_text'] = data.get('supplement_label_text', '')

    folder = Path(brand)
    folder.mkdir(parents=True, exist_ok=True)

    file_name = product.replace('\\', ',')
    file_path = "Output" / folder / f"{file_name}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(better, f, indent=4)

    if better.get("category") != "Other":
        if len(better["v3_functional_ingredients_info"]) > 0:
            products_collection.insert_one(better)

print("Done!")
