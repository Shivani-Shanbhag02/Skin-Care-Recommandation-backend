from typing import Any, Dict

SKIN_TYPES: Dict[int, str] = {
    0: "dry",
    1: "normal",
    2: "oily"
}

CONCERNS: Dict[int, str] = {
    0: "Redness",
    1: "dark spots",
    2: "pigmentation",
    3: "none" 
}

PRODUCTS: Dict[str, Any] = {
    "cleanser": {
        "oily": ["Salicylic Acid Cleanser", "Foaming Gel Cleanser"],
        "dry": ["Hydrating Cream Cleanser"],
        "normal": ["Mild Cleanser"]
    },
    "serum_am": {
        "Redness": ["Soothe Centella Asiatica Serum"],
        "pigmentation": ["Vitamin C Serum"],
        "dark spots": ["Alpha Arbutin Serum"],
        "none": ["Hyaluronic Acid Serum"]  # Pure hydration for maintenance
    },
    "serum_pm": {
        "Redness": ["Niacinamide & Azelaic Acid Serum"],
        "pigmentation": ["Vitamin C Serum"],
        "dark spots": ["Alpha Arbutin Serum"],
        "none": ["Gentle Niacinamide Serum"] # Strengthens skin barrier without aggressive actives
    },
    "moisturizer": {
        "oily": ["Oil-free Gel Moisturizer"],
        "dry": ["Ceramide Moisturizer"],
        "normal": ["Balanced Moisturizer"]
    },
    "sunscreen": ["SPF 50 Gel Sunscreen", "Broad Spectrum Sunscreen"]
}

def generate_routine_and_products(skin_type_idx: int, concern_idx: int) -> Dict[str, Any]:
    skin_type = SKIN_TYPES.get(skin_type_idx, "normal")
    
    # Use .get() with a default fallback to "none" if the index is out of bounds or missing
    concern = CONCERNS.get(concern_idx, "none")

    cleanser = PRODUCTS["cleanser"].get(skin_type, PRODUCTS["cleanser"]["normal"])
    moisturizer = PRODUCTS["moisturizer"].get(skin_type, PRODUCTS["moisturizer"]["normal"])
    
    # Fetch serums. If the concern isn't found, fallback to the "none" maintenance routine
    serum_am = PRODUCTS["serum_am"].get(concern, PRODUCTS["serum_am"]["none"])
    serum_pm = PRODUCTS["serum_pm"].get(concern, PRODUCTS["serum_pm"]["none"])

    # If there is no concern, we customize the step naming to look cleaner on the UI
    serum_step_am = "Hydrating Serum (AM)" if concern == "none" else "Targeted Serum (AM)"
    serum_step_pm = "Barrier Support Serum (PM)" if concern == "none" else "Treatment Serum (PM)"

    routine = {
        "morning": [
            {"step": "Cleanser", "products": cleanser},
            {"step": serum_step_am, "products": serum_am},
            {"step": "Moisturizer", "products": moisturizer},
            {"step": "Sunscreen", "products": PRODUCTS["sunscreen"]}
        ],
        "night": [
            {"step": "Cleanser", "products": cleanser},
            {"step": serum_step_pm, "products": serum_pm},
            {"step": "Moisturizer", "products": moisturizer}
        ]
    }

    return {
        "skin_type": skin_type.capitalize(),
        "concern": "No Major Concerns (Maintenance Mode)" if concern == "none" else concern,
        "routine": routine
    }