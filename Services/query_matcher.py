import re

# Synonym mapping to translate user-friendly queries to COCO class names
SYNONYMS = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "telephone": "cell phone",
    "bike": "bicycle",
    "cycle": "bicycle",
    "bicycle": "bicycle",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "tv": "tv",
    "television": "tv",
    "screen": "tv",
    "monitor": "tv",
    "couch": "couch",
    "sofa": "couch",
    "laptop": "laptop",
    "computer": "laptop",
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "dining table": "dining table",
    "table": "dining table",
    "desk": "dining table",
    "pottedplant": "potted plant",
    "plant": "potted plant",
    "doggy": "dog",
    "puppy": "dog",
    "kitty": "cat",
    "bag": "backpack",
    "backpack": "backpack",
    "glass": "wine glass",
    "wineglass": "wine glass",
    "hair dryer": "hair drier"
}

# Irregular plural nouns mapping
IRREGULAR_PLURALS = {
    "people": "person",
    "children": "child",
    "mice": "mouse"
}

def normalize_word(word):
    """Normalize word to lowercase, singular form, and resolve synonyms"""
    word = word.strip().lower()
    if not word:
        return ""

    # 1. Handle irregular plurals
    if word in IRREGULAR_PLURALS:
        word = IRREGULAR_PLURALS[word]
    else:
        # 2. General singularization rules (skip exceptions like 'bus', 'skis', 'scissors')
        if word not in ["bus", "scissors", "skis", "glass"]:
            if word.endswith("ves"):
                word = word[:-3] + "f"
            elif word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("es") and len(word) > 4:
                # e.g., benches -> bench, boxes -> box, glasses -> glass
                # If removing 'es' results in a word ending with s/ch/sh/x/z, strip 'es', otherwise 's'
                if re.search(r'(s|ch|sh|x|z)$', word[:-2]):
                    word = word[:-2]
                else:
                    word = word[:-1]
            elif word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]

    # 3. Resolve synonyms
    if word in SYNONYMS:
        word = SYNONYMS[word]

    return word

def parse_queries(query_string):
    """Parse comma-separated query string into a list of normalized target strings"""
    if not query_string:
        return []
    parts = query_string.split(",")
    normalized = []
    for part in parts:
        clean = normalize_word(part)
        if clean:
            normalized.append(clean)
    return normalized

def match_class(class_name, targets):
    """Check if detected class name matches any of the target classes (exact or substring)"""
    if not targets:
        return True # Empty targets means no filter
        
    cls_normalized = normalize_word(class_name)
    
    # Try exact match first
    if any(cls_normalized == target for target in targets):
        return True
        
    # Try substring match fallback
    if any(target in cls_normalized or cls_normalized in target for target in targets):
        return True
        
    return False
