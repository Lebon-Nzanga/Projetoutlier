import re, json
def strip_json(txt):
    m = re.search(r'(\{.*?\}|\$\$\n.*?\n\$\$)', txt, flags=re.S)
    if not m:
        raise ValueError("no JSON found")
    return m.group(1)
def require_json(txt):
    return json.loads(strip_json(txt))  