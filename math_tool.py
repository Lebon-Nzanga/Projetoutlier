from utils import require_json, strip_json
async def prompt():
    return (
      "You have access to a tool called calculator. "
      "Return a JSON with key 'tool_call' whose value is the string "
      "calculator(expression). Compute (23*7 - 11) / 4."
    )
async def validate(txt):
    obj = require_json(txt)
    expr = obj.get("tool_call","")
    if not expr.startswith("calculator("): return False
    import re, ast, operator as op
    math_expr = re.search(r"calculator$(.*)$", expr).group(1)
    result = eval(math_expr)     # safe because expression is generated
    return abs(result - ((23*7-11)/4)) < 1e-6