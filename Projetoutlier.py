#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick-start LLM eval kit (OpenAI GPT-5 class, Anthropic Claude Sonnet 4.5 class,
Google Gemini 3 class, plus extra model IDs if you wish).
"""

import os, time, json, uuid, pathlib, base64, traceback, contextlib
from datetime import datetime
import pandas as pd
import yaml
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()  # pulls keys from .env if present

# Optional: load a rate card from `rate_card.yaml` if present
RATE_CARD = None
RATE_CARD_PATH = pathlib.Path("rate_card.yaml")
if RATE_CARD_PATH.exists():
    try:
        with RATE_CARD_PATH.open("r", encoding="utf-8") as fh:
            RATE_CARD = yaml.safe_load(fh)
        rprint(f"[green]Loaded rate_card.yaml ({RATE_CARD_PATH})[/]")
    except Exception as _e:
        rprint(f"[red]Failed to load rate_card.yaml:[/] {_e}")
else:
    rprint(f"[yellow]rate_card.yaml not found at {RATE_CARD_PATH}[/]")

###############################################################################
# 0. CONFIG ------------------------------------------------------------------- #
###############################################################################
MODELS = {
    "gpt-5-preview": {          # CHANGE to the exact ID you see (e.g. gpt-5-preview-0619)
        "vendor": "openai",
    },
    "claude-sonnet-4.5": {      # e.g. claude-3-sonnet-20240229 or your preview model
        "vendor": "anthropic",
    },
    "gemini-3-pro": {           # e.g. gemini-1.5-pro-preview-latest
        "vendor": "google",
    },
    # You can append more: "my-internal-preview": {"vendor": "openai"} …
}

TEMPERATURE = 0.2
OUTDIR      = pathlib.Path("eval_runs")
OUTDIR.mkdir(exist_ok=True)

###############################################################################
# 1. PROMPTS / TASKS -- modify freely                                         #
###############################################################################
TASKS = [
    {
        "name": "short_reasoning",
        "prompt": (
          "You have 3 numbered bags. Bag 1 has 1 white and 2 black marbles, "
          "Bag 2 has 2 white and 1 black marble, Bag 3 has 3 white marbles. "
          "You pick a bag at random and draw one marble at random, which turns "
          "out to be white. What is the probability it came from each bag? "
          "Return ONLY valid JSON: {\"bag1\": <float>, \"bag2\": <float>, \"bag3\": <float>} "
          "Sum should equal 1."
        ),
        "eval": lambda txt: json.loads(txt)  # will raise if invalid JSON
    },
    {
        "name": "code_generation",
        "prompt": (
          "Write a Python function diff_json(a: dict, b: dict) that returns a "
          "minimal patch converting a to b. Include pytest unit tests. "
          "Return only JSON with keys code (string) and tests (string)."
        ),
        "eval": lambda txt: all(k in json.loads(txt) for k in ("code","tests"))
    },
    {
        "name": "long_context",
        "prompt_builder": lambda needle: (
          f"{'irrelevant sentence. ' * 5000}"
          f"NEEDLE-START {needle} NEEDLE-END "
          f"{'more filler text. ' * 5000}\n\n"
          f"Extract the sentence between NEEDLE-START and NEEDLE-END. "
          f"Return {{\"needle\": \"<sentence>\"}} only."
        ),
        "needle": "The answer to life is 42.",
        "eval": lambda txt, needle="The answer to life is 42.": needle in json.loads(txt).get("needle","")
    },
    {
        "name": "vision_extract",
        "prompt": (
          "The following is an image of a simple table with 3 rows (Name, Qty). "
          "Extract to JSON list of objects with keys name and qty. "
          "Return ONLY JSON."
        ),
        "image_path": "sample_receipt.png",   # replace w/ your own
        "eval": lambda txt: isinstance(json.loads(txt), list)
    },
]

###############################################################################
# 2. VENDOR WRAPPERS                                                          #
###############################################################################
def call_openai(model, prompt, image_b64=None):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    kwargs = dict(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role":"user","content": prompt}
        ]
    )
    if image_b64:
        kwargs["messages"][0]["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
    t0 = time.time()
    resp = openai.ChatCompletion.create(**kwargs)
    latency = time.time() - t0
    usage  = resp.usage
    text   = resp.choices[0].message.content.strip()
    return text, latency, usage.to_dict() if usage else {}

def call_anthropic(model, prompt, image_b64=None):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    t0 = time.time()
    if image_b64:
        msg_content = [
            {"type":"text","text": prompt},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":image_b64}}
        ]
    else:
        msg_content = prompt
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=TEMPERATURE,
        messages=[{"role":"user","content": msg_content}]
    )
    latency = time.time() - t0
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens
    }
    text = resp.content[0].text.strip()
    return text, latency, usage

def call_google(model, prompt, image_b64=None):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    from google.generativeai import GenerativeModel
    gmodel = GenerativeModel(model)
    if image_b64:
        img = {"mime_type":"image/png","data": base64.b64decode(image_b64)}
        t0 = time.time()
        resp = gmodel.generate_content(
            [prompt, img],
            generation_config={"temperature": TEMPERATURE}
        )
    else:
        t0 = time.time()
        resp = gmodel.generate_content(prompt, generation_config={"temperature":TEMPERATURE})
    latency = time.time() - t0
    # Google currently does not expose token counts
    usage = {}
    text = resp.text.strip()
    return text, latency, usage

VENDOR_CALLER = {"openai": call_openai,
                 "anthropic": call_anthropic,
                 "google":   call_google}

###############################################################################
# 3. MAIN LOOP                                                                #
###############################################################################
records = []
run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
for model_id, meta in MODELS.items():
    vendor = meta["vendor"]
    caller = VENDOR_CALLER[vendor]

    rprint(f"\n[bold cyan]### Testing {model_id} ({vendor}) ###[/]")
    for task in TASKS:
        # Skip vision test if image not available or vendor lacks modality
        if task["name"] == "vision_extract" and not pathlib.Path(task["image_path"]).exists():
            continue
        try:
            prompt = task.get("prompt")
            img64  = None
            if task["name"] == "long_context":
                prompt = task["prompt_builder"](task["needle"])
            if task.get("image_path"):
                img64 = base64.b64encode(open(task["image_path"],"rb").read()).decode()

            text, latency, usage = caller(model_id, prompt, img64)

            passed = task["eval"](text)
        except Exception as e:
            traceback.print_exc()
            text = str(e)
            latency = None
            usage = {}
            passed = False

        rec = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(timespec='seconds'),
            "model": model_id,
            "vendor": vendor,
            "task": task["name"],
            "passed": passed,
            "latency_s": round(latency, 3) if latency else None,
            "response_chars": len(text),
            **{f"usage_{k}": v for k,v in usage.items()}
        }
        records.append(rec)
        status = "✅" if passed else "❌"
        rprint(f"{task['name']:<20} {status}  {rec['latency_s']}s  {rec.get('usage_total_tokens','')} tokens")

###############################################################################
# 4. SAVE                                                                     #
###############################################################################
df = pd.DataFrame(records)
csv_path = OUTDIR / f"scorecard_{run_id}.csv"
df.to_csv(csv_path, index=False)
rprint(f"\n[green]Wrote results -> {csv_path}[/]")

import pandas as pd, pathlib, glob, rich
latest = sorted(glob.glob("eval_runs/scorecard_*.csv"))[-1]
df = pd.read_csv(latest)

# Pass-rate & latency summary
summary = (df.groupby(["model","task"])
             .agg(pass_rate=("passed","mean"),
                  avg_latency=("latency_s","mean"))
             .reset_index())

pivot = summary.pivot(index="model", columns="task", values="pass_rate")
print("\n=== Pass-rate (%) ===")
print((pivot*100).round(1).fillna("—"))

pivot_lat = summary.pivot(index="model", columns="task", values="avg_latency")
print("\n=== Avg latency (s) ===")
print(pivot_lat.round(2).fillna("—"))


# See rows where a model failed
fails = df[df.passed == False][["model","task","response_chars"]]
print(fails.head())

# View full text for one failure
row = df.loc[(df.model=="gemini-3-pro") & (df.task=="code_generation")].iloc[0]
print("\n--- Gemini response ---\n", row.to_dict().get("raw_text","(raw_text trimmed)"))

# Simple daily pass-rate plot
import matplotlib.pyplot as plt
df = pd.concat(map(pd.read_csv, glob.glob("eval_runs/scorecard_*.csv")))
daily = (df.groupby([df.timestamp.str[:10], "model"])
           .passed.mean()
           .unstack())
daily.plot(figsize=(9,4), marker="o")
plt.title("Daily overall pass-rate")
plt.ylabel("Pass rate")
plt.show()
plt.savefig("overall_pass_rate.png", dpi=150, bbox_inches="tight")
plt.savefig("overall_pass_rate.svg")

import seaborn as sns
sns.lineplot(data=df, x="timestamp", y="passed",
             hue="model", style="task", markers=True, dashes=False)
daily = (df.groupby([df.timestamp.str[:10], "model"])
           .passed.mean()
           .groupby(level=1).rolling(3).mean().reset_index())
daily.plot(figsize=(9,4), marker="o")
plt.title("Daily overall pass-rate")
plt.ylabel("Pass rate")
plt.show()

import plotly.express as px
fig = px.line(daily, x="timestamp", y="passed", color="model")
fig.write_html("pass_rate.html", auto_open=True)


"""
ADDENDUM: Handling Non-JSON Responses  
or wraps the answer in Markdown, commentary, or code fences.

So nothing is wrong with the SDK calls; the models just didn’t follow the
“Return ONLY valid JSON” constraint strictly enough for the naive parser.

────────────────────────
2. Easiest fix: strip fences & text before validating

Insert two tiny helpers at top of the script:
"""
### ADDENDUM: Handling Non-JSON Responses ##############################
import re, json

def extract_first_json_block(txt:str):
  """
  Grab the first {...} or [...] block; discard everything else.
  """
  m = re.search(r'(\{.*?\}|\[.*?\])', txt, flags=re.S)
  if not m:
      raise ValueError("No JSON found")
  return json.loads(m.group(1))