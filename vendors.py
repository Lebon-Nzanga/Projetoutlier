import os
import openai
import asyncio
import tiktoken
from utils import strip_json


async def call_openai(model_id, prompt, *, temperature=0.2, images=None, json_mode=False):
    """Call OpenAI ChatCompletion asynchronously.

    Parameters:
      - model_id: model identifier string
      - prompt: text prompt
      - temperature: float
      - images: optional image payload
      - json_mode: if True, request JSON response format

    Returns:
      tuple(text, usage_dict)
    """
    loop = asyncio.get_running_loop()
    func = openai.ChatCompletion.acreate
    kwargs = dict(
        model=model_id,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if images:
        kwargs["messages"][0]["content"] = images
    resp = await func(**kwargs)
    text = resp.choices[0].message.content.strip()
    usage = resp.usage.to_dict() if getattr(resp, "usage", None) else {}
    return text, usage
