import yaml, asyncio, pandas as pd, time, uuid, datetime, importlib
from vendors import VENDOR_CALL
cfg = yaml.safe_load(open("config.yaml"))
run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:6]
records = []
sem = asyncio.Semaphore(cfg["concurrency"])

async def run_one(model, task_name):
    mod = importlib.import_module(f"tasks.{task_name}")
    prompt = await mod.prompt()
    async with sem:
        t0=time.time()
        txt, usage = await VENDOR_CALL[model["vendor"]](model["id"], prompt)
        latency=time.time()-t0
    try:
        ok = await mod.validate(txt)
    except Exception:
        ok = False
    return dict(run_id=run_id, model=model["id"], vendor=model["vendor"],
                task=task_name, passed=ok, latency=latency, raw=txt[:4000],
                **{f"tokens_{k}":v for k,v in usage.items()})

async def main():
    jobs=[run_one(m,t) for m in cfg["models"] for t in cfg["tasks"]]
    df = pd.DataFrame(await asyncio.gather(*jobs))
    df.to_csv(f"eval_runs/scorecard_{run_id}.csv",index=False)
asyncio.run(main())