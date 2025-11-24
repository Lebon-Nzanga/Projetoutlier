import pandas as pd, plotly.express as px, glob, yaml
df = pd.concat(map(pd.read_csv, glob.glob("eval_runs/scorecard_*.csv")))
df["date"]=df.run_id.str[:8]
pass_rate = (df.groupby(["date","model"]).passed.mean().reset_index())
fig = px.line(pass_rate, x="date", y="passed", color="model",
              title="Overall pass-rate").update_yaxes(range=[0,1])
fig.write_html("dashboard.html", auto_open=True)