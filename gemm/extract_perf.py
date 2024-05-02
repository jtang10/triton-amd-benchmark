import pandas as pd
import yaml
import re

def extract(fname):
    tag = "fork" if "fork" in fname else "upstream"
    with open(fname) as f:
        file_content = f.read()
    perfs = re.findall(
        r'TFLOPS\: ((?:[0-9]*[.])?[0-9]+) time\(us\)\: ((?:[0-9]*[.])?[0-9]+)',
        file_content)
    with open(fname) as f:
        configs = yaml.safe_load(f)
    assert len(perfs) == len(configs)
    for perf, config in zip(perfs, configs):
        config["Throughput (TFlops)"] = perf[0]
        config["time (us)"] = perf[1]
        config["Type"] = tag

    return configs

d1 = extract("upstream.yaml")
df1 = pd.DataFrame(d1)
# d2 = extract("pytorch_benchmark_fork_full.yaml")
# df2 = pd.DataFrame(d2)
# df = pd.concat([df1, df2])
df1.to_csv("pytorch_20240424.csv", index=False)