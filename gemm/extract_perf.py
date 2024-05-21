import pandas as pd
import yaml
import re
import argparse

from pathlib import Path


def extract(fname, tag='triton'):
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
        config["ID"] = f"fff{config["M"]}"#-{config["N"]}-{config["K"]}"
        # config["mnk"] = config["M"] * config["N"] * config["K"]
        # config['config abbr'] = f"M{config["M"]}_N{config["N"]}_K{config["K"]}_BM{config["BLOCK_SIZE_M"]}_BN{config["BLOCK_SIZE_N"]}_BK{config["BLOCK_SIZE_k"]}_GM{config["GROUP_SIZE_M"]}_SK{config["SPLIT_k"]}_nW{config["num_warps"]}_nS{config["num_stages"]}_EU{config["waves_per_eu"]}_kP{config["kpack"]}_mfma{config["mfmaInstrSize"]}"
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("result", type=str, help="perf result file")
    parser.add_argument("--tag", type=str, default="triton", help="perf result file")  # noqa e501
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    fname = Path(args.result)
    d1 = extract(fname, tag=args.tag)
    df1 = pd.DataFrame(d1)
    df1.to_csv(fname.stem + ".csv", index=False)


if __name__ == "__main__":
    main()
