import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clean some continuations")
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()
    if args.json != "":
        with open(args.json) as f:
            data = json.load(f)
        new_file_name = args.json.split("'.json")[0] + "_clean.json"
        data = [
            {
                "prompt": item["prompt"],
                "continuation": item["continuation"].split("</s>")[0],
                "real": item["real"],
            }
            for item in data
        ]
        with open(new_file_name, "w") as f:
            json.dump(data, f)
