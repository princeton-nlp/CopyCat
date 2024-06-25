import sys
import pandas as pd
import json
import os
import t2v_metrics
from tqdm import tqdm
import argparse
import random


def main(args):
    # Load requirements
    requirements = pd.read_csv(args.entity_file, index_col=1)
    all_requirements = set(requirements["keyword"])
    requirements = requirements.to_dict("index")

    # Load data
    with open(os.path.join(args.dir, "log.json"), "r") as f:
        data = json.load(f)

    dataset = []
    for res in tqdm(data, desc="Processing data"):
        img_path = os.path.join("DIR", res["image_path"])
        target = res["target"]
        requirement = requirements[target]["keyword"]

        dataset.append({"images": [img_path], "texts": [requirement]})

    # Compute scores
    clip_flant5_score = t2v_metrics.VQAScore(
        model="clip-flant5-xxl",
        device="cuda",
    )
    scores = clip_flant5_score.batch_forward(dataset=dataset, batch_size=16)

    # Save results
    target_file = os.path.join(args.output_dir, "consistency.csv")
    new_results = pd.DataFrame(
        {"dir": [args.dir], "score": [scores.cpu().numpy().mean()]}
    )

    if os.path.exists(target_file):
        existing_results = pd.read_csv(target_file)
        all_results = pd.concat([existing_results, new_results], ignore_index=True)
    else:
        all_results = new_results

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    all_results.to_csv(target_file, index=False)
    print(f"Results saved to {target_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and evaluate text-to-image consistency"
    )

    parser.add_argument(
        "--entity_file",
        type=str,
        default="./prompts/consistency/key_entitie_n1.csv",
        help="Path to the entity file",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./results/playground-v2.5-1024px-aesthetic/targets/run-1/neg_prompt_none",
        help="Directory containing log.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prompts/consistency",
        help="Directory to save results",
    )
    args = parser.parse_args()
    main(args)
