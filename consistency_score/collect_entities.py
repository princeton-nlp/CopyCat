from openai import OpenAI
import pandas as pd
import argparse
from tqdm import tqdm
import os


def main(args):
    # Read input CSV file
    df = pd.read_csv(args.input_file)
    targets = df["target"].tolist()

    client = OpenAI()

    keywords = []
    for target in tqdm(targets, desc="Processing targets"):
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Define the key entity of a character as its type of animal, or if human, specify age or occupation.\nWhat is {target}'s key entity? Respond in fewer than {args.num_words} words, starting directly.",
                },
            ],
        )

        keyword = completion.choices[0].message.content
        keywords.append(keyword)

    # Add keywords to the DataFrame
    df["keyword"] = keywords

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results to CSV
    output_file = os.path.join(args.output_dir, f"key_entities_n{args.num_words}.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process targets and generate key entities"
    )
    parser.add_argument(
        "--num_words",
        type=int,
        default=1,
        help="Maximum number of words in the entity description",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./prompts/targets.csv",
        help="Path to the input CSV file containing targets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prompts/consistency",
        help="Directory to save results",
    )
    args = parser.parse_args()

    main(args)
