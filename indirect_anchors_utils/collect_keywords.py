import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


def generate_keywords(client, target, num_keywords):
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Please give me {num_keywords} keywords describing {target}'s appearance or you believe are very related to {target}, separated by comma. Start your response directly.",
                },
            ],
            temperature=0,  # for greedy decoding
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating keywords for {target}: {str(e)}")
        return ""


def main(num_keywords):
    targets = pd.read_csv("../prompts/targets.csv")["target"].tolist()
    df = pd.DataFrame({"target": targets})

    client = OpenAI()

    keywords = []
    for target in tqdm(targets, desc="Generating keywords"):
        ans = generate_keywords(client, target, num_keywords)
        if target in ans:
            ans = ans.replace(target, "").replace(", ,", ",").lstrip(" ,")
        keywords.append(ans)
        print(f"{target}: {ans}")

    df["keyword"] = keywords
    output_file = f"../prompts/keywords_lm/n={num_keywords}_related_greedy.csv"
    df.to_csv(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate keywords for targets using OpenAI API"
    )
    parser.add_argument(
        "--num_keywords", type=int, required=True, help="Number of keywords to generate"
    )
    args = parser.parse_args()

    main(args.num_keywords)
