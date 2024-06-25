import pandas as pd
import argparse
import tqdm
from openai import OpenAI
import json
import os
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
from pathlib import Path
import urllib.request

# from ipdb import set_trace as bp
from PIL import Image
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="prompts/targets.csv",
        help="path to the prompts",
    )
    parser.add_argument("--model", type=str, default="dall-e-3", help="model name")
    parser.add_argument("--run_id", type=int, default=1, help="which run is this?")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/human_description",
        help="path to save the data",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        choices=[
            "none",
            "copyright",
            "target",
            "keyword-1",
            "keyword-3",
            "keyword-5",
            "keyword-5-greedy",
            "keyword-5-embedding",
            "keyword-5-laion",
            "keyword-5-greedy-and-laion-5",
            "keyword-5-and-laion-5",
            "keyword-10",
            "keyword-1-noname",
            "keyword-3-noname",
            "keyword-5-noname",
            "keyword-10-noname",
            "keyword-5-greedy-noname",
            "keyword-5-embedding-noname",
            "keyword-5-laion-noname",
            "keyword-5-greedy-and-laion-5-noname",
        ],
        default="none",
        help="Negative prompt",
    )
    parser.add_argument(
        "--dalle_rewrite",
        type=bool,
        default=False,
        help="whether to apply Dalle rewrite to the prompt.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data_name = args.data.replace(".csv", "").split("/")[-1]
    print(data_name)
    log_folder = f"{args.output_dir}/{args.model}/{data_name}/run-{args.run_id}/neg_prompt_{args.neg_prompt}"

    target_key = "target"
    prompts = pd.read_csv(args.data, encoding="unicode_escape").to_dict("records")
    prompt_key = "prompt"
    targets = pd.read_csv("prompts/targets.csv")["target"].values.tolist()
    print(len(targets))

    if args.model in ["dall-e-3"]:
        client = OpenAI()
    elif args.model in [
        "playground-v2.5-1024px-aesthetic",
        "stable-diffusion-xl-base-1.0",
        "PixArt-XL-2-512x512",
        "damo-vilab/text-to-video-ms-1.7b",
        "IF-I-XL-v1.0",
    ]:
        if args.model == "playground-v2.5-1024px-aesthetic":
            pipe = DiffusionPipeline.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
        elif args.model == "PixArt-XL-2-512x512":
            pipe = DiffusionPipeline.from_pretrained(
                "PixArt-alpha/PixArt-XL-2-512x512",
            ).to("cuda")
        elif args.model == "stable-diffusion-xl-base-1.0":
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0"
            ).to("cuda")
        elif args.model == "damo-vilab/text-to-video-ms-1.7b":
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
        elif args.model == "IF-I-XL-v1.0":
            # stage 1
            stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
            )
            stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_1.enable_model_cpu_offload()

            # stage 2
            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-L-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_2.enable_model_cpu_offload()

            # stage 3
            safety_modules = {
                "feature_extractor": stage_1.feature_extractor,
                "safety_checker": stage_1.safety_checker,
                "watermarker": stage_1.watermarker,
            }
            stage_3 = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                **safety_modules,
                torch_dtype=torch.float16,
            )
            stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_3.enable_model_cpu_offload()

            pass

    res_dict = []

    if "keyword" in args.neg_prompt:
        if args.neg_prompt == "keyword-5-greedy":
            keywords = (
                pd.read_csv("prompts/keywords_lm/n=5_related_greedy.csv")
                .set_index("target")
                .T.to_dict("list")
            )

        elif args.neg_prompt == "keyword-5-embedding":
            keywords = (
                pd.read_csv(f"prompts/keywords_embedding/n=5.csv")
                .set_index("target")
                .T.to_dict("list")
            )
        elif args.neg_prompt == "keyword-5-laion":
            keywords = (
                pd.read_csv(f"prompts/keywords_co-occurrence/50keywords_laion2b_top5.csv")
                .set_index("target")
                .T.to_dict("list")
            )
        else:
            num_keywords = args.neg_prompt.split("-")[1]
            keywords = (
                pd.read_csv(f"prompts/keywords_lm/n={num_keywords}_related_greedy.csv")
                .set_index("target")
                .T.to_dict("list")
            )

        if "and-laion-5" in args.neg_prompt:
            keywords_complementary = (
                pd.read_csv("prompts/keywords_co-occurrence/50keywords_laion2b_top5.csv")
                .set_index("target")
                .T.to_dict("list")
            )
            for key in keywords.keys():
                if key in keywords_complementary.keys():
                    keywords[key][1] = (
                        keywords[key][1] + "," + keywords_complementary[key][1]
                    )
                else:
                    print(f"Character not found: {key}")

    for ip, prompt_dict in tqdm.tqdm(enumerate(prompts)):
        broken_prompt = {
            "prompt": prompt_dict[prompt_key],
            "target": prompt_dict[target_key],
        }

        has_error = False

        if "video" in args.model:
            img_path = f"{log_folder}/images/{ip}" + "_{}.png"
        else:
            img_path = f"{log_folder}/images/{ip}.png"
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)

        if args.model in ["dall-e-3"]:
            try:
                response = client.images.generate(
                    model=args.model,
                    prompt=broken_prompt["prompt"],
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    style="vivid",
                )
                revised_prompt = response.data[0].revised_prompt
                url = response.data[0].url
                error_msg = ""
                urllib.request.urlretrieve(url, img_path)
            except Exception as e:
                print(f"Error with prompt: {broken_prompt}")
                error_msg = str(e)
                has_error = True
                revised_prompt = ""
                url = ""
                img_path = ""
        elif args.model in [
            "playground-v2.5-1024px-aesthetic",
            "stable-diffusion-xl-base-1.0",
            "PixArt-XL-2-512x512",
            "damo-vilab/text-to-video-ms-1.7b",
        ]:
            if args.neg_prompt == "target":
                negative_prompt = prompt_dict[target_key]

            elif args.neg_prompt == "copyright":
                negative_prompt = "copyrighted character"
            elif "keyword" in args.neg_prompt:
                target_name = prompt_dict[target_key]
                if target_name in keywords.keys():
                    target_description = keywords[target_name][1]
                else:
                    target_description = ""
                    print(f"{target_name}: no description found")
                if "noname" not in args.neg_prompt:
                    negative_prompt = f"{target_name}, {target_description}"
                else:
                    negative_prompt = target_description
                negative_prompt = negative_prompt.replace("\n", "").replace(
                    ", cartoon", ""
                )
                print(negative_prompt)
            elif args.neg_prompt == "none":
                negative_prompt = ""

            # check video or image generation
            if args.model == "damo-vilab/text-to-video-ms-1.7b":
                video_frames = pipe(broken_prompt["prompt"]).frames[0]
                for frame_index in [0, 6, 15]:
                    first_frame = video_frames[frame_index, ...]
                    first_frame = (first_frame * 255).astype(np.uint8)
                    image = Image.fromarray(first_frame, "RGB")
                    image.save(img_path.format(frame_index))
            else:
                image = pipe(
                    prompt=broken_prompt["prompt"],
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=3,
                ).images[0]
                image.save(img_path)
            revised_prompt = ""
            url = ""
            error_msg = ""
            image.save(img_path)
        elif args.model in [
            "IF-I-XL-v1.0",
        ]:
            if args.neg_prompt == "target":
                negative_prompt = prompt_dict[target_key]
            elif args.neg_prompt == "copyright":
                negative_prompt = "copyrighted character"
            elif "keyword" in args.neg_prompt:
                target_name = prompt_dict[target_key]
                if target_name in keywords.keys():
                    target_description = keywords[target_name][1]
                else:
                    target_description = ""
                    print(f"{target_name}: no description found")
                if "noname" not in args.neg_prompt:
                    negative_prompt = f"{target_name}, {target_description}"
                else:
                    negative_prompt = target_description
                negative_prompt = negative_prompt.replace("\n", "").replace(
                    ", cartoon", ""
                )
                print(negative_prompt)
            elif args.neg_prompt == "none":
                negative_prompt = ""

            prompt = broken_prompt["prompt"]
            # text embeds

            generator = torch.manual_seed(args.run_id)
            if negative_prompt == "":
                prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
                image = stage_1(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images
                pt_to_pil(image)[0].save("./if_stage_I.png")
                print("passed stage 1")

                # stage 2:
                image = stage_2(
                    image=image,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images
                pt_to_pil(image)[0].save("./if_stage_II.png")
                print("passed stage 2")
            else:
                prompt_embeds, negative_embeds = stage_1.encode_prompt(
                    prompt=prompt, negative_prompt=negative_prompt
                )
                # stage 1:
                image = stage_1(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images
                pt_to_pil(image)[0].save("./if_stage_I.png")
                print("passed stage 1")

                # stage 2:
                image = stage_2(
                    image=image,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                ).images
                pt_to_pil(image)[0].save("./if_stage_II.png")
                print("passed stage 2")

            # stage 3:
            image = stage_3(
                prompt=prompt, image=image, generator=generator, noise_level=100
            ).images
            image[0].save(img_path)
            print("passed stage 3")

        res_dict.append(
            {
                "target": prompt_dict[target_key],
                "prompt": broken_prompt['prompt'],
                "negative prompt": negative_prompt,
                "image_path": img_path,
                "has_error": has_error,
            }
        )

        with open(f"{log_folder}/log.json", "w") as file:
            file.write(json.dumps(res_dict, indent=4))
