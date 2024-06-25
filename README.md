<h1> <span><img src="assets/logo.png" alt="CopyCat" width="40" style="display: block; margin-left: auto; margin-right: auto;"/></span> Fantastic Copyrighted Beasts and How (Not) to Generate Them </h1>

<p>
This repository contains the code to implement methods and experiments presented in the paper: <i><a href="https://arxiv.org/abs/2406.14526">Fantastic Copyrighted Beasts and How (Not) to Generate Them</a></i>, by <a href="https://x.com/LuxiHeLucy">Luxi He<sup>*1</sup></a>, <a href="https://hazelsuko07.github.io/yangsibo/">Yangsibo Huang<sup>*1</sup></a>, <a href="https://swj0419.github.io/">Weijia Shi<sup>*2</sup></a>, <a href="https://tinghaoxie.com/">Tinghao Xie<sup>1</sup></a>, <a href="https://hliu.cc/">Haotian Liu<sup>3</sup></a>,  <a href="https://yuewang.xyz/">Yue Wang<sup>4</sup></a>, <a href="https://www.cs.washington.edu/people/faculty/lsz">Luke Zettlemoyer<sup>2</sup></a>, <a href="https://pluskid.org/">Chiyuan Zhang</a>, <a href="https://www.cs.princeton.edu/~danqic/">Danqi Chen<sup>1</sup></a>, <a href="https://www.peterhenderson.co">Peter Henderson<sup>1</sup></a>.
</p>
<p><sup>*</sup>Equal contribution</p>

<p><sup>1</sup>Princeton University, <sup>2</sup>University of
Washington, <sup>3</sup>University of Wisconsin-Madison, <sup>4</sup>University of Southern
California</p>

[Website](https://copycat-eval.github.io/) | [Paper](https://arxiv.org/abs/2406.14526)

## Table of Contents

- [Environment Setup](#environment)
- [Image Generation](#image-generation)
- [Input Prompts](#input-prompts)
- [Character Detection](#character-detection)
- [Consistency Evaluation](#consistency-evaluation)
- [Citation](#citation)

## Environment

We provide the environment for our experiments in environment.yml. You can reproduce the environment using `conda env create -f environment.yml `. Note that you need to specify environment path within the environment.yml file.

## Image Generation

You can generate images using one of our five image-generation models and one video-generation model by calling the `main.py` function. For example:

```bash
python main.py \
    --model playground-v2.5-1024px-aesthetic \
    --output_dir 'results' \
    --data prompts/targets_dalle_prompt.csv \
    --neg_prompt keyword-5-and-laion-5 \
    --run_id 1
```

Specification of arguments:

- `--model`: Currently support these following models for generation:
  - `playground-v2.5-1024px-aesthetic`, [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
  - `stable-diffusion-xl-base-1.0`, [SDXL v1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - `PixArt-XL-2-512x512`, [PixArt](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512)
  - `IF-I-XL-v1.0`, [DeepFloyd-IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
  - `damo-vilab/text-to-video-ms-1.7b`, [VideoFusion](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)
- `--output_dir`: Directory for storing model output
- `--neg_prompt`: Optional negative prompts to be included in generation. The default is `"none"`. Our code currently supports the following negative prompts
  - `"target"`: target character's name
  - `"keyword-5-greedy"`:top 5 keywords selected through LM's greedy decoding
  - `"keyword-5-embedding"`: top 5 keywords selected through embedding similarity comparison
  - `"keyword-5-LAION"`: top 5 keywords selected through co-occurrence analysis with LAION.
  - You can also custimize your negative prompts by adding/modifying options in `main.py` to load relevant files.
- `--data`: Directory to input prompt for model generation.
- `--run_id`: Identifier for different runs. Will be reflected in output directory name.

In addition to the generated image, we store a `log.json` file containing a dictionary for each generation with the following key information:

```json
{
  "target": "Mario",
  "prompt": "Mario",
  "negative_prompt": "",
  "image_path": "23.png",
  "has_error": false
}
```

## Input Prompts

The `prompts` folder contains different categories of input prompts. The full list of 50 characters in our evaluation suite $\mathsf{CopyCat}$ are listed in `targets.csv`. `targets_dalle_prompt.csv` is a collection of DALL·E rewritten prompts using target name as input to the rewriting pipeline.

- **Keywords-based prompts**: The `keywords` folder contains LM generated keywords with/ without greedy decoding. The `keywords_co-occurrence` folder contains prompts consisting of keywords having top-occurrence frequency with popular training corpora like LAION. The `keywords_embedding` folder contains keywords with high embedding space similarity to target characters.

- **Description-based prompts**: `targets_60words.csv` contains character descriptions around 60 words in length (to keep under CLIP's 77 tokens input length restriction). `targets_60words_dallerewrite.csv` are the rewritten description-based prompts.

The generation scripts for related keywords and rewritten prompts are located in the `indirect_anchors_utils` directory. Run `python indirect_anchors_utils/collect_keywords.py` or `python indirect_anchors_utils/get_dalle_rewrite.py` to generate these two sets of indirect anchor prompts respectively.

## Character Detection

**Generate binary score for each generated image**:
Example:

```bash
python detect_score/gpt4v_eval.py --dir OUTPUT_DIR
```

Replace `OUTPUT_DIR` with the folder containing output files.

The GPT-4v evaluator stores each evaluation in `binary_result.json` under `OUTPUT_DIR` in the following format:

```json
{
  "id": "23.png",
  "character": "Mario",
  "score": 1,
  "has_error": false,
  "error_msg": ""
}
```

For video evaluations, which are conducted on the first, middle, and last frames of the video, run:

```bash
python gpt4_eval_video.py --dir OUTPUT_DIR
```

## Consistency Evaluation

1. Collect key characteristics

To generate key characteristics for the characters:

```bash
python consistency_score/collect_entities.py --num_words N
```

Replace `N` with the number of desired key characteristics.

2. Run consistency check

Install [t2v-metrics](https://github.com/linzhiqiu/t2v_metrics)

```bash
pip install t2v-metrics
```

Run evaluation

```bash
python consistency_score/calculate_consistency.py --entity_file ENTITY_FILE --dir DIR
```

Replace `ENTITY_FILE` with the path to the file containing the key characteristics (generated in step 1). Replace `DIR` with the path to the directory containing the generated images you want to evaluate.

## Citation

If you find our code and paper helpful, please consider citing our work:

```bibtex
@article{he2024fantastic,
  title={Fantastic Copyrighted Beasts and How (Not) to Generate Them},
  author={Luxi He and Yangsibo Huang and Weijia Shi and Tinghao Xie and Haotian Liu and Yue Wang and Luke Zettlemoyer and Chiyuan Zhang and Danqi Chen and Peter Henderson},
  journal={arXiv preprint arXiv:2406.14526},
  year={2024}
}
```
