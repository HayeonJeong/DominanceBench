# Dominant vs. Dominated: DominanceBench prompts and DvD Score

Prompts and scoring code for **"Dominant vs. Dominated: Concept-Level Generative Collapse in Diffusion Models"** (ACCV 2026).

<!-- TODO: add paper link, author list, and project page before making the repository public. -->

This repository contains:

- **DominanceBench** (300 prompts) and the additional prompt sets used in the paper.
- **DvD Score** code, which uses `Qwen/Qwen2-VL-2B-Instruct` as a yes/no judge.

Image generation code, attention analysis (Focus Score), and head ablation code are not part of this release.

## Repository layout

```
dvd-release/
├── README.md
├── requirements.txt
├── score_images.py                          # entry point
├── data/
│   ├── dominancebench_300.csv               # DominanceBench
│   └── independent_prompts_300.csv          # independent template-based set (no SD1.4 filtering)
└── dvd_score/
    ├── questions.py                         # question sets and answer instructions
    ├── judge.py                             # Qwen2-VL yes/no judge
    ├── scorer.py                            # per-image DvD Score
    ├── dataset.py                           # prompt CSV and image lookup
    ├── config.py                            # command-line options
    └── pipeline.py                          # scoring loop and CSV output
```

## Prompt sets

### `dominancebench_300.csv`

| column | meaning |
|---|---|
| `case_num` | prompt id (1-300) |
| `content_type` | `artist`, `landmark`, or `character` (100 each) |
| `content` | the concept expected to dominate (e.g. `rembrandt`) |
| `nm_word` | the object expected to be suppressed (e.g. `notebook`) |
| `prompt` | text prompt given to the diffusion model |

### `independent_prompts_300.csv`

300 prompts built from the same concept/object vocabulary with fixed templates, without SD1.4-based filtering. Same columns as above plus `object_prompt_phrase`, `template_id`, and `source`.

## DvD Score

For each generated image, the judge answers five yes/no questions about the dominant concept and five about the object (see `dvd_score/questions.py`).

- `c` = number of "Yes" answers to the concept questions (0-5)
- `n` = number of "Yes" answers to the object questions (0-5)
- **DvD = c × (5 − n) / 25 × 100**, in [0, 100]

DvD is 100 when the concept is fully present and the object is fully absent, and 0 when either the concept is absent or the object is fully present. The per-prompt score (`dvd_mean_case` in the output) is the mean over that prompt's images.

Judge details that affect the numbers:

- Model: `Qwen/Qwen2-VL-2B-Instruct`, greedy decoding, `max_new_tokens=5`, bfloat16 autocast. An answer counts as "Yes" only if the decoded text starts with `yes`.
- Each question is asked in a separate call with one image.
- Artist concept questions use a lenient instruction (a stylistic resemblance counts as "Yes"). All other questions, including the object questions for artist prompts, use a strict "Answer only 'Yes' or 'No'" instruction.
- The instruction strings end with `/no_think`. This token has no special meaning for Qwen2-VL; it is kept so that the released code sends exactly the same text as the code used for the paper.
- Processor pixel limits: `min_pixels = 256·28·28`, `max_pixels = 1280·28·28`. For 512×512 and 768×768 images these limits do not change the resized input (504×504 and 756×756 either way).

## Generation settings used in the paper

| | SD 1.4 | SD 2.1 |
|---|---|---|
| model | `CompVis/stable-diffusion-v1-4` | `stabilityai/stable-diffusion-2-1` |
| resolution | 512×512 | 768×768 |
| scheduler | DDIM, 50 steps | DDIM, 50 steps |
| guidance scale | 7.5 | 7.5 |
| seeds per prompt | 10 | 10 |

The 10 seeds are `random.Random(42)` draws of `randint(1, 10000)`, identical for every prompt:

```
1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675
```

`stabilityai/stable-diffusion-2-1` is no longer downloadable from Hugging Face. `sd2-community/stable-diffusion-2-1` hosts the same UNet weights.

## Usage

```bash
pip install -r requirements.txt

python score_images.py \
    --meta_csv data/dominancebench_300.csv \
    --gen_dir outputs/sd14_dominancebench \
    --output_csv results/sd14_dominancebench_dvd.csv \
    --device 0
```

`--gen_dir` must contain one folder per prompt named `caseNNN_<anything>` (zero-padded to three digits, e.g. `case001_notebook_windmill_rembrandt`), with that prompt's images inside.

Options:

| option | default | meaning |
|---|---|---|
| `--device` | `0` | CUDA index, `-1` for CPU |
| `--model_name` | `Qwen/Qwen2-VL-2B-Instruct` | judge model |
| `--case_start`, `--case_end` | none | score only this inclusive range of `case_num` |
| `--max_images_per_case` | `0` | use only the first N images per prompt (0 = all) |
| `--allow_gallery_fallback` | off | if a case folder is empty, use `<gen_dir>/0_gallery/caseNNN_*` images |

Output: one row per image with the 10 answers (`content_answers`, `nm_answers`), `content_score`, `nm_score`, `dvd_score`, and `dvd_mean_case`.

A single Qwen2-VL-2B judge fits on a 24 GB GPU.

## License

<!-- TODO: choose a license for the code and the prompt files. -->

## Citation

<!-- TODO: add BibTeX after the proceedings entry is available. -->
