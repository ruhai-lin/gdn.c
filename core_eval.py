"""
CORE metric evaluation for a trained GDN base model (as in the DCLM paper).

Streamlined, self-contained port of nanochat's CORE eval. Run as:

    python core_eval.py --checkpoint outputs/climbmix110M_50B_100k/climbmix110M.pt

This evaluates the checkpoint on the CORE benchmark and writes results next to
the checkpoint (same directory), e.g. outputs/climbmix110M_50B_100k/:
  - <checkpoint_stem>.bin   the exported run.c checkpoint
  - core_eval.csv           per-task accuracy / centered score + CORE metric

The eval_bundle (task configs + data) is downloaded on demand to data/eval_bundle/.
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile

import requests
import torch
import torch.nn.functional as F
from jinja2 import Template
from tqdm import tqdm
import yaml

from export import load_checkpoint, model_export
from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

# -----------------------------------------------------------------------------
# Prompt rendering utilities (ported from nanochat/core_eval.py)

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompt for a language modeling task."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """Find the length of the common prefix ('left') or suffix ('right')."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # two prompts: without and with continuation. We locate the continuation via
    # the common prefix length (robust to subword merges at the boundary).
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx = find_common_length([tokens_without, tokens_with], direction='left')
    end_idx = len(tokens_with)
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """BxT token ids -> (BxT losses, BxT argmax predictions). Last column nan."""
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        outputs.view(batch_size * seq_len, -1).float(),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    losses[:, -1] = float('nan')
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # The model can't forward sequences beyond max_seq_len, so crop from the left.
    if getattr(model, 'max_seq_len', None) is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(max(s - num_to_crop, 0))
                new_end_idxs.append(max(e - num_to_crop, 0))
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()  # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id).to(device)

    losses, predictions = forward_model(model, input_ids)

    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    else:  # multiple_choice or schema
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """Evaluate one task across all of its examples, return mean accuracy."""
    correct = 0
    for idx in range(len(data)):
        correct += float(evaluate_example(idx, model, tokenizer, data, device, task_meta))
    return correct / len(data)

# -----------------------------------------------------------------------------
# Wrappers to adapt our GDN model / sentencepiece tokenizer to the eval interface

class ModelWrapper:
    """Exposes max_seq_len and a __call__ that returns all-position logits."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.max_seq_len = model.params.max_seq_len

    def __call__(self, input_ids):
        # GDNLM.forward returns full (B,T,V) logits when targets is given.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model(input_ids, targets=input_ids)


class TokenizerWrapper:
    """Adapts the sentencepiece Tokenizer to the nanochat eval calling convention."""
    def __init__(self, enc):
        self.enc = enc

    def __call__(self, prompts, prepend=None):
        # prompts: list[str] -> list[list[int]], prepending BOS when requested
        bos = prepend is not None
        return [self.enc.encode(p, bos=bos, eos=False) for p in prompts]

    def get_bos_token_id(self):
        return self.enc.bos_id

# -----------------------------------------------------------------------------
# eval_bundle download

def place_eval_bundle(zip_path, eval_bundle_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)


def ensure_eval_bundle():
    eval_bundle_dir = os.path.join(DATA_CACHE_DIR, "eval_bundle")
    if os.path.exists(eval_bundle_dir):
        return eval_bundle_dir
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_CACHE_DIR, "eval_bundle.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading eval bundle from {EVAL_BUNDLE_URL}...")
        resp = requests.get(EVAL_BUNDLE_URL, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                bar.update(f.write(chunk))
    print("Unpacking eval bundle...")
    place_eval_bundle(zip_path, eval_bundle_dir)
    return eval_bundle_dir

# -----------------------------------------------------------------------------
# CORE evaluation

def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """Evaluate on the CORE benchmark. Returns results, centered_results, core_metric."""
    eval_bundle_dir = ensure_eval_bundle()
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        tasks = yaml.safe_load(f)['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results, centered_results = {}, {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' '),
        }
        data_path = os.path.join(data_base_path, task['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        rb = random_baselines[label]
        centered = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
        centered_results[label] = centered
        elapsed = time.time() - start_time
        print(f"  {label:<35} acc {accuracy:.4f} | centered {centered:.4f} | "
              f"{task_meta['num_fewshot']}-shot {task_meta['task_type']} | {elapsed:.1f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}

# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CORE metric evaluation for a GDN base model")
    parser.add_argument("--checkpoint", default=os.path.join("outputs", "15M", "climbmix15M.pt"))
    parser.add_argument("--tokenizer", default="tokenizer.model", help="sentencepiece tokenizer model")
    parser.add_argument("--max_per_task", type=int, default=-1, help="max examples per task (-1 = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint)) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load model and tokenizer
    model = load_checkpoint(args.checkpoint).to(device)
    model.eval()
    enc = Tokenizer(args.tokenizer)
    wrapped_model = ModelWrapper(model, device)
    wrapped_tokenizer = TokenizerWrapper(enc)

    # Export the run.c checkpoint into the output folder
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    bin_path = os.path.join(out_dir, f"{ckpt_stem}.bin")
    model_export(model, bin_path)

    # Run CORE evaluation
    print("\nRunning CORE evaluation...")
    core = evaluate_core(wrapped_model, wrapped_tokenizer, device, max_per_task=args.max_per_task)

    # Write the results table
    csv_path = os.path.join(out_dir, "core_eval.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Centered"])
        for label in core["results"]:
            writer.writerow([label, f"{core['results'][label]:.6f}", f"{core['centered_results'][label]:.6f}"])
        writer.writerow(["CORE", "", f"{core['core_metric']:.6f}"])

    print(f"\nCORE metric: {core['core_metric']:.4f}")
    print(f"Results written to: {csv_path}")
    print(f"Checkpoint exported to: {bin_path}")


if __name__ == "__main__":
    main()
