#!/usr/bin/env python3
"""Simple standalone benchmark runner for the v2 TreeFlash HF repo."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tree_flash import TreeFlash


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['question']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['problem']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{row['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: row["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda row: row["prompt"],
        "multi_turn": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a v2 TreeFlash drafter.")
    parser.add_argument("--drafter", required=True, help="Drafter model id or local checkpoint path.")
    parser.add_argument("--target", required=True, help="Target/verifier model id or path.")
    parser.add_argument("--tree-size", type=int, required=True, help="Draft tree size.")
    parser.add_argument("--verifier-temperature", type=float, required=True)
    parser.add_argument("--quality-datasets", choices=tuple(DATASETS), nargs="+", required=True)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument(
        "--is-chain",
        "--is_chain",
        dest="is_chain",
        action="store_true",
        help="Use DFlash-style chain drafting.",
    )
    parser.add_argument(
        "--compute-speedup",
        "--compute_speedup",
        dest="compute_speedup",
        action="store_true",
        help="Run vanilla verifier decoding and report drafter throughput speedup.",
    )
    parser.add_argument("--output-dir", default="./results")
    return parser.parse_args()


def load_quality_dataset(name: str, n_samples: int) -> list[dict[str, Any]]:
    config = DATASETS[name]
    dataset = load_dataset(*config["load_args"], **config["load_kwargs"])
    examples: list[dict[str, Any]] = []
    for row in dataset:
        if config.get("multi_turn"):
            turns = list(config["format"](row))
        else:
            turns = [config["format"](row)]
        examples.append({"turns": turns})

    if n_samples > 0 and len(examples) > n_samples:
        rng = random.Random(42)
        indices = sorted(rng.sample(range(len(examples)), k=n_samples))
        examples = [examples[index] for index in indices]
    return examples


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def decode_time_s(response: Any) -> float:
    return float(response.time_per_output_token) * int(response.num_output_tokens)


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tree_size": int(args.tree_size),
        "is_chain": bool(args.is_chain),
    }
    if not args.is_chain:
        kwargs["top_m"] = int(args.top_m)
    return kwargs


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    value_mean = mean(values)
    variance = sum((value - value_mean) ** 2 for value in values) / len(values)
    return {"mean": float(value_mean), "std": float(variance**0.5)}


def run_dataset(
    *,
    dataset_name: str,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    drafter: Any,
    target: Any,
    tokenizer: Any,
    gen_kwargs: dict[str, Any],
    stop_token_ids: list[int] | None,
) -> dict[str, Any]:
    output_tokens = 0
    decode_time = 0.0
    vanilla_output_tokens = 0
    vanilla_decode_time = 0.0
    acceptance_lengths: list[float] = []
    sample_throughputs: list[float] = []
    vanilla_sample_throughputs: list[float] = []
    sample_speedups: list[float] = []
    sample_results: list[dict[str, Any]] = []

    for sample_index, example in tqdm.tqdm(
        enumerate(examples),
        total=len(examples),
        desc=f"benchmark/{dataset_name}",
    ):
        messages: list[dict[str, str]] = []
        sample_output_tokens = 0
        sample_decode_time = 0.0
        sample_vanilla_output_tokens = 0
        sample_vanilla_decode_time = 0.0
        sample_acceptance_lengths: list[float] = []
        turns: list[dict[str, Any]] = []

        for turn_index, user_turn in enumerate(example["turns"]):
            messages.append({"role": "user", "content": user_turn})
            input_text = apply_chat_template(tokenizer, messages)
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            vanilla_turn: dict[str, Any] = {}
            if args.compute_speedup:
                vanilla_response = drafter.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    stop_token_ids=stop_token_ids,
                    temperature=float(args.verifier_temperature),
                    is_vanilla=True,
                    return_stats=True,
                )
                vanilla_turn_decode_time = decode_time_s(vanilla_response)
                vanilla_output_tokens += int(vanilla_response.num_output_tokens)
                sample_vanilla_output_tokens += int(vanilla_response.num_output_tokens)
                vanilla_decode_time += vanilla_turn_decode_time
                sample_vanilla_decode_time += vanilla_turn_decode_time
                vanilla_turn = {
                    "vanilla_num_output_tokens": int(vanilla_response.num_output_tokens),
                    "vanilla_decode_time_s": vanilla_turn_decode_time,
                    "vanilla_throughput_tok_per_s": (
                        float(vanilla_response.num_output_tokens) / max(vanilla_turn_decode_time, 1e-6)
                        if int(vanilla_response.num_output_tokens) > 0
                        else 0.0
                    ),
                }

            response = drafter.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
                stop_token_ids=stop_token_ids,
                temperature=float(args.verifier_temperature),
                return_stats=True,
                **gen_kwargs,
            )

            turn_decode_time = decode_time_s(response)
            generated_ids = response.output_ids[0, response.num_input_tokens :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": generated_text})

            turn_acceptance_lengths = [float(value) for value in response.acceptance_lengths]
            acceptance_lengths.extend(turn_acceptance_lengths)
            sample_acceptance_lengths.extend(turn_acceptance_lengths)
            output_tokens += int(response.num_output_tokens)
            sample_output_tokens += int(response.num_output_tokens)
            decode_time += turn_decode_time
            sample_decode_time += turn_decode_time

            turns.append(
                {
                    "turn_index": turn_index,
                    "input_text": input_text,
                    "generated_text": generated_text,
                    "num_input_tokens": int(response.num_input_tokens),
                    "num_output_tokens": int(response.num_output_tokens),
                    "decode_time_s": turn_decode_time,
                    "time_to_first_token_s": float(response.time_to_first_token),
                    "acceptance_lengths": [int(value) for value in response.acceptance_lengths],
                    **vanilla_turn,
                }
            )

        sample_throughput = (
            float(sample_output_tokens) / max(sample_decode_time, 1e-6)
            if sample_output_tokens > 0
            else 0.0
        )
        sample_throughputs.append(sample_throughput)
        sample_result = {
            "sample_index": sample_index,
            "num_output_tokens": sample_output_tokens,
            "decode_time_s": sample_decode_time,
            "throughput_tok_per_s": sample_throughput,
            "avg_acceptance": (
                float(mean(sample_acceptance_lengths)) if sample_acceptance_lengths else 0.0
            ),
            "turns": turns,
        }
        if args.compute_speedup:
            vanilla_sample_throughput = (
                float(sample_vanilla_output_tokens) / max(sample_vanilla_decode_time, 1e-6)
                if sample_vanilla_output_tokens > 0
                else 0.0
            )
            sample_speedup = (
                sample_throughput / vanilla_sample_throughput
                if vanilla_sample_throughput > 0.0
                else 0.0
            )
            vanilla_sample_throughputs.append(vanilla_sample_throughput)
            sample_speedups.append(sample_speedup)
            sample_result.update(
                {
                    "vanilla_num_output_tokens": sample_vanilla_output_tokens,
                    "vanilla_decode_time_s": sample_vanilla_decode_time,
                    "vanilla_throughput_tok_per_s": vanilla_sample_throughput,
                    "speedup": sample_speedup,
                }
            )
        sample_results.append(
            sample_result
        )

    throughput = float(output_tokens) / max(decode_time, 1e-6) if output_tokens > 0 else 0.0
    metrics: dict[str, Any] = {
        "num_samples": len(examples),
        "num_output_tokens": output_tokens,
        "decode_time_s": decode_time,
        "throughput_tok_per_s": throughput,
        "avg_acceptance": float(mean(acceptance_lengths)) if acceptance_lengths else 0.0,
        "sample_throughput_tok_per_s": summarize(sample_throughputs),
        "acceptance_length": summarize(acceptance_lengths),
    }
    if args.compute_speedup:
        vanilla_throughput = (
            float(vanilla_output_tokens) / max(vanilla_decode_time, 1e-6)
            if vanilla_output_tokens > 0
            else 0.0
        )
        metrics.update(
            {
                "vanilla_num_output_tokens": vanilla_output_tokens,
                "vanilla_decode_time_s": vanilla_decode_time,
                "vanilla_throughput_tok_per_s": vanilla_throughput,
                "speedup": throughput / vanilla_throughput if vanilla_throughput > 0.0 else 0.0,
                "vanilla_sample_throughput_tok_per_s": summarize(vanilla_sample_throughputs),
                "sample_speedup": summarize(sample_speedups),
            }
        )
    return {
        "metrics": metrics,
        "samples": sample_results,
    }


@torch.inference_mode()
def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    drafter = TreeFlash.from_pretrained(
        args.drafter,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    target.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)

    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    gen_kwargs = generation_kwargs(args)

    results_by_dataset: dict[str, Any] = {}
    for dataset_name in args.quality_datasets:
        examples = load_quality_dataset(dataset_name, int(args.n_samples))
        results_by_dataset[dataset_name] = run_dataset(
            dataset_name=dataset_name,
            examples=examples,
            args=args,
            drafter=drafter,
            target=target,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            stop_token_ids=stop_token_ids,
        )

    return {
        "config": {
            "drafter": args.drafter,
            "target": args.target,
            "tree_size": int(args.tree_size),
            "verifier_temperature": float(args.verifier_temperature),
            "quality_datasets": list(args.quality_datasets),
            "top_m": int(args.top_m),
            "max_new_tokens": int(args.max_new_tokens),
            "n_samples": int(args.n_samples),
            "is_chain": bool(args.is_chain),
            "compute_speedup": bool(args.compute_speedup),
        },
        "datasets": results_by_dataset,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(args)
    results_path = output_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "results_path": str(results_path),
        "config": results["config"],
        "metrics_by_dataset": {
            dataset_name: dataset_result["metrics"]
            for dataset_name, dataset_result in results["datasets"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
