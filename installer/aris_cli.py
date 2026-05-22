"""Single-binary CLI entry point for the Aris .exe build.

PyInstaller packs this module into a one-file executable. It dispatches to
the existing Aris entry points based on the first positional argument:

    aris.exe serve     [--host ... --port ...]      # FastAPI server
    aris.exe generate  --config CFG --prompt "..."  # one-shot completion
    aris.exe train     --config CFG ...             # pretraining loop
    aris.exe finetune  --base CKPT --data ...       # SFT / RM / PPO / CAI
    aris.exe info                                   # config + GPU info
    aris.exe version                                # show version
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_version(_: argparse.Namespace) -> int:
    try:
        from llm import __version__
    except Exception:
        __version__ = "unknown"
    print(f"aris {__version__}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    info: dict = {"aris_version": "unknown"}
    try:
        from llm import __version__

        info["aris_version"] = __version__
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        from llm.monitoring.gpu import sample_gpu_stats

        info["gpu_stats"] = [g.__dict__ for g in sample_gpu_stats()]
    except Exception:
        pass
    print(json.dumps(info, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.config:
        os.environ["ARIS_MODEL_CONFIG"] = args.config
    if args.checkpoint:
        os.environ["ARIS_CHECKPOINT"] = args.checkpoint
    if args.tokenizer:
        os.environ["ARIS_TOKENIZER_META"] = args.tokenizer
    if args.log_level:
        os.environ["ARIS_LOG_LEVEL"] = args.log_level

    from llm.api.server import build_app

    uvicorn.run(build_app(), host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    import torch

    from llm.inference.engine import GenerationRequest, InferenceEngine
    from llm.inference.sampling import SamplingParams
    from llm.model.aris import ArisForCausalLM
    from llm.model.config import ArisConfig
    from llm.tokenizer.tokenizer import ArisTokenizer

    config = ArisConfig.from_yaml(args.config)
    model = ArisForCausalLM(config)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    if args.tokenizer:
        tokenizer = ArisTokenizer.from_metadata(args.tokenizer)
    else:
        tokenizer = ArisTokenizer(backend="tiktoken", model_path="cl100k_base.tiktoken")

    engine = InferenceEngine(model=model, tokenizer=tokenizer)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    result = engine.generate(GenerationRequest(prompt=args.prompt, sampling=sampling))
    print(result.text)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from llm.training import pretrain

    sys.argv = ["aris-train"] + args.extra
    pretrain.main()
    return 0


def _cmd_finetune(args: argparse.Namespace) -> int:
    # Lightweight in-process job runner mirroring the /finetune route.
    print(json.dumps({"status": "not implemented in single-binary build", "method": args.method}))
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aris", description="Aris LLM single-binary CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    sub.add_parser("info", help="environment + GPU info as JSON").set_defaults(func=_cmd_info)

    serve = sub.add_parser("serve", help="run the FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", default=None, help="path to a model config YAML")
    serve.add_argument("--checkpoint", default=None)
    serve.add_argument("--tokenizer", default=None)
    serve.add_argument("--log-level", default="INFO")
    serve.set_defaults(func=_cmd_serve)

    gen = sub.add_parser("generate", help="one-shot text generation")
    gen.add_argument("--config", required=True)
    gen.add_argument("--checkpoint", default=None)
    gen.add_argument("--tokenizer", default=None)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--seed", type=int, default=None)
    gen.set_defaults(func=_cmd_generate)

    train = sub.add_parser("train", help="launch pretraining (delegates to llm.training.pretrain)")
    train.add_argument("extra", nargs=argparse.REMAINDER)
    train.set_defaults(func=_cmd_train)

    ft = sub.add_parser("finetune", help="finetuning entry (sft/rm/ppo/constitutional)")
    ft.add_argument("--method", choices=["sft", "rm", "ppo", "constitutional"], default="sft")
    ft.set_defaults(func=_cmd_finetune)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
