import argparse
import sys
import os


def create_parser():
    parser = argparse.ArgumentParser(
        prog="legionheto",
        description="LEGIONHETO - Comprehensive LLM Fine-Tuning Framework",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    train_parser.add_argument("--dataset", "-d", required=True, help="Dataset path")
    train_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--method", choices=["sft", "dpo", "orpo"], default="sft", help="Training method")
    
    merge_parser = subparsers.add_parser("merge", help="Merge models")
    merge_parser.add_argument("--models", nargs="+", required=True, help="Model paths to merge")
    merge_parser.add_argument("--method", choices=["slerp", "ties", "dare"], default="slerp", help="Merge method")
    merge_parser.add_argument("--output", "-o", required=True, help="Output path")
    
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model", "-m", required=True, help="Model path")
    export_parser.add_argument("--output", "-o", required=True, help="Output path")
    export_parser.add_argument("--format", choices=["gguf", "merged", "adapter"], default="gguf", help="Export format")
    export_parser.add_argument("--quantization", default="Q4_K_M", help="GGUF quantization type")
    
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    
    version_parser = subparsers.add_parser("version", help="Show version")
    
    return parser


def handle_train(args):
    print(f"Training {args.model} with {args.method} method...")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")


def handle_merge(args):
    print(f"Merging models using {args.method} method...")
    print(f"Models: {args.models}")
    print(f"Output: {args.output}")


def handle_export(args):
    print(f"Exporting model to {args.format} format...")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    if args.format == "gguf":
        print(f"Quantization: {args.quantization}")


def handle_info(args):
    from ..core.registry import ModelRegistry
    
    config = ModelRegistry.get_optimal_config(args.model)
    if config:
        print(f"\nModel: {args.model}")
        print(f"Architecture: {config.get('target_modules', 'unknown')}")
        print(f"Supports Flash Attention: {config.get('supports_flash_attn', False)}")
        print(f"Supports MLA: {config.get('supports_mla', False)}")
        print(f"Recommended LoRA r: {config.get('lora_r', 16)}")
        print(f"Recommended LoRA alpha: {config.get('lora_alpha', 32)}")
        print(f"Recommended learning rate: {config.get('learning_rate', 2e-4)}")
    else:
        print(f"Unknown model: {args.model}")


def handle_version():
    print("LEGIONHETO v0.2.0 - Comprehensive LLM Fine-Tuning Framework")
    print("Supports: Llama, Mistral, Qwen, DeepSeek, Phi, Gemma, Command, Falcon, Yi, and more")


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "train":
        handle_train(args)
    elif args.command == "merge":
        handle_merge(args)
    elif args.command == "export":
        handle_export(args)
    elif args.command == "info":
        handle_info(args)
    elif args.command == "version":
        handle_version()


if __name__ == "__main__":
    main()
