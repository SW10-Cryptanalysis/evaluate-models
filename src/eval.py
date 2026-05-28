import argparse
from classes.native_cipher_evaluator import NativeMappingEvaluator
from classes.vllm_cipher_evaluator import VLLMCipherEvaluator

def main() -> None:
    """Entry point for the evaluation script routing to the appropriate engine."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    if args.mapping:
        evaluator = NativeMappingEvaluator(
            model_path=args.model_path,
            use_spaces=args.spaces,
            mapping=args.mapping,
        )
    else:
        evaluator = VLLMCipherEvaluator(
            model_path=args.model_path,
            use_spaces=args.spaces,
            mapping=args.mapping,
        )

    evaluator.run()


if __name__ == "__main__":
    main()
