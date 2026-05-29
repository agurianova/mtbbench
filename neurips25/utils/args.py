import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Run the benchmark with specific parameters.")
    parser.add_argument("--doctor_model", type=str, default="google/gemma-3-12b-it", help="Name of doctor model to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./data/agent_logs/gemma/", help="Directory to save the result of the run.")
    parser.add_argument("--dataset", type=str, default="hancock", help="Dataset to evaluate on (hancock only).")
    parser.add_argument(
        "--skip_choice_prob_scoring",
        action="store_true",
        help="Skip extra vLLM pass for choice-probability confidence (saves VRAM).",
    )
    parser.add_argument(
        "--no_population_stats",
        action="store_true",
        help="Do not inject SEER/TCGA cohort statistics into survival/recurrence questions.",
    )
    return parser.parse_args()