import argparse
import os
from helpers import get_dataset, run_benchmark_task
from loguru import logger


def main(args):
    """
    Main function to handle the benchmark task.

    Args:
        args: Parsed arguments from argparse.
    """
    if not os.path.exists(args.input_path):
        raise ValueError(f"Error: Input path '{args.input_path}' does not exist.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Output path '{args.output_path}' created.")

    if args.task not in ["hospitalization", "critical_outcome", "ed_reattendance"]:
        raise ValueError(
            f"Error: Invalid task '{args.task}'. Choose from 'hospitalization', 'critical_outcome', 'ed_reattendance'.")

    logger.info(f"Running benchmark with the following parameters:")
    logger.info(f"  Input path: {args.input_path}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Task: {args.task}")

    logger.info("Getting dataset...")
    X_train, y_train, X_test, y_test, df_train, df_test = get_dataset(args.input_path, args.task)

    logger.info(f"Starting benchmark: {args.task}...")
    run_benchmark_task(X_train, y_train, X_test, y_test, df_train, df_test, task=args.task, input_path=args.input_path,
                       output_path=args.output_path)

    logger.info("Benchmark completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark tasks.")
    parser.add_argument("--input_path", required=True, type=str, help="Path to the folder with input files.")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save output files.")
    parser.add_argument("--task", required=True, type=str,
                        choices=["hospitalization", "critical_outcome", "ed_reattendance"], help="Task to perform.")

    args = parser.parse_args()
    main(args)
