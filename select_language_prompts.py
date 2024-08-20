# This script selects prompts for patching experiments using:
# - the first 1024 sequences of the pile validation dataset
# - n-gram distributions over this data produced by ngram models fit on the pile training dataset
from pathlib import Path
from argparse import ArgumentParser


from src.language.filter_data import filter_data
from src.language.collect_metrics import collect_model_data
from src.language.hf_client import get_basic_pythia_model_names


def parse_args():
    parser = ArgumentParser(description='Search for interesting data points between contiguous checkpoints')
    
    group = parser.add_argument_group("Model arguments")
    group.add_argument('--model', type=str, nargs='+', help='Model names')
    group.add_argument('--start', type=int, required=True, help='Start checkpoint')
    group.add_argument('--end', type=int, required=True, help='End checkpoint')
    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--n', type=int,nargs='+', default=[2], help='n-grams to gather data for')
    
    group = parser.add_argument_group("Data arguments")
    group.add_argument('--db_path', type=str, default="/mnt/ssd-1/tensor_db", help='Location to save data in SQLite database')
    
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db_path)

    # Collect metrics for each model checkpoint
    for model_name in args.model or get_basic_pythia_model_names():
        collect_model_data(
            model_name,
            args.start,
            args.end,
            args.batch_size,
            args.n,
            db_path,
        )

    # Use metrics to filter data to select prompts
    filter_data(args.model, Path(args.out), start=args.start, end=args.end, n=args.n)


if __name__ == '__main__':
    main()