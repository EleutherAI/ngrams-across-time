import argparse
from pathlib import Path
from collect_metrics import get_metric_function
from filter_data import filter_data

from ngrams_across_time.experiments.load_models import load_models

def main():
    parser = argparse.ArgumentParser(description="Run analysis on language or image models")
    parser.add_argument("model_name", nargs='?', default="", help="Name of the model to analyze")
    parser.add_argument("dataset", nargs='?', default="", help="Name of the dataset to use")
    parser.add_argument("--modality", choices=["language", "image"], default="language", help="Modality of the model")
    parser.add_argument("--start", type=int, default=1, help="Start checkpoint")
    parser.add_argument("--end", type=int, default=16384, help="End checkpoint")
    parser.add_argument("--order", type=int, default=2, help="Target n-gram order")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--quantile", type=float, default=0.1, help="Quantile for filtering")
    args = parser.parse_args()

    db_path = Path("data/metric_db")
    data_dir = Path("data/filtered")

    models, dataset = load_models(args.modality, args.model_name, args.dataset, args.order)

    metric_fn = get_metric_function(
        args.model_name,
        db_path,
        dataset,
        models,
        args.modality,
        batch_size=args.batch_size,
        target_order=args.order
    )
    for checkpoint in [128,256]:#models.keys():
        metric_fn(checkpoint)

    if args.modality == "language":
        metric = "kl"
    else:
        metric = "loss"

    # Filter data
    filter_data(
        args.model_name,
        db_path,
        dataset,
        args.start,
        args.end,
        args.order,
        args.modality,
        quantile=args.quantile,
        metric=metric,
        data_dir=data_dir
    )

if __name__ == "__main__":
    main()