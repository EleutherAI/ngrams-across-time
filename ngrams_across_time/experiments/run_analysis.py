import argparse
from pathlib import Path
from collect_metrics import get_metric_function
from filter_data import filter_data
from ngrams_across_time.image.load_image_model_data import get_available_image_models, load_models_and_synthetic_images, get_available_checkpoints as get_image_checkpoints
from ngrams_across_time.language.load_language_model_data import load_models_and_ngrams
from ngrams_across_time.language.hf_client import get_basic_pythia_model_names, get_model_checkpoints as get_language_checkpoints

loader = {
    'language': load_models_and_ngrams,
    'image': load_models_and_synthetic_images,
}

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

    try:
        models, dataset = loader[args.modality](args.model_name, args.order, args.dataset)
    except Exception as e:
        print(f"Error loading model and dataset: {e}")
        print("This is likely because an unavailable model or dataset was specified.")
        if args.modality == "language":
            models = get_basic_pythia_model_names()
            print("Available language models:")
            for model in models:
                print(f"- {model}")
            checkpoints = get_language_checkpoints(model)
            print(f"Available checkpoints for {model}:")
            for step, revision in sorted(checkpoints.items()):
                print(f"Step: {step}, Revision: {revision}")
        else:  # image
            models = get_available_image_models()
            print("Available image models:")
            for model in models:
                print(f"- {model}")
            model_name, dataset = model.split(" (")
            dataset = dataset.rstrip(")")
            checkpoints = get_image_checkpoints(model_name, dataset)
            print(f"Available checkpoints for {model_name} on {dataset}:")
            for checkpoint in sorted(checkpoints):
                print(f"Checkpoint: {checkpoint}")
    #     return

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
        dataset,  # data is loaded inside filter_data
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