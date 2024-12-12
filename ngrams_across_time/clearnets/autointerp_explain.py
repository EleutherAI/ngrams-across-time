import asyncio
import json
import os
from functools import partial

import orjson
import torch
import time
from simple_parsing import ArgumentParser

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer, DetectionScorer


# run with python -m ngrams_across_time.clearnets.autointerp_explain --model "data/tinystories/sparse-8m-max-e=200-esp=15-s=42" --module "roneneldan/TinyStories/sparse-feedfwd-transformer/.transformer.h.6.mlp" --n_random 50 --n_examples_test 50 --train_type "quantiles" --n_examples_train 40 --n_quantiles 10 --width 255 


def main(args):
    modules = args.modules
    feature_cfg: FeatureConfig = args.feature_options
    experiment_cfg: ExperimentConfig = args.experiment_options
    shown_examples = args.shown_examples
    n_features = args.features
    start_feature = args.start_feature
    sae_model = args.model
    feature_dict = {
        f"{module}": torch.arange(start_feature, start_feature + n_features)
        for module in modules
    }
    dataset = FeatureDataset(
        raw_dir=f"raw_features",
        cfg=feature_cfg,
        modules=modules,
        features=feature_dict,
    )

    constructor = partial(
        default_constructor,
        tokens=dataset.tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples,
    )
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    ### Load client ###

    client = Offline(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        max_memory=0.8,
        max_model_len=5120,
    )

    ### Build Explainer pipe ###
    def explainer_postprocess(result):

        safe_feature_name = str(result.record.feature).replace("/", "--")
        with open(f"results/explanations/{sae_model}/{experiment_name}/{safe_feature_name}.txt", "wb") as f: f.write(orjson.dumps(result.explanation))

        return result

    # try making the directory if it doesn't exist
    os.makedirs(f"results/explanations/{sae_model}/{experiment_name}", exist_ok=True)

    explainer_pipe = process_wrapper(
        DefaultExplainer(
            client,
            tokenizer=dataset.tokenizer,
            threshold=0.3,
        ),
        postprocess=explainer_postprocess,
    )

    # save the experiment config
    with open(
        f"results/explanations/{sae_model}/{experiment_name}/experiment_config.json",
        "w",
    ) as f:
        print(experiment_cfg.to_dict())
        f.write(json.dumps(experiment_cfg.to_dict()))

    ### Build Scorer pipe ###

    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.random_examples

        return record

    def scorer_postprocess(result, score_dir):
        record = result.record
        safe_feature_name = str(record.feature).replace("/", "--")
        with open(
            f"results/scores/{sae_model}/{experiment_name}/{score_dir}/{safe_feature_name}.txt",
            "wb",
        ) as f:
            f.write(orjson.dumps(result.score))

    os.makedirs(
        f"results/scores/{sae_model}/{experiment_name}/detection", exist_ok=True
    )
    os.makedirs(f"results/scores/{sae_model}/{experiment_name}/fuzz", exist_ok=True)

    # save the experiment config
    with open(
        f"results/scores/{sae_model}/{experiment_name}/detection/experiment_config.json",
        "w",
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    with open(
        f"results/scores/{sae_model}/{experiment_name}/fuzz/experiment_config.json", "w"
    ) as f:
        f.write(json.dumps(experiment_cfg.to_dict()))

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                tokenizer=dataset.tokenizer, # type: ignore
                batch_size=shown_examples,
                verbose=False,
                log_prob=True,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="detection"),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                tokenizer=dataset.tokenizer, # type: ignore
                batch_size=shown_examples,
                verbose=False,
                log_prob=True,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir="fuzz"),
        ),
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        # The loader must generate pipeline inputs
        loader,
        explainer_pipe,
        scorer_pipe,
    )
    start_time = time.time()
    asyncio.run(pipeline.run(50))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shown_examples", type=int, default=5)
    parser.add_argument("--start_feature", type=int, default=0)
    parser.add_argument("--model", type=str, default="gemma/16k")
    parser.add_argument("--modules", nargs="+", default=[".model.layers.10"])
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_arguments(ExperimentConfig, dest="experiment_options")
    parser.add_arguments(FeatureConfig, dest="feature_options")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    main(args)
