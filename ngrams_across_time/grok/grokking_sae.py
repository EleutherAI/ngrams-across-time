# TODO train an SAE on each checkpoint
import os

import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
import einops
from tqdm import tqdm

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from nnsight import LanguageModel
from sae import SaeConfig, SaeTrainer, TrainConfig, Sae
from datasets import Dataset as HfDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from ngrams_across_time.utils.utils import assert_type
from ngrams_across_time.feature_circuits.circuit import get_circuit

device = torch.device("cuda")

# def set_patchable(hooked_model, state: bool):
#     hooked_model.set_use_attn_result(state)
#     hooked_model.set_use_hook_mlp_in(state)
#     hooked_model.set_use_split_qkv_input(state)
#     return hooked_model

def main():
    DATA_SEED = 598
    torch.manual_seed(DATA_SEED)

    # Generate dataset
    p = 113
    frac_train = 0.3
    a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
    equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)
    dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
    labels = (dataset[:, 0] + dataset[:, 1]) % p

    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    num_epochs = 32
    train_dataset = HfDataset.from_dict({
        "input_ids": train_data.repeat(num_epochs, 1),
        "labels": train_labels.repeat(num_epochs),
    })
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Define model
    hf_model = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=p + 1,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        max_position_embeddings=3,
        hidden_dropout=0,
        classifier_dropout=0.,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        device=device.type,
        seed=999,
        init_weights=True,
        normalization_type=None,
    ))
    
    # Model checkpoints
    PTH_LOCATION = "workspace/grokking_demo.pth"
    cached_data = torch.load(PTH_LOCATION)
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]

    # EAP dataloader
    train_data = train_data.cuda()
    answers = [batch.unsqueeze(dim=0).cuda() for batch in list(train_labels.unbind())]

    checkpoint_ablation_loss_increases = []
    for epoch, state_dict in tqdm(list(zip(checkpoint_epochs, model_checkpoints))[::10]):
        # Load model checkpoint
        hf_model.load_state_dict(state_dict)
        hf_model.cuda() # type: ignore

        # Train SAE on checkpoints
        run_name = f"grok.{epoch}"
        cfg = TrainConfig(
            SaeConfig(multi_topk=True), 
            batch_size=16,
            run_name=run_name,
            log_to_wandb=False,
            hookpoints=[
                "gpt_neox.embed_in", 
                "gpt_neox.layers.0",  # currently no way to specify with wildcards
                "gpt_neox.layers.*.attention.dense", 
                "gpt_neox.layers.*.mlp.dense_4h_to_h",
                "gpt_neox.final_layer_norm"
            ],
        )
        trainer = SaeTrainer(cfg, train_dataset, hf_model)
        trainer.fit()

        tokenizer = Tokenizer(model=WordLevel(vocab={str(i): i for i in range(p + 1)}, unk_token="[UNK]"))
        model = LanguageModel(hf_model, tokenizer)

        embed = model.gpt_neox.embed_in
        attns = [layer.attention for layer in model.gpt_neox.layers]
        mlps = [layer.mlp for layer in model.gpt_neox.layers]
        resids = [layer for layer in model.gpt_neox.layers]
        dictionaries = {}
        dictionaries[embed] = Sae.load_from_disk(
            os.path.join(run_name, 'gpt_neox.embed_in'),
            device=device
        )
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[attns[i]] = Sae.load_from_disk(
                os.path.join(run_name, f'gpt_neox.layers.{i}.attention.dense'),
                device=device
            )
            dictionaries[mlps[i]] = Sae.load_from_disk(
                os.path.join(run_name, f'gpt_neox.layers.{i}.mlp.dense_4h_to_h'),
                device=device
            )
            dictionaries[resids[i]] = Sae.load_from_disk(
                os.path.join(run_name, f'gpt_neox.layers.{i}'),
                device=device
            )

        # Run EAP on Patchable model with SAEs hooked in
        # Count the loss increase when ablating an arbitrary number of edges (10):
        # def sae_hook(value, hook, sae):
        #     return sae(value).sae_out
            
        # hooked_model.add_hook(f'blocks.0.hook_resid_post', partial(sae_hook, sae=sae)) # type: ignore

        # with patchable(hooked_model) as model:            
        #     ablation_model = patchable_model(
        #         deepcopy(model), 
        #         factorized=True, 
        #         device=device, 
        #         separate_qkv=True, 
        #         seq_len=train_data.shape[-1],
        #         slice_output="last_seq"
        #     )
        # ablation_model.cuda()

        mean_loss_increase_at_10 = get_loss_increase(
            model, 
            dataloader, 
            dictionaries, 
            train_data, 
            answers, 
            embed,
            attns,
            mlps,
            resids,
            k=10)
        checkpoint_ablation_loss_increases.append(mean_loss_increase_at_10)                
        print(checkpoint_ablation_loss_increases[-1])


def get_loss_increase(
        model, 
        dataloader, 
        dictionaries, 
        train_data, 
        answers, 
        embed,
        attns,
        mlps,
        resids,
        k=10, 
        threshold=0.01
    ):
    
    # edge_prune_scores: PruneScores = mask_gradient_prune_scores(
    #     model=ablation_model, dataloader=dataloader, official_edges=set(), 
    #     grad_function="logit", answer_function="avg_diff", mask_val=0.0
    # )
    # logits = run_circuits(ablation_model, dataloader, [k], prune_scores=edge_prune_scores, ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT)
    # batch_logits = list(logits.values())[0] # this key will be edges - num_edges
    # batch_logits = assert_type(dict, batch_logits)
    
    loss_increases = []
    for batch in dataloader: 
        # patched_logits = batch_logits[batch.key]
        def get_patched_logits() -> Tensor:
            def metric_fn(model):
                return (
                    -1 * torch.gather(
                        F.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=answers[:-1].view(-1, 1)
                    ).squeeze(-1)
                )
            
            nodes, edges = get_circuit(
                train_data[:-1],
                train_data[1:],
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                aggregation="sum",
                node_threshold=threshold,
                edge_threshold=threshold,
            )

            # Run circuit
            breakpoint()
            return torch.zeros(1)

        patched_logits = get_patched_logits()
        patched_loss = F.cross_entropy(patched_logits.to(device).squeeze(), batch.answers.to(device).squeeze())
        
        unpatched_logits = model(batch.clean)[:, -1, :]
        unpatched_loss = F.cross_entropy(unpatched_logits.squeeze(), batch.answers.to(device).squeeze())
        
        loss_increases.append((patched_loss - unpatched_loss).item())
    
    return np.mean(loss_increases)



if __name__ == "__main__":
    # sae = Sae.load_from_disk("sae-ckpts/transformer.h.0")
    # load the model
    main()