import requests
import re

def get_model_checkpoints(model_id):
    api_url = f"https://huggingface.co/api/models/{model_id}/refs"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        refs = response.json()
        return parse_checkpoint_branches(refs['branches'])
    else:
        print(f"Error: {response.status_code}")
        return None

def parse_checkpoint_branches(branches) -> dict[int, str]:
    revisions = {}
    for branch in branches:
        if not branch['ref'].startswith('refs/heads/'):
            continue

        ref_name = branch['ref'].split('/', 2)[-1]
        match = re.match(r'(?:step[_]?|revision)?(\d+)', ref_name)
        if match:
            step = int(match.group(1))
            revisions[step] = ref_name

    return revisions