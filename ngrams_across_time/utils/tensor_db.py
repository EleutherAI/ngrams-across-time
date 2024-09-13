# Query tensors by tags and store them in a JSON-based database
import os
from typing import Any
import torch
from torch import Tensor
import time
import uuid
from tinydb import TinyDB, Query


class TensorDatabase:
    def __init__(self, db_path: str, tensor_dir: str):
        self.db_path = db_path
        self.tensor_dir = tensor_dir
        os.makedirs(self.tensor_dir, exist_ok=True)
        self.db = TinyDB(db_path)


    def add_tensor(self, tensor: Tensor, tags: dict[str, Any]) -> None:
        timestamp = int(time.time() * 1000000)
        unique_id = uuid.uuid4().hex[:3]
        filename = f"tensor_{timestamp}_{unique_id}.pt"
        
        torch.save(tensor, os.path.join(self.tensor_dir, filename))
        
        document = {"filename": filename, "tags": tags}
        self.db.insert(document)


    def query_tensors(self, **tags: Any) -> list[dict[str, Any]]:
        Tag = Query()
        query = Tag.tags.fragment(tags)
        results = self.db.search(query)

        output = []
        for result in results:
            tensor = torch.load(os.path.join(self.tensor_dir, result['filename']))
            tags_with_id = result['tags'].copy()
            tags_with_id['doc_id'] = result.doc_id
            output.append({
                "tags": tags_with_id,
                "tensor": tensor
            })
        return output


    def query_last(self, **tags: Any) -> dict[str, Any] | None:
        results = self.query_tensors(**tags)
        return results[-1] if results else None


    def remove_tensor(self, doc_id: int) -> None:
        result = self.db.get(doc_id=doc_id)
        if result:
            tensor_path = os.path.join(self.tensor_dir, result['filename']) # type: ignore
            if os.path.exists(tensor_path):
                os.remove(tensor_path)
            self.db.remove(doc_ids=[doc_id])


    def close(self) -> None:
        self.db.close()


    def clean_invalid_tensors(self) -> None:
        for item in self.db.all():
            tensor_path = os.path.join(self.tensor_dir, item['filename'])
            try:
                loaded = torch.load(tensor_path)
                if not isinstance(loaded, torch.Tensor):
                    self.remove_tensor(item.doc_id)
            except:
                self.remove_tensor(item.doc_id)

# Example usage
if __name__ == "__main__":
    db = TensorDatabase("test_tensor_db", "test_tensors")
    
    tensor: torch.Tensor = torch.rand(1024, 2049)
    # Arbitrary tags
    tags: dict[str, Any] = {
        "metric": "kl-div",
        "ngram": 3,
        "model": "EleutherAI/pythia-14m",
        "checkpoint": -6,
    }
    db.add_tensor(tensor, tags)
    
    results: list[dict[str, Any]] = db.query_tensors(metric="kl-div", model="EleutherAI/pythia-14m", checkpoint=-6)
    assert len(results) == 1

    doc_id = results[0]['tags']['doc_id']
    db.remove_tensor(doc_id=doc_id)

    results: list[dict[str, Any]] = db.query_tensors(metric="kl-div", model="EleutherAI/pythia-14m", checkpoint=-6)

    assert len(results) == 0
