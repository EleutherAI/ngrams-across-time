# Query tensors by tags and store them in a SQLite database

import os
import sqlite3
from typing import Any 
import torch
import time
import uuid

class TensorDatabase:
    def __init__(self, db_path: str, tensor_dir: str):
        self.db_path = db_path
        self.tensor_dir = tensor_dir

        os.makedirs(self.tensor_dir, exist_ok=True)
        
        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        self.create_table()


    def create_table(self) -> None:
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tensors (
                id INTEGER PRIMARY KEY,
                filename TEXT UNIQUE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                tensor_id INTEGER,
                key TEXT,
                value_type TEXT,
                value_int INTEGER,
                value_real REAL,
                value_text TEXT,
                FOREIGN KEY (tensor_id) REFERENCES tensors (id)
            )
        ''')
        self.conn.commit()


    def add_tensor(self, tensor: torch.Tensor, tags: dict[str, Any]) -> None:
        timestamp = int(time.time() * 1000000)
        unique_id = uuid.uuid4().hex[:3]
        filename = f"tensor_{timestamp}_{unique_id}.pt"
        torch.save(tensor, os.path.join(self.tensor_dir, filename))
        
        self.cursor.execute('INSERT INTO tensors (filename) VALUES (?)', (filename,))
        tensor_id: int = self.cursor.lastrowid # type: ignore
        
        for key, value in tags.items():
            if isinstance(value, int):
                self.cursor.execute('INSERT INTO tags (tensor_id, key, value_type, value_int) VALUES (?, ?, ?, ?)',
                                    (tensor_id, key, 'int', value))
            elif isinstance(value, float):
                self.cursor.execute('INSERT INTO tags (tensor_id, key, value_type, value_real) VALUES (?, ?, ?, ?)',
                                    (tensor_id, key, 'float', value))
            else:
                self.cursor.execute('INSERT INTO tags (tensor_id, key, value_type, value_text) VALUES (?, ?, ?, ?)',
                                    (tensor_id, key, 'text', str(value)))
        
        self.conn.commit()


    def query_tensors(self, **tags: Any) -> list[dict[str, Any]]:
        query: str = '''
            SELECT DISTINCT t.id, t.filename 
            FROM tensors t
        '''
        conditions: list[str] = []
        values: list[Any] = []
        
        for i, (key, value) in enumerate(tags.items()):
            query += f' JOIN tags tag{i} ON t.id = tag{i}.tensor_id '
            if isinstance(value, int):
                conditions.append(f"(tag{i}.key = ? AND tag{i}.value_type = 'int' AND tag{i}.value_int = ?)")
            elif isinstance(value, float):
                conditions.append(f"(tag{i}.key = ? AND tag{i}.value_type = 'float' AND tag{i}.value_real = ?)")
            else:
                conditions.append(f"(tag{i}.key = ? AND tag{i}.value_type = 'text' AND tag{i}.value_text = ?)")
            values.extend([key, value])
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        self.cursor.execute(query, values)
        results: list[tuple[int, str]] = self.cursor.fetchall()
        
        output: list[dict[str, Any]] = []
        for tensor_id, filename in results:
            tensor = torch.load(os.path.join(self.tensor_dir, filename))
            tags = self.get_all_tags(tensor_id)
            output.append({
                "tags": tags,
                "tensor": tensor
            })
        
        return output


    def query_last(self, **tags: Any) -> dict[str, Any]:
        return self.query_tensors(**tags)[-1]
    

    def remove_tensor(self, id: int) -> None:
        self.cursor.execute('''
            DELETE FROM tags
            WHERE tensor_id = ?
        ''', (id,))

        self.cursor.execute('''
            SELECT filename
            FROM tensors
            WHERE id = ?
        ''', (id,))

        tensor_filename: str = self.cursor.fetchone()[0]
        os.remove(os.path.join(self.tensor_dir, tensor_filename))
        
        self.cursor.execute('''
            DELETE FROM tensors
            WHERE id = ?
        ''', (id,))
        self.conn.commit()


    def get_all_tags(self, tensor_id: int) -> dict[str, Any]:
        self.cursor.execute('''
            SELECT key, value_type, value_int, value_real, value_text
            FROM tags
            WHERE tensor_id = ?
        ''', (tensor_id,))
        results = self.cursor.fetchall()
        
        tags = {'id': tensor_id}  # Include the tensor ID in the tags
        for key, value_type, value_int, value_real, value_text in results:
            if value_type == 'int':
                tags[key] = value_int
            elif value_type == 'float':
                tags[key] = value_real
            else:
                tags[key] = value_text
        return tags


    def close(self) -> None:
        self.conn.close()


    def print_all_data(self) -> None:
        print("All tensors:")
        self.cursor.execute("SELECT * FROM tensors")
        tensors = self.cursor.fetchall()
        for tensor in tensors:
            print(tensor)
        
        print("\nAll tags:")
        self.cursor.execute("SELECT * FROM tags")
        tags = self.cursor.fetchall()
        for tag in tags:
            print(tuple([item for item in tag if item is not None]))


# Example usage
if __name__ == "__main__":
    db = TensorDatabase("test_tensor_db.sqlite", "test_tensors")
    
    # Add a tensor
    tensor: torch.Tensor = torch.rand(1024, 2049)
    tags: dict[str, Any] = {
        "metric": "kl-div",
        "ngram": 3,
        "model": "EleutherAI/pythia-14m",
        "checkpoint": -6,
    }
    db.add_tensor(tensor, tags)
    
    # Query tensors
    results: list[dict[str, Any]] = db.query_tensors(metric="kl-div", model="EleutherAI/pythia-14m", checkpoint=-6)
    assert len(results) == 1

    # Remove tensor
    id = results[0]['tags']['id']
    db.remove_tensor(id)

    # Query tensors
    results: list[dict[str, Any]] = db.query_tensors(metric="kl-div", model="EleutherAI/pythia-14m", checkpoint=-6)

    assert len(results) == 0

    db.close()