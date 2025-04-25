import json
from pathlib import Path
from typing import List, Optional

__all__ = ["workers", "endpoints"]

CONFIG_PATH = Path(__file__).parent / "node_config.json"

class WorkerConfig:
    def __init__(self, ip: str, port: int, device: Optional[str] = None):
        self.ip = ip
        self.port = port
        self.device = device if device else "Anonymous"

    @property
    def endpoint(self) -> str:
        return f"http://{self.ip}:{self.port}"

def load_worker_configs() -> List[WorkerConfig]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        configs = json.load(file)

    workers: List[WorkerConfig] = []
    for config in configs["workers"]:
        workers.append(WorkerConfig(**config))

    return workers


workers = load_worker_configs()
endpoints = [worker.endpoint for worker in workers]