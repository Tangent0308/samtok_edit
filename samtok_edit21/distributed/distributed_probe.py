"""Check all-rank NCCL collectives before expensive model loading."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from samtok_edit21.distributed.cluster import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Every worker must expose exactly eight GPUs")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    value = torch.tensor([rank + 1.0], device=f"cuda:{local}")
    dist.all_reduce(value)
    assert value.item() == world * (world + 1) / 2
    # Exercise a larger payload as well as the rendezvous/control path.
    payload = torch.full((1024 * 1024,), float(rank), device=f"cuda:{local}")
    dist.broadcast(payload, 0)
    assert payload.count_nonzero().item() == 0
    atomic_json(Path(args.output)/f"rank{rank}.json", {
        "rank": rank, "local_rank": local, "world_size": world, "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(local), "all_reduce": value.item(), "broadcast": "passed"})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
