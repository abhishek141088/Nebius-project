"""Minimal 2-node NCCL all_reduce to produce concrete evidence of real
inter-node GPU communication (not just two independent single-node jobs).
Run with NCCL_DEBUG=INFO so NCCL itself logs which channel/transport it
used between the two nodes.
"""
import sys

sys.path.insert(0, "/mnt/data/assignment/pylibs")

import socket  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    tensor = torch.full((10,), float(rank), device="cuda")
    print(f"[{hostname}] rank {rank}/{world_size}: before all_reduce = {tensor[0].item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

    print(f"[{hostname}] rank {rank}/{world_size}: after all_reduce = {tensor[0].item()}")
    expected = sum(range(world_size))
    assert tensor[0].item() == expected, f"expected {expected}, got {tensor[0].item()}"
    print(f"[{hostname}] rank {rank}: all_reduce correct across {world_size} ranks")


if __name__ == "__main__":
    main()
