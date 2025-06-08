import torch
import subprocess
import re

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total',
         '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # Parse the output
    gpu_memory = {}
    for i, line in enumerate(result.strip().split('\n')):
        free_mem, total_mem = map(int, line.split(','))
        gpu_memory[i] = free_mem

    return gpu_memory

def get_device_with_most_free_memory():
    """Get the CUDA device with the most free memory.
    Returns
    -------
    device: torch.device
        The CUDA device with the most free memory
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    gpu_memory = get_gpu_memory_map()
    if not gpu_memory:
        return torch.device('cpu')

    # Get the device with most free memory
    device_id = max(gpu_memory.items(), key=lambda x: x[1])[0]
    return torch.device(f'cuda:{device_id}')

def set_device_with_most_free_memory():
    """Set the CUDA device with the most free memory as the current device.
    Returns
    -------
    device: torch.device
        The CUDA device that was set
    """
    device = get_device_with_most_free_memory()
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device
