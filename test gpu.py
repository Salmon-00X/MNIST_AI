import tensorflow as tf
from typing import Optional, Tuple

# Print TensorFlow version and GPU devices
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

def check_gpu_availability(
    cuda_only: bool = False,
    min_cuda_compute_capability: Optional[Tuple[int, int]] = None
) -> bool:
    
    return tf.test.is_gpu_available(
        cuda_only=cuda_only,
        min_cuda_compute_capability=min_cuda_compute_capability
    )

# Example usage
if __name__ == "__main__":
    print("Basic GPU check:", check_gpu_availability())  # True or False