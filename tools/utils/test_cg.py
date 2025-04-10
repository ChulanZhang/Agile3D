import time
import threading
from contention import GPUContentionGenerator

def main():
    gpu_util_to_level = {
        10: 4096,
        20: 8888,
        30: 15000,
        40: 21000,
        50: 26888,
        60: 33888,
        70: 39888,
        80: 45200,
        90: 52200,
        99: 66666}
    
    # Create an instance of the GPUContentionGenerator
    gpu_contention_gen = GPUContentionGenerator(initial_level=4096, cpu_cores=[11])

    # Loop through the GPU utilization levels and update the workload level
    for util, level in gpu_util_to_level.items():
        print(f"Setting GPU utilization to {util}% with level {level}")
        gpu_contention_gen.level = level  # Update the level
        # Start the GPU contention
        contention_thread = threading.Thread(target=gpu_contention_gen.start_contention)
        contention_thread.start()
        # Run for 60 seconds
        time.sleep(10)
        # Stop the GPU contention
        gpu_contention_gen.stop_contention()
        # Wait for the contention thread to finish
        contention_thread.join()

if __name__ == "__main__":
    main()
