# Calicrate the GPU contention generator on AGX Xavier
# Usage: python3 recalibrate.py

import time
import subprocess
import numpy
import math
import threading
import psutil
import os
from numba import cuda

class GPUContentionGenerator:
    def __init__(self, initial_level=512, cpu_cores=[0]):
        self.level = initial_level
        try:
            self.device = cuda.select_device(0)  # Select CUDA device
        except Exception as e:
            print(f"Failed to select CUDA device: {e}")
            raise
        self.contention_running = False  # Flag to control contention
        self.tegra_process = None  # To store tegrastats subprocess
        # Set CPU affinity during initialization
        #self.set_cpu_affinity(cpu_cores)
    
    def set_cpu_affinity(self, core_ids):
        """Sets the CPU affinity for this process."""
        pid = os.getpid()  # Get the current process ID
        try:
            psutil.Process(pid).cpu_affinity(core_ids)  # Set CPU affinity
            print(f"Set CPU affinity to cores: {core_ids}")
        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")
    
    @staticmethod
    @cuda.jit
    def contention_kernel(array):
        pos = cuda.grid(1)
        if pos < array.size:
            array[pos] += math.sin(pos) * math.cos(pos)

    # def contention_kernel(array):
    #     pos = cuda.grid(1)
    #     tx = cuda.threadIdx.x 
    #     if pos < array.size:
    #         array[pos] += tx  # Element add computation
    
    def start_contention(self):
        if self.level <= 0:
            print("Contention level must be greater than zero.")
            return
        
        ContentionSize = int(self.level)
        data = numpy.random.rand(ContentionSize)
        multiplier = data.size / 512
        threadsperblock, blockspergrid = 128, 4

        # copy data to device
        device_data = cuda.to_device(data)

        self.contention_running = True  # Set flag to indicate contention is running
        while self.contention_running:
            self.contention_kernel[math.ceil(multiplier * blockspergrid), threadsperblock](device_data)
    
    def stop_contention(self):
        self.contention_running = False  # Stop the contention loop
        print("Contention has been stopped.")

    def decay_factor(self, step, total_steps, initial_factor=1.1, final_factor=1):
        factor = initial_factor * (final_factor / initial_factor) ** (step / total_steps)
        return factor

    def run_profiling(self):
        try:
            # Start profiling the GPU module usage under different workload levels
            total_steps = 60
            for step in range(total_steps):
                print('Workload Level:', self.level)

                # Save the output from tegrastats
                tegra_cmd = f'tegrastats --logfile logs/log_tegrastats_{self.level}.txt'
                self.tegra_process = subprocess.Popen(tegra_cmd, shell=True)

                # Start the GPU contention generator in a separate thread
                contention_thread = threading.Thread(target=self.start_contention)
                contention_thread.start()

                # Run for 60 seconds
                time.sleep(60)

                # Stop the GPU contention
                self.stop_contention()

                # Wait for the contention thread to finish
                contention_thread.join()

                # Stop tegrastats
                cmd = "tegrastats --stop"
                p = subprocess.Popen(cmd, shell = True)
                output = p.communicate()[0]

                # Increase the workload level for the next round
                factor = self.decay_factor(step, total_steps, initial_factor=1.1, final_factor=1)
                self.level = int(self.level * factor)

        except KeyboardInterrupt:
            print("Interrupted by user. Cleaning up...")
            self.stop_contention()  # Stop contention
            # Optionally, you can wait for the thread to finish here if it's running
            if contention_thread.is_alive():
                contention_thread.join()  # Wait for thread to finish if it's still running
            if self.tegra_process:  # Ensure tegrastats subprocess is terminated
                # Stop tegrastats
                cmd = "tegrastats --stop"
                p = subprocess.Popen(cmd, shell = True)
                output = p.communicate()[0]
            print("Finsih cleaning up...")
    
def main():
    # Create an instance of the generator
    gpu_contention_gen = GPUContentionGenerator(initial_level=4096, cpu_cores=[11])
    gpu_contention_gen.run_profiling()

if __name__ == "__main__":
    main()
