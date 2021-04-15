import subprocess
import time
import csv
import os


__all__ = [
    "Profiler"
]


class Profiler:
    """
    Nvidia-smi based gpu memory profiler.
    Runs on start() nvidia-smi profiling in the separate process.

    This process sample memory usage with fixed step and saves it into temporary csv table.
    When user calls stop() profiler terminates subprocess and extracts peak memory usage estimated by profiler.

    How to use:
    >>> p = Profiler()
    >>> p.start()
    >>> # my_function_to_profile()
    >>> peak = p.stop() # This value show peak mem usage by your computation (in MiBs)
    """

    __slots__ = ["file_name", "params", "command", "file", "process", "data", "peak", "mem", "sampling_step"]

    def __init__(self, sampling_step_ms=1):
        assert int(sampling_step_ms) > 0

        self.file_name = "_mem_profile_nvidia_smi_.csv"
        self.params = "memory.total,memory.used"
        self.sampling_step = int(sampling_step_ms)
        self.command = ["nvidia-smi", f"--query-gpu={self.params}", "--format=csv", "-lms", f"{self.sampling_step}"]
        self.file = None
        self.process = None
        self.data = None
        self.peak = None
        self.mem = None

    def start(self):
        self.file = open(self.file_name, "w")
        self.process = subprocess.Popen(args=self.command, stdout=self.file)
        time.sleep(1)

    def stop(self):
        self.process.terminate()
        self.__load_mem_profile()
        self.__find_peak()
        return self.peak

    def __load_mem_profile(self):
        table_list = []

        # Extract table
        self.file = open(self.file_name, "r")
        reader = csv.reader(self.file, delimiter=",")
        next(reader)
        # print("Import table with header: ", header)
        for row in reader:
            table_list.append((int(row[0].replace("MiB", "")), int(row[1].replace("MiB", ""))))

        # Close file and remove it
        self.file.close()
        os.remove(self.file_name)
        self.data = table_list

    def __find_peak(self):
        min_mem = 1e10
        max_mem = 0

        for _, m in self.data:
            min_mem = min(min_mem, m)
            max_mem = max(max_mem, m)

        self.peak = max_mem - min_mem
        self.mem = {"max": max_mem, "min": min_mem, "usage peak": self.peak}

    @property
    def usage_peak(self):
        return self.peak

    @property
    def memory_stat(self):
        return self.mem


p = Profiler()
p.start()
p.stop()
print(p.memory_stat)
