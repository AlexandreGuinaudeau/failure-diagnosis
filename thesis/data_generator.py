import subprocess
import re
import os
import time
import logging
from threading import Thread
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def pretty_print(d, tabs=0):
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            print("  " * tabs + k)
            pretty_print(d[k], tabs+1)
    else:
        print("  " * tabs + str(d))


class _Generator(Thread):
    def __init__(self, fn):
        Thread.__init__(self)
        self.fn = fn

    def run(self):
        return self.fn()


class DataGenerator:
    def __init__(self, duration=10, float_operations=100000, memory=100000000, **kwargs):
        # self.tmp_dir = os.path.join(os.getcwd(), "data", "tmp")
        # if not os.path.exists(self.tmp_dir):
        #     os.mkdir(self.tmp_dir)
        # self.tmp_file = os.tmpfile()
        # with open(self.tmp_file, "w") as in_f:
        #     in_f.write(str(np.random.randn(10000)))
        self._stop_path = os.path.join(os.getcwd(), "_stop_generator.txt")
        self.float_operations = float_operations
        self.duration = duration
        self.memory = memory
        self.threads = {}
        self.kwargs = dict(kwargs)
        self.logger = logging.getLogger("DataGenerator")
        self.logger.setLevel(logging.DEBUG)
        self.stats = {}
        self.threads["stats"] = [_Generator(self.get_stats_forever)]
        self.threads["stats"][0].start()

    def _is_stopped(self):
        return os.path.exists(self._stop_path)

    def shutdown(self):
        with open(self._stop_path, "w"):
            pass
        self.join_all()
        os.remove(self._stop_path)

    def join_all(self):
        for kind in self.threads.keys():
            for t in self.threads[kind]:
                t.join()
            self.threads.pop(kind)

    def join_cpu(self):
        for t in self.threads["cpu"]:
            t.join()
        self.threads["cpu"] = []

    def run_cpu(self, n=10):
        """
        Multi-thread process to let the CPU reach ~90%
        """
        def f(cycle=None, v=1):
            if cycle == 0:
                self.logger.info(str(v))
                return v
            if cycle is None:
                cycle = self.duration
            for j in range(self.float_operations):
                v += j*v
                v = v % 42398
                v += 34
            return f(cycle-1, v)

        start = time.time()
        self.threads["cpu"] = []
        for i in range(n):
            self.threads["cpu"].append(_Generator(f))
            self.threads["cpu"][-1].start()
        end = time.time()
        self.logger.warning("run_cpu(%i) : %f s" % (n, end-start))

        return True

    def run_memory(self, n=10):
        """
        A process that will apply for a nearly 2G memory space
        """
        def f():
            l = []
            for i in range(n):
                l.append(np.random.randn(self.memory))
            print(len(l[-1]))

        self.threads["mem"] = _Generator(f)
        self.threads["mem"].start()
        self.threads["mem"].join()
        return True

    def run_disk(self):
        """
        A copy files process which can sharply increase disk transfer rate
        """

    def get_stats(self):
        """
        istats --no-graphs
        """
        res = subprocess.check_output(["istats", "--no-graphs"]).split("\n")
        d = {}
        category = None
        for line in res:
            line = str(line)
            if line in ('', 'For more stats run `istats extra` and follow the instructions.'):
                continue
            if line.startswith("---"):
                category = line[4:-4].strip()
                d[category] = {}
            else:
                title, value = line.split(":")
                numbers = re.findall("\d+\.?\d*", value.strip())
                units = re.findall("[^\d.]+", value.strip())
                if len(units) < len(numbers):
                    units.append("")
                for i, n in enumerate(numbers):
                    d[category]["%s (%s)" % (title.strip(), units[i].strip())] = float(numbers[i].strip())
        for c in d.keys():
            if c not in self.stats.keys():
                self.stats[c] = {}
            for t in d[c].keys():
                if t not in self.stats[c].keys():
                    self.stats[c][t] = []
                self.stats[c][t].append(d[c][t])
        return d

    def get_stats_forever(self, seconds=1):
        while not self._is_stopped():
            self.get_stats()
            time.sleep(seconds)


if __name__ == "__main__":
    d_ = DataGenerator()
    time.sleep(5)
    d_.shutdown()
    pretty_print(d_.stats)
