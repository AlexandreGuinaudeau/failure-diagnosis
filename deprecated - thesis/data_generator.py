import subprocess
import re
import os
import psutil
import time
import logging
from threading import Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


def pretty_print(d, tabs=0):
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            print("  " * tabs + k)
            pretty_print(d[k], tabs+1)
    else:
        print("  " * tabs + str(d))


def get_all_stats():
    def _pretty_print_stats(name, stat_dict, unit_dict=None):
        unit_dict = unit_dict or {}
        s = ""
        for k, v in stat_dict.items():
            if k in unit_dict.keys():
                unit = unit_dict[k]
            elif name in unit_dict.keys():
                unit = unit_dict[name]
            else:
                unit = ""
            s += "%s_%s:%s%s%s\n" % (name, k, " " * (23 - len(name) - len(k)), v, unit)
        return s

    stats = ""
    stats += "--- Process Stats ---\n"
    stats += "Processes running: %i\n" % len(psutil.pids())

    stats += "\n--- Memory Stats ---\n"
    mem_dict = {"virtual": psutil.virtual_memory().__dict__,
                "swap": psutil.swap_memory().__dict__}
    unit_dict = {"percent": "%"}
    for k, d in mem_dict.items():
        stats += _pretty_print_stats(k, d, unit_dict)

    stats += "\n--- CPU Stats ---\n"
    cpu_dict = {"times": psutil.cpu_times().__dict__,
                "times_percent": psutil.cpu_times_percent().__dict__,
                "stats": psutil.cpu_stats().__dict__}
    unit_dict = {"times_percent": "%"}
    for k, d in cpu_dict.items():
        stats += _pretty_print_stats(k, d, unit_dict)
    stats += "cpu_percent:             %s%%\n" % psutil.cpu_percent()
    stats += "cpu_count:               %s%%\n" % psutil.cpu_count()

    stats += "\n--- Network Stats ---\n"
    stats += _pretty_print_stats("net_io_counters", psutil.net_io_counters().__dict__)

    stats += "\n--- Disk Stats ---\n"
    stats += _pretty_print_stats("disk_io_counters", psutil.disk_io_counters().__dict__)
    stats += "\n"

    stats += subprocess.check_output(["istats", "--no-graphs"])
    return stats


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
        self.threads = {"stats": [], "cpu": [], "mem": [], "disk": []}
        self.kwargs = dict(kwargs)
        self.logger = logging.getLogger("DataGenerator")
        self.logger.setLevel(logging.DEBUG)
        self.stats = {}
        self.stats_path = os.path.realpath(os.path.join(__file__, "..", "stats"))
        self.get_stats_forever()

    def get_stats_forever(self):
        self.threads["stats"] = [_Generator(self._get_stats_forever)]
        self.threads["stats"][0].start()

    def _is_stopped(self):
        return os.path.exists(self._stop_path)

    def cleanup(self):
        self.shutdown()
        for subdir in ["display", "logs"]:
            dir_path = os.path.join(self.stats_path, subdir)
            tmp_dir_path = os.path.join(self.stats_path, "_" + subdir)
            os.rename(dir_path, tmp_dir_path)
            os.mkdir(dir_path)
            for name in os.listdir(tmp_dir_path):
                os.remove(os.path.join(tmp_dir_path, name))
            os.rmdir(tmp_dir_path)

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

        self.threads["mem"].append(_Generator(f))
        self.threads["mem"][-1].start()
        return True

    def run_disk(self):
        """
        A copy files process which can sharply increase disk transfer rate
        """
        def f():
            tmp_dir = os.path.realpath(os.path.join(__file__, "..", "tmp"))
            return

        self.threads["disk"].append(_Generator(f))
        self.threads["disk"][-1].start()
        return True


    def get_stats(self, save_as_files):
        """
        istats --no-graphs
        """
        res = get_all_stats().split("\n")
        d = {}
        category = None
        for line in res:
            line = str(line)
            if line in ('', 'For more stats run `istats extra` and follow the instructions.'):
                continue
            if line.startswith("---"):
                category = line[4:-4].strip()
                if category not in d.keys():
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
            if save_as_files:
                for t in d[c].keys():
                    filename = re.sub("\W", "", re.sub("\s", "_", c.lower() + "-" + t.lower())) + ".csv"
                    path = os.path.join(self.stats_path, "logs", filename)
                    if not os.path.exists(path):
                        open(path, "w").close()
                    with open(path, "a") as in_f:
                        in_f.write("%f, %s\n" % (time.time(), d[c][t]))
            else:
                if c not in self.stats.keys():
                    self.stats[c] = {}
                for t in d[c].keys():
                    if t not in self.stats[c].keys():
                        self.stats[c][t] = []
                    self.stats[c][t].append(d[c][t])
        return d

    def _get_stats_forever(self, seconds=1, save_as_files=True):
        while not self._is_stopped():
            self.get_stats(save_as_files)
            time.sleep(seconds)
        print "Stopped getting stats."

    def display_stats(self):
        for name in os.listdir(os.path.join(self.stats_path, "logs")):
            file_name = os.path.join(self.stats_path, "logs", name)
            df = pd.read_csv(file_name, header=None)
            df.columns = ['time', name[:-4]]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.plot()
            plt.savefig(os.path.join(self.stats_path, "display", name[:-4] + ".png"))
            plt.clf()


if __name__ == "__main__":
    d_ = DataGenerator()
    d_.cleanup()
    time.sleep(5)
    d_.run_cpu()
    time.sleep(5)
    d_.display_stats()
    d_.shutdown()
    pretty_print(d_.stats)
