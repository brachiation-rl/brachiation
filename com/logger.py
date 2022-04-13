import os
import csv
import numpy as np


class CSVLogger(object):
    def __init__(self, log_dir, filename="progress.csv"):
        self.csvfile = open(os.path.join(log_dir, filename), "w")
        self.writer = None

    def init_writer(self, keys):
        if self.writer is None:
            self.writer = csv.DictWriter(self.csvfile, fieldnames=list(keys))
            self.writer.writeheader()

    def log_epoch(self, data):
        if "stats" in data:
            for key, values in data["stats"].items():
                data["mean_" + key] = float(sum(values) / len(values))
                data["min_" + key] = float(min(values))
                data["max_" + key] = float(max(values))
        del data["stats"]

        self.init_writer(data.keys())
        self.writer.writerow(data)
        self.csvfile.flush()

    def __del__(self):
        self.csvfile.close()
