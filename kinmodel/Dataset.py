"""Defines the Dataset class.

"""
import string
import numpy as np


class Dataset:
    def __init__(self, name="", times=None, concs=None):
        self.name = name
        self.times = times
        self.concs = concs

    @property
    def total_data_points(self):
        return self.concs.size - np.isnan(self.concs).sum()

    @property
    def num_times(self):
        return len(self.times)

    @property
    def max_time(self):
        return max(self.times)

    @classmethod
    def read_raw_data(cls, model, data_filename) -> ['Dataset']:
        """Load data from file, formated as a csv file.

        File is assumed to have the following structure:
            - Rows with only the first cell filled with a string (not
              interpretable as a number) and the remaining cells empty
              are titles for new datasets, which follow. Each experiment
              must have a title after the first.
            - All rows with the first column interpretable as a number
              are assumed to contain data.
            - All other rows ignored.

        """

        def _is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        with open(data_filename) as datafile:
            datasets = [cls()]
            all_times = []
            all_concs = []
            curr_ds_times = []
            curr_ds_concs = []
            for line in datafile:
                curline = line.replace("\n", "").split(",")
                if _is_number(curline[0]):
                    # Line contains data
                    curr_ds_times.append(float(curline[0]))
                    line_concs = []
                    for n in range(model.num_data_concs):
                        if n+1 < len(curline):
                            if curline[n+1] != "":
                                line_concs.append(float(curline[n+1]))
                            else:
                                line_concs.append(np.nan)
                        else:
                            line_concs.append(np.nan)
                    curr_ds_concs.append(line_concs)
                elif curline[0] != '' and curline[1:] == ['']*(len(curline)-1):
                    # Line contains dataset name
                    if curr_ds_times:
                        # A dataset already exists, move on to next one
                        all_times.append(curr_ds_times)
                        all_concs.append(curr_ds_concs)
                        curr_ds_times = []
                        curr_ds_concs = []
                        datasets.append(cls())
                        datasets[-1].name = "".join(
                                c for c in curline[0] if c in string.printable)
                    else:
                        # This is the first dataset
                        datasets[-1].name = "".join(
                                c for c in curline[0] if c in string.printable)
            # Record times for last dataset
            all_times.append(curr_ds_times)
            all_concs.append(curr_ds_concs)

            # Sort and store data for all datasets
            for s in range(len(datasets)):
                datasets[s].times = np.array(all_times[s])
                unsorted_data = np.array(all_concs[s])
                sorted_data = np.empty_like(unsorted_data)
                for n in range(model.num_data_concs):
                    sorted_data[:, n] = unsorted_data[:, model.sort_order[n]]
                datasets[s].concs = sorted_data

        return datasets
