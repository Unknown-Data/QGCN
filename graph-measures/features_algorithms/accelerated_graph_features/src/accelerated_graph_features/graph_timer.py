from datetime import datetime
from os.path import exists


class FeatureTimer:
    """
    The FeatureTimer class is intended to be used to benchmark the C++ graph features in comparison to the Python
    Features.
    The timer can save several time points in the middle of the run, enabling us to measure the time for the overhead
    and the actual feature calculation separately.

    Note
    ^^^^
    The same instance of FeatureTimer can be used for multiple runs, where all runs use the same interim points
    (i.e. the titles are the same for all runs).

    Terms
    ^^^^^
    Start - when the id is given and the run begins.

    Mark - a point in the middle of the run where we want to save the time until and from now.

    Stop - the end of the run, where the times of the run are calculated and written to the file.


    Example Usage
    ^^^^^^^^^^^^^

    >>> t = FeatureTimer('example_feature_times',['Conversion Time','Feature calculation time'])
    >>> t.start('example_50_nodes_100_edges')
    >>> # Conversion routine
    >>> t.mark()
    >>> # Feature calculation
    >>> t.stop()
    >>> # Now we can use the same instance for a new run
    >>> t.start('example_200_nodes_1000_edges')
    >>> # Conversion routine
    >>> t.mark()
    >>> # Feature calculation
    >>> t.stop()


    """

    def __init__(self, save_file_name, titles=None, num_of_interim_stops=0):
        """

        :param save_file_name: the csv file to save to
        :param titles: a list of titles for the csv file, if not given, defaults to generic names.
        :param num_of_interim_stops: the number of marks in the middle of the run.

        Note
        ^^^^
        Either titles or num_of_interim_stops must be given. If both are given and the lengths do not match,
        the length of titles overrides num_of_interim_stops.

        """
        if '.csv' not in save_file_name:
            save_file_name += '.csv'

        self.file_name = save_file_name
        self.num_of_interim_stops = num_of_interim_stops

        if titles is not None:
            if len(titles) != num_of_interim_stops + 1:
                self.num_of_interim_stops = len(titles) - 1
        else:
            titles = ['run id'] + ['mark {}'.format(i) for i in range(1, num_of_interim_stops + 1)] + ['end']

        self.titles = ['run id'] + titles
        self.times = []
        self.run_id = ''

    def start(self, run_id):
        """
        Start running the timer.

        :param run_id: the id of the current run, to be used in the results csv

        """
        self.run_id = run_id
        self._clear()
        self.mark()

    def mark(self):
        """
        Mark this point as a midpoint for the time measurements, essentially starting a new section of the run.

        """
        self.times.append(datetime.now())

    def stop(self):
        """
        End the timer's run.

        This function also writes the run's results to the csv file.


        """
        self.mark()
        self._write()
        self._clear()
        pass

    def _clear(self):
        self.times = []

    def _write(self):
        if not exists(self.file_name):
            self._init_save_file()
        time_diffs = []

        for i in range(len(self.times) - 1):
            begin_time: datetime = self.times[i]
            end_time: datetime = self.times[i + 1]
            # save the time difference in milliseconds
            diff = (end_time - begin_time).total_seconds() * 10 ** 6
            time_diffs.append(diff)

        if len(time_diffs) != len(self.titles) - 1:
            raise AssertionError(
                "There should be {} marks, {} were found".format(len(self.titles) - 1, len(time_diffs)))

        with open(self.file_name, 'a') as f:
            current_row = self.run_id + ',' + ','.join([str(x) for x in time_diffs])
            f.write(current_row + '\n')

    def _init_save_file(self):
        with open(self.file_name, 'w+') as f:
            f.write(','.join(self.titles) + '\n')
