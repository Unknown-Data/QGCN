from _datetime import datetime
from loggers import PrintLogger
from enum import Enum
import pandas as pd
from datetime import timedelta


from multi_graph import MultiGraph


class TemporalGraph:
    def __init__(self, database_name, csv_source, time_format, time_col, src_col, dst_col, weight_col=None,
                 weeks=0, days=0, hours=0, minutes=0, seconds=0, logger=None, time_format_out=None, directed=False):
        self._csv_source = csv_source
        self._database_name = database_name
        self._directed = directed
        self._time_format = time_format
        self._time_col = time_col
        self._src_col = src_col
        self._dst_col = dst_col
        self._weight_col = weight_col
        self._format_out = time_format_out if time_format_out else time_format
        self._timedelta = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
        self._time_col = time_col
        self._src_col = src_col
        self._logger = logger if logger else PrintLogger()
        self._mg_dictionary = self._break_by_time()

    def to_multi_graph(self):
        mg = MultiGraph(self._database_name, graphs_source=self._mg_dictionary, directed=self._directed)
        mg.sort_by(lambda x: datetime.strptime(x, self._format_out))
        return mg

    def _break_by_time(self):
        dict_time_edge_list = {}

        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._csv_source)
        self._format_data(data_df)
        # make TIME column the index column and sort data by it
        # data_df = data_df.set_index(self._time_col).sort_index()
        data_df = data_df.sort_values(self._time_col)

        # first day is the first row
        curr_time = data_df[self._time_col][data_df.first_valid_index()]
        # file_time = open(os.path.join(OUT_TMP_PATH, curr_day.date.strftime(format_out)), "wt")
        # next day is the  floor(<current_day>) + <one_time_interval>
        prev_time = curr_time
        next_time = prev_time + self._timedelta
        for index, row in data_df.iterrows():
            curr_time = data_df[self._time_col][index]
            # if curr day is in next time interval
            if curr_time >= next_time:
                while curr_time >= next_time:
                    prev_time = next_time
                    next_time = prev_time + self._timedelta

            # write edge to file
            src = str(row[self._src_col])
            dst = str(row[self._dst_col])
            weight = float(row[self._weight_col]) if self._weight_col else 1
            key = prev_time.strftime(self._format_out)
            if key not in dict_time_edge_list:
                dict_time_edge_list[key] = []
            dict_time_edge_list[key].append((src, dst, weight))
        return dict_time_edge_list

    def _format_data(self, graph_df):
        if self._time_format == "numerate":
            return
        if self._time_format == "MIL":
            graph_df[self._time_col] = graph_df[self._time_col] / 1000  # mili to seconds
        if self._time_format == "MIL" or format == "SEC":
            graph_df[self._time_col] = \
                graph_df[self._time_col].apply(lambda x: datetime.fromtimestamp(x))  # to datetime format
        else:
            # to datetime
            graph_df[self._time_col] = graph_df[self._time_col].apply(lambda x: datetime.strptime(str(x), self._time_format))


if __name__ == "__main__":
    pd.set_option('display.max_columns', 2)
    pd.set_option('display.max_rows', 2)
    pd.set_option('display.width', 1000)

    tg = TemporalGraph("FireWall", "Firewall-04062012.csv", '%d/%b/%Y %H%M:%S', "Date/time", "Source IP",
                       "Destination IP", minutes=2)
    e = 0
