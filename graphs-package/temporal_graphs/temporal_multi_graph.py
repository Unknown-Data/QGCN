import copy
from datetime import timedelta, datetime
from loggers import PrintLogger
import pandas as pd
from datetime import timedelta
from multi_graph import MultiGraph
from collections import Counter
import os


class TemporalMultiGraph:
    def __init__(self, database_name, csv_source, time_format, time_col, src_col, dst_col, subgraph_name_col,
                 weight_col=None, label_col=None, weeks=0, days=0, hours=0, minutes=0, seconds=0, logger=None,
                 time_format_out=None, directed=False):
        self._labels = {}
        self._order = {}
        self._times = []
        self._times_index = {}
        self._graphs_for_time = {}
        self._subgraph_name_col = subgraph_name_col
        self._csv_source = csv_source
        self._database_name = database_name
        self._directed = directed
        self._time_format = time_format
        self._time_col = time_col
        self._src_col = src_col
        self._dst_col = dst_col
        self._weight_col = weight_col
        self._label_col = label_col
        self._format_out = self._time_format_out(time_format_out)
        self._timedelta = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
        self._time_col = time_col
        self._src_col = src_col
        self._logger = logger if logger else PrintLogger()
        self._edge_list_dict = self._break_by_time()
        self._mg_dict = self._build_multi_graphs()
        self._number_of_times = len(self._edge_list_dict)

    @property
    def labels(self):
        return self._labels

    def _time_format_out(self, format):
        if format:
            return format
        if self._time_format == "MIL" or self._time_format == "SEC":
            return '%d_%b_%Y %H_%M_%S'
        return self._time_format

    def _break_by_time(self):
        index_order = 0
        time_index = 0
        dict_time_to_mg_dict = {}

        # open basic data csv (with all edges of all times)
        data_df = pd.read_csv(self._csv_source)
        self._format_data(data_df)
        # make TIME column the index column and sort data by it
        data_df = data_df.sort_values(self._time_col)

        # first day is the first row
        curr_time = data_df[self._time_col][data_df.first_valid_index()]
        # file_time = open(os.path.join(OUT_TMP_PATH, curr_day.date.strftime(format_out)), "wt")
        curr_time_mg_dict = {}
        # next day is the  floor(<current_day>) + <one_time_interval>
        prev_time = curr_time
        next_time = prev_time + self._timedelta
        for index, row in data_df.iterrows():
            curr_time = data_df[self._time_col][index]
            # if curr day is in next time interval
            if curr_time >= next_time:
                # save current time multi-graph_dictionary
                dict_time_to_mg_dict[prev_time.strftime(self._format_out)] = curr_time_mg_dict
                curr_time_mg_dict = {}
                self._times.append(prev_time.strftime(self._format_out))
                self._times_index[prev_time.strftime(self._format_out)] = time_index
                time_index += 1
                while curr_time >= next_time:
                    prev_time = next_time
                    next_time = prev_time + self._timedelta

            # write edge
            graph_name = str(row[self._subgraph_name_col])
            src = str(row[self._src_col])
            dst = str(row[self._dst_col])
            weight = float(row[self._weight_col]) if self._weight_col else 1
            if graph_name not in curr_time_mg_dict:
                curr_time_mg_dict[graph_name] = []
            curr_time_mg_dict[graph_name].append((src, dst, weight))
            # keep order of graphs
            if graph_name not in self._order:
                self._order[graph_name] = index_order
                index_order += 1
            if prev_time.strftime(self._format_out) not in self._graphs_for_time:
                self._graphs_for_time[prev_time.strftime(self._format_out)] = []
            if graph_name not in self._graphs_for_time[prev_time.strftime(self._format_out)]:
                self._graphs_for_time[prev_time.strftime(self._format_out)].append(graph_name)
            # add label
            if self._label_col:
                self._labels[graph_name] = self._labels.get(graph_name, []) + [row[self._label_col]]
        # fix labels to br the most common
        for name, labels_list in self._labels.items():
            self._labels[name] = Counter(labels_list).most_common()[0][0]
        return dict_time_to_mg_dict

    def _build_multi_graphs(self):
        time_mg_dict = {}
        for time, mg_dict in self._edge_list_dict.items():
            # create multi graph for each time and order the subgraphs according to the order dictionary (FIFO)
            mg = MultiGraph(time, graphs_source=mg_dict, directed=self._directed)
            mg.sort_by(self._func_order)
            time_mg_dict[time] = mg
        return time_mg_dict

    def _func_order(self, graph_name):
        return self._order[graph_name]

    def _format_data(self, graph_df):
        if self._time_format == "MIL":
            graph_df[self._time_col] = graph_df[self._time_col] / 1000  # mili to seconds
        if self._time_format == "MIL" or format == "SEC":
            graph_df[self._time_col] = \
                graph_df[self._time_col].apply(lambda x: datetime.fromtimestamp(x))  # to datetime format
        else:
            # to datetime
            graph_df[self._time_col] = graph_df[self._time_col].apply(lambda x: datetime.strptime(x, self._time_format))

    def multi_graphs_by_time(self, start=None, end=None):
        start = start if start else 0
        stop = end if end else self._number_of_times
        for i in range(start, stop):
            yield self._mg_dict[self._times[i]]

    def multi_graph_by_window(self, window_size=None, start_time=0):
        if start_time < 0 or start_time > self._number_of_times:
            self._logger.error("invalid start time = " + str(start_time) + ", total intervals = " + str(self._number_of_times))
            return
        # build base mg
        mg = MultiGraph(self._database_name + "window", graphs_source=self._edge_list_dict[self._times[0]],
                        directed=self._directed)
        for i in range(1, start_time):
            mg.add_edges(self._edge_list_dict[self._times[i]])

        window_size = window_size if window_size else self._number_of_times
        for i in range(start_time, self._number_of_times):
            mg.suspend_logger()
            temp = copy.deepcopy(mg)
            mg.wake_logger()
            yield temp

            to_remove = i - window_size
            if to_remove >= 0:
                mg.remove_edges(self._edge_list_dict[self._times[to_remove]])
            mg.add_edges(self._edge_list_dict[self._times[i]])


if __name__ == "__main__":
    _params = {
        'logger_name': "logger",
        # Data parameters
        'days_split': 1,
        'start_time': 10,
        'window_size': None,
        'database': 'Refael',
        'data_file_name': 'Refael_07_18.csv',  # should be in ../data/
        'date_format': "%Y-%m-%d",  # Refael
        'directed': True,
        'white_label': 1,
        # graph_measures + beta vectors parameters
        'max_connected': False,
        # ML- parameters
        'min_nodes': 10,
        # AL - parameters
        'batch_size': 2,
        'queries_per_time': 2,
        'eps': 0.01,
        'target_recall': 0.7,
        'reveal_target': 0.6,
        'ml_method': "XG_Boost"
    }
    _database_ = TemporalMultiGraph(_params['database'],
                                    _params['data_file_name'],
                                    time_format='MIL',
                                    time_col='StartTime',
                                    src_col='SourceID',
                                    dst_col='DestinationID',
                                    label_col='target',
                                    subgraph_name_col='Community',
                                    days=_params['days_split'],
                                    time_format_out=_params['date_format'],
                                    directed=_params['directed'])
    for multi_graph in _database_.multi_graph_by_window(_params['window_size'], _params['start_time']):
        None
    e = 0
