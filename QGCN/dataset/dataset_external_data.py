"""
Given some extended data (for example features), this is the class that handles it. It is recommended to have external
data for your dataset but it is not a requirement.
"""

import json
import os
import pandas as pd
from copy import deepcopy

UNKNOWN_SYM = "__UNKNOWN__"


class ExternalData:
    def __init__(self, params, idx_to_symbol=None):
        # load the params file (json) in the "external" section.
        self._params = params["external"] if type(params) is dict else json.load(open(params, "rt"))["external"]
        # path to base directory of the project
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        # path to external data file

        # { ... graph_id: { node: [..value..] } ... }
        self._embed, self._continuous, self._idx_to_symbol, self._symbol_to_idx = self._read_file()

        # fix idx to sym and opposite
        self._idx_to_symbol = self._idx_to_symbol if idx_to_symbol is None else idx_to_symbol
        self._symbol_to_idx = self._symbol_to_idx if idx_to_symbol is None else  \
            {ftr: {symbol: i for i, symbol in enumerate(idx_to_symbol_list)}
             for ftr, idx_to_symbol_list in idx_to_symbol.items()}

    @property
    def idx_to_symbol_dict(self):
        return self._idx_to_symbol

    @property
    def is_embed(self):
        return False if self._embed is None else True

    @property
    def is_continuous(self):
        return False if self._continuous is None else True

    @property
    def embed_headers(self):
        return deepcopy(self._params["embeddings"])

    @property
    def continuous_headers(self):
        return deepcopy(self._params["continuous"])

    def embed_feature(self, g_id, node):
        # self._embed[g_id][node] = symbol_list of the node in graph
        return [self._symbol_to_idx[ftr].get(self._embed[g_id][node][i], self._symbol_to_idx[ftr][UNKNOWN_SYM])
                for i, ftr in enumerate(self._params["embeddings"])]

    # return number of symbols for specific embedding
    def len_embed(self, idx_str=None):
        if idx_str is None:
            lengths = []
            for ftr, symbol_list in self._idx_to_symbol.items():
                lengths.append(len(symbol_list))
            # how many feature that need to be embed
            return lengths
        ftr = self._params["embeddings"][idx_str] if type(idx_str) == int else idx_str
        return len(self._idx_to_symbol[ftr])

    def continuous_feature(self, g_id, node):
        return self._continuous[g_id][node]

    # read file into two dictionaries one for embedding values and one for rational values.
    def _read_file(self):
        idx_to_symbol = {ftr: [UNKNOWN_SYM] for ftr in self._params["embeddings"]}

        # when having values to embed or continuous values, save them in a dictionary
        embed_dict = {} if self._params["embeddings"] else None
        value_dict = {} if self._params["continuous"] else None

        # handle external data (several types of features)
        external_data_df = pd.read_csv(self._params["file_path"])
        # read all rows
        for index, data in external_data_df.iterrows():
            # extract data
            g_id = str(data[self._params["graph_col"]])
            node = str(data[self._params["node_col"]])

            # extract embeddings + keep list idx to embed symbol for each feature
            embed_list = []
            for ftr in self._params["embeddings"]:
                symbol = str(data[ftr])
                embed_list.append(symbol)
                if symbol not in idx_to_symbol[ftr]:
                    idx_to_symbol[ftr].append(symbol)

            # list of continuous features (no need in embedding)
            value_list = [data[ftr] for ftr in self._params["continuous"]]

            # add to dictionaries { ... graph_id: { node: [..list_value..] } ... }
            if embed_dict is not None:
                embed_dict[g_id] = embed_dict.get(g_id, {})
                embed_dict[g_id][node] = embed_list
            if value_dict is not None:
                value_dict[g_id] = value_dict.get(g_id, {})
                value_dict[g_id][node] = value_list

        symbol_to_idx = {ftr: {symbol: i for i, symbol in enumerate(idx_to_symbol_list)} for ftr, idx_to_symbol_list in
                         idx_to_symbol.items()}
        return embed_dict, value_dict, idx_to_symbol, symbol_to_idx
