# Topological Graph Features

Topological feature calculators infrastructure.

## Calculating Features
The feature calculators work on a gnx instance.
At first we'll define a graph (networkx Graph) and a logger

```python
import networkx as nx
from loggers import PrintLogger

gnx = nx.DiGraph()  # should be a subclass of Graph
gnx.add_edges_from([(0, 1), (0, 2), (1, 3), (3, 2)])

logger = PrintLogger("MyLogger")
```

On that graph we'll want to calculate the topological features.
We can do that in 2 ways:

* Calculate a specific feature.

```python
import numpy as np
from features_algorithms.vertices.louvain import LouvainCalculator

feature = LouvainCalculator(gnx, logger=logger)
feature.build()

mx = feature.to_matrix(mtype=np.matrix)
```

* Calculate a set of Features.

```python
import numpy as np
from features_infra.graph_features import GraphFeatures

from features_algorithms.vertices.louvain import LouvainCalculator
from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator

features_meta = {
  "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
  "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
}

features = GraphFeatures(gnx, features_meta, logger=logger)
features.build()

mx = features.to_matrix(mtype=np.matrix)
```
