from typing import List, Tuple


class TfIdf_Cluster:
    def __init__(self, clusters: List[Tuple[int,str]], inertia:float) -> None:
        self.clusters = clusters
        self.inertia = inertia
        self.top_words_clusters : dict[int,str] = None