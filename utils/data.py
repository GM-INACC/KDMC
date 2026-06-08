import pandas as pd
import numpy as np

class DataLoader:

    def __init__(
        self,
        datasetpath: str,
        labelpath: str,
        sorted: bool = False,
        is_syn: bool = False
    ):
        if sorted:
            self.data = pd.read_csv(datasetpath).sort_index(axis=1)
        else:
            self.data = pd.read_csv(datasetpath)
        self.label = pd.read_csv(labelpath)

        self.node_list     = self.data.columns.to_list()
        self.num_variables = len(self.node_list)
        self.is_syn        = is_syn

        self.X_full = self.data.to_numpy()

        self.true_dag = self.get_label_adj(
            self.num_variables, self.label, self.node_list
        )

    def get_label_adj(self, num_variables: int, label: pd.DataFrame, node_list: list) -> np.ndarray:
        adj_true = np.zeros((num_variables, num_variables))
        columns = label.columns.tolist()

        # Edge list format: two columns like [source, target].
        if label.shape[1] == 2 and not set(columns).issuperset(set(node_list)):
            for start, end in label.values.tolist():
                i = node_list.index(start)
                j = node_list.index(end)
                adj_true[i, j] = 1
            return adj_true

        # Adjacency-matrix format: first column is row labels, remaining columns are node names.
        label_matrix = label.copy()
        first_col = str(label_matrix.columns[0])
        if first_col.startswith("Unnamed:") or first_col == "":
            label_matrix = label_matrix.set_index(label_matrix.columns[0])

        if set(label_matrix.columns) != set(node_list):
            missing = [name for name in node_list if name not in label_matrix.columns]
            extra = [name for name in label_matrix.columns if name not in node_list]
            raise ValueError(
                "Label columns do not match data columns. "
                f"Missing in label: {missing}; extra in label: {extra}"
            )

        label_matrix = label_matrix.reindex(columns=node_list)

        if list(label_matrix.index) != node_list:
            label_matrix = label_matrix.reindex(index=node_list)

        values = label_matrix.to_numpy(dtype=float)
        if values.shape != (num_variables, num_variables):
            raise ValueError(
                f"Adjacency label matrix has shape {values.shape}, expected {(num_variables, num_variables)}"
            )

        adj_true[:, :] = values
        return adj_true
