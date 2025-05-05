# Unsupervised_Learning/label_propagation.py

import numpy as np
from collections import Counter
import networkx as nx # For graph structure input, though not strictly required by the algorithm logic itself

class LabelPropagation:
    """
    Label Propagation Algorithm for Community Detection in Graphs.

    This implementation follows the basic LPA where nodes iteratively
    adopt the label that is most frequent among their neighbors.

    Parameters
    ----------
    max_iter : int, optional (default=100)
        The maximum number of iterations to perform.
    random_state : int, optional (default=None)
        Seed for the random number generator used for shuffling node update order.
    """
    def __init__(self, max_iter=100, random_state=None):
        self.max_iter = max_iter
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)
        self.labels_ = {} # Stores the final labels for each node
        self.iterations_ = 0 # Actual iterations run

    def fit(self, G):
        """
        Fit the Label Propagation model.

        Parameters
        ----------
        G : NetworkX Graph or similar adjacency structure
            The graph on which to perform label propagation.
            Needs methods G.nodes() and G.neighbors(node).
            Can also be an adjacency matrix/list if adapted.
        """
        if not hasattr(G, 'nodes') or not hasattr(G, 'neighbors'):
             raise TypeError("Input G must be a graph structure with 'nodes' and 'neighbors' methods (like NetworkX Graph).")

        # Initialize labels: each node starts with its own unique label
        # Using node ID itself as the initial label
        nodes = list(G.nodes())
        self.labels_ = {node: node for node in nodes}

        for iteration in range(self.max_iter):
            self.iterations_ = iteration + 1
            changed = False
            # Process nodes in a random order to prevent bias
            nodes_shuffled = self._rng.permutation(nodes)

            for node in nodes_shuffled:
                # Get labels of neighbors
                neighbor_labels = [self.labels_[neighbor] for neighbor in G.neighbors(node)]

                if neighbor_labels:
                    # Find the most frequent label among neighbors
                    # Use Counter for efficiency
                    label_counts = Counter(neighbor_labels)
                    # Handle ties by choosing one randomly (or deterministically, e.g., smallest label)
                    # Here, Counter.most_common(1) picks one deterministically if counts are equal
                    most_common_label, _ = label_counts.most_common(1)[0]

                    # Update label if it's different from the most common neighbor label
                    if self.labels_[node] != most_common_label:
                        self.labels_[node] = most_common_label
                        changed = True

            # Check for convergence
            if not changed:
                print(f"Label Propagation converged after {self.iterations_} iterations.")
                break
        else: # Runs if loop finishes without break
             print(f"Label Propagation reached max iterations ({self.max_iter}) without full convergence.")


        return self

    def fit_predict(self, G):
        """
        Fit the model and return the labels.

        Parameters
        ----------
        G : NetworkX Graph or similar adjacency structure
            The graph.

        Returns
        -------
        labels : dict
            A dictionary mapping each node to its final assigned label (community).
        """
        self.fit(G)
        return self.labels_