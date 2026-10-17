"""Scikit-learn-compatible neural models used in model comparison.

The graph classifier builds a feature-similarity graph over training examples.
Each row is a node; edges connect the nearest training rows, and a mean
neighbour message is concatenated with each node's own features.
"""
import numpy as np
try:
    import tensorflow as tf
except ImportError:  # TensorFlow is optional; sklearn models remain trainable.
    tf = None
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class TensorFlowDNNClassifier(ClassifierMixin, BaseEstimator):
    """Dense neural classifier implemented with TensorFlow's Keras API."""

    def __init__(self, hidden_layer_sizes=(128, 64, 32), dropout=0.25,
                 learning_rate=0.001, epochs=100, batch_size=128, patience=8,
                 random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        if tf is None:
            raise ImportError("TensorFlow is not installed; skip the DNN candidate or install TensorFlow.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y.astype(int))
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X).astype(np.float32)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        layers = [tf.keras.layers.Input(shape=(X_scaled.shape[1],))]
        for width in self.hidden_layer_sizes:
            layers.append(tf.keras.layers.Dense(width, activation="relu"))
            if self.dropout:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.keras_model_ = tf.keras.Sequential(layers)
        self.keras_model_.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self.history_ = self.keras_model_.fit(
            X_scaled, y,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0,
        ).history
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        positive = self.keras_model_.predict(X_scaled, verbose=0).reshape(-1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GANNClassifier(ClassifierMixin, BaseEstimator):
    """Genetic-algorithm search over MLP architectures and learning settings."""

    def __init__(self, population_size=4, generations=2, max_iter=140,
                 random_state=42):
        self.population_size = population_size
        self.generations = generations
        self.max_iter = max_iter
        self.random_state = random_state

    def _model(self, chromosome, seed):
        hidden, alpha, learning_rate = chromosome
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=12,
            random_state=seed,
        )

    @staticmethod
    def _score(model, X, y):
        probabilities = model.predict_proba(X)[:, 1]
        return brier_score_loss(y, probabilities)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        x_fit, x_valid, y_fit, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        hidden_choices = [(64, 32), (96, 48), (64, 32, 16), (128, 64)]
        alpha_choices = [0.0001, 0.001, 0.01]
        learning_rate_choices = [0.001, 0.003, 0.01]

        def random_chromosome():
            return (
                hidden_choices[rng.randint(len(hidden_choices))],
                alpha_choices[rng.randint(len(alpha_choices))],
                learning_rate_choices[rng.randint(len(learning_rate_choices))],
            )

        population = [random_chromosome() for _ in range(self.population_size)]
        fitness = {}
        for generation in range(self.generations):
            for chromosome in population:
                if chromosome not in fitness:
                    candidate = self._model(chromosome, self.random_state + generation)
                    candidate.fit(x_fit, y_fit)
                    fitness[chromosome] = self._score(candidate, x_valid, y_valid)
            ranked = sorted(population, key=lambda gene: fitness[gene])
            if generation + 1 < self.generations:
                parents = ranked[:max(2, len(ranked) // 2)]
                next_population = parents[:]
                while len(next_population) < self.population_size:
                    first = parents[rng.randint(len(parents))]
                    second = parents[rng.randint(len(parents))]
                    child = [first[0], second[1], first[2]]
                    if rng.rand() < 0.4:
                        child[0] = hidden_choices[rng.randint(len(hidden_choices))]
                    if rng.rand() < 0.4:
                        child[1] = alpha_choices[rng.randint(len(alpha_choices))]
                    if rng.rand() < 0.4:
                        child[2] = learning_rate_choices[rng.randint(len(learning_rate_choices))]
                    next_population.append(tuple(child))
                population = next_population

        self.best_config_ = min(fitness, key=fitness.get)
        self.model_ = self._model(self.best_config_, self.random_state)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


class GraphNeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    """One-step graph message-passing classifier over feature-similar rows."""

    def __init__(self, n_neighbors=8, hidden_layer_sizes=(64, 32), max_iter=220,
                 random_state=42):
        self.n_neighbors = n_neighbors
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler().fit(X)
        self.X_train_ = self.scaler_.transform(X)
        count = min(self.n_neighbors + 1, len(self.X_train_))
        self.neighbors_ = NearestNeighbors(n_neighbors=count).fit(self.X_train_)
        indices = self.neighbors_.kneighbors(self.X_train_, return_distance=False)

        aggregate = np.zeros_like(self.X_train_)
        for row_index, row_neighbors in enumerate(indices):
            # Self-loops are excluded; the remaining neighbours send messages.
            selected = [idx for idx in row_neighbors if idx != row_index][:self.n_neighbors]
            if selected:
                aggregate[row_index] = self.X_train_[selected].mean(axis=0)
        graph_features = np.hstack([self.X_train_, aggregate])
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=0.01,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=self.random_state,
        )
        self.model_.fit(graph_features, y)
        return self

    def _features(self, X):
        X_scaled = self.scaler_.transform(np.asarray(X, dtype=float))
        indices = self.neighbors_.kneighbors(
            X_scaled, n_neighbors=min(self.n_neighbors, len(self.X_train_)),
            return_distance=False,
        )
        aggregate = self.X_train_[indices].mean(axis=1)
        return np.hstack([X_scaled, aggregate])

    def predict_proba(self, X):
        return self.model_.predict_proba(self._features(X))

    def predict(self, X):
        return self.model_.predict(self._features(X))
