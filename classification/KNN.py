import numpy as np
import math
np.set_printoptions(suppress=True)

class KNN:
    def __init__(self, k=3):
        self.k = k;

    def fit(self, X, y):
        self.X_train = X;
        self.y_train = y;

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        votes = {}
        distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)
        k_nearest_neigh = list(zip(k_indices, distances[k_indices]))[:self.k]
        for i in range(len(k_nearest_neigh)):
            clas = self.y_train[k_nearest_neigh[i][0]]
            if clas in votes:
                votes[clas] += 1
            else:
                votes[clas] = 1
            k_nearest_neigh[i] = list(k_nearest_neigh[i])
            k_nearest_neigh[i].append(clas)
        sorted_votes = dict(sorted(votes.items(), key=lambda item: item[1], reverse=True))
        result = self._check_similar_votes(sorted_votes)
        if not result:
            print(sorted_votes)
            return next(iter(sorted_votes))
        else:
            print(self._weighted_voting(k_nearest_neigh, result))
            return self._weighted_voting(k_nearest_neigh, result)


    def _distance(self, x, y):
        distanse = 0
        for i in range(len(x)):
            if isinstance(x[i], str):
                distanse += self._similarity(x[i], y[i])
            else:
                distanse += self._euclidean_distance(x[i], y[i])
        return math.sqrt(distanse)


    def _similarity(self, x, y):
        return 1 if x == y else 0

    def _euclidean_distance(self, x, y):
        return (x - y)**2

    def _weighted_voting(self, neigh, votes):
        weighted_voting = {}
        for vote in votes:
            for i in range(len(neigh)):
                if vote == neigh[i][-1]:
                    if vote in weighted_voting:
                        weighted_voting[vote] += (1/neigh[i][-2]**2)
                    else:
                        weighted_voting[vote] = (1/neigh[i][-2]**2)

        sorted_votes = dict(sorted(weighted_voting.items(), key=lambda item: item[1], reverse=True))
        return next(iter(sorted_votes))


    def _check_similar_votes(self, votes):
        identical_keys = []
        unique_values = set(votes.values())

        for value in unique_values:
            keys_with_value = [key for key, val in votes.items() if val == value]
            if len(keys_with_value) > 1:
                identical_keys.extend(keys_with_value)

        return identical_keys