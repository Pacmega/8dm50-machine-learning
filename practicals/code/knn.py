from scipy.spatial.distance import cdist
import numpy as np

from scipy.spatial.distance import cdist
import numpy as np

class KnnBase(object):
    def __init__(self, train_data, train_targets):
        self.train_data = self.normalise_dataset(train_data)
        self.targets = train_targets


    def normalise_dataset(self, dataset):
        def normalise(feature):
            mean = np.mean(feature)
            std  = np.std(feature)
            return (feature - mean) / std
        
        return np.apply_along_axis(normalise, axis=0, arr=dataset)


    def get_k_nearest_targets(self, new_data, k):
        # For each data point calculate Euclidean distance to all other points
        transformed_data = self.normalise_dataset(new_data)
        distances = cdist(transformed_data, self.train_data)
        
        # Take k nearest points, get their target values
        k_nearest_indices = np.argsort(distances, axis=0)[:,:k]
        k_nearest_classes = self.targets[k_nearest_indices]

        # Take most common target value for each point and return that value
        return k_nearest_classes


class KnnClassifier(KnnBase):
    def __init__(self, train_data, train_targets):
        super().__init__(train_data, train_targets)
    
    
    def predict_classes(self, new_data, k):
        nearest_targets = self.get_k_nearest_targets(new_data, k)
        return np.round(np.mean(nearest_targets, axis=1))
    
    
class KnnRegressor(KnnBase):
    def __init__(self, train_data, train_targets):
        super().__init__(train_data, train_targets)
    
    
    def predict_regression(self, new_data, k):
        nearest_targets = self.get_k_nearest_targets(new_data, k)
        return np.mean(nearest_targets, axis=1)
