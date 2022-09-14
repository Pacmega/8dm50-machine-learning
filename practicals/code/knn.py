from scipy.spatial.distance import cdist
import numpy as np

class KnnBase(object):
    def __init__(self, train_data, train_targets):
        self.feature_means = np.mean(train_data,axis=0)
        self.feature_stds  = np.std(train_data,axis=0)
        self.norm_train_data = (train_data - self.feature_means) / self.feature_stds
        self.targets = train_targets

    def get_k_nearest_targets(self, new_data, k):
        # Transform each new data point using the feature mean & std
        #   from the train data, and calculate Euclidean distance to
        #   all training data points.
        transformed_data = (new_data-self.feature_means)/self.feature_stds
        distances = cdist(transformed_data, self.norm_train_data)
        
        # Take k nearest points, get their target values
        indices = np.argpartition(distances, k) #find indices of k nearest neighbours
        k_nearest_classes = self.targets[indices[:,:k]] #find results of k nearest neighbours
        
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
    