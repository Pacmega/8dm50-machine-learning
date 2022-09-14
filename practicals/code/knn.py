'''
Module containing the base class and the classification and regression
knn classes.
'''
from scipy.spatial.distance import cdist
import numpy as np

class KnnBase(object):
    '''
    Base class that the KnnClassifier and KnnRegressor derive their
    key functionality from. Can be used directly as well, but the
    derived classes are meant for simpler usage.
    '''
    def __init__(self, train_data, train_targets):
        self.feature_means = np.mean(train_data,axis=0)
        self.feature_stds  = np.std(train_data,axis=0)
        self.norm_train_data = (train_data - self.feature_means) / self.feature_stds
        self.targets = train_targets


    def get_k_nearest_targets(self, new_data, k):
        '''
        Main function to do determine the k nearest neighbors that were
        in the train data, for each row that is in new_data. Function
        is agnostic to what the target is exactly, as it returns an
        array of target values of the neighbors and leaves the final
        processing to its caller.
        @param new_data: a numpy array of new data points to predict
            for. Each row should be one point, and all points must
            have the same features in the same order as in training.
        @param k: an integer specifying how many of the nearest
            neighbors should be returned per point.
        @return: an array of shape (new_data.shape[0], k), with the
            target values of each row's k nearest neighbors.
        '''
        # Transform each new data point using the feature mean & std
        #   from the train data, and calculate Euclidean distance to
        #   all training data points.
        transformed_data = (new_data-self.feature_means)/self.feature_stds
        distances = cdist(transformed_data, self.norm_train_data)
        
        # Take k nearest points, get their target values
        indices = np.argpartition(distances, k)
        k_nearest_targets = self.targets[indices[:,:k]]
        
        return k_nearest_targets


class KnnClassifier(KnnBase):
    '''
    Class that builds upon the KnnBase class, and processes its
    predictions per data point into the predicted class for that point.
    '''
    def __init__(self, train_data, train_targets):
        super().__init__(train_data, train_targets)
    
    
    def predict_classes(self, new_data, k):
        '''
        Predict the classes of all new data points (rows).
        @param new_data: a numpy array of new data points to predict
            for. Each row should be one point, and all points must
            have the same features in the same order as in training.
        @param k: an integer specifying how many of the nearest
            neighbors should be returned per point.
        @return: an array of shape (new_data.shape[0],), containing
            the newly predicted class for each point in new_data.
        '''
        nearest_targets = self.get_k_nearest_targets(new_data, k)
        return np.round(np.mean(nearest_targets, axis=1))
    
    
class KnnRegressor(KnnBase):
    '''
    Class that builds upon the KnnBase class, and averages its
    predictions per data point to get the mean prediction for 
    a regression.
    '''
    def __init__(self, train_data, train_targets):
        super().__init__(train_data, train_targets)
    
    
    def predict_regression(self, new_data, k):
        '''
        Predict the regression output of all new data points (rows)
            by averaging the values of the k nearest neighbors.
        @param new_data: a numpy array of new data points to predict
            for. Each row should be one point, and all points must
            have the same features in the same order as in training.
        @param k: an integer specifying how many of the nearest
            neighbors should be returned per point.
        @return: an array of shape (new_data.shape[0],), containing
            the newly predicted value for each point in new_data.
        '''
        nearest_targets = self.get_k_nearest_targets(new_data, k)
        return np.mean(nearest_targets, axis=1)
