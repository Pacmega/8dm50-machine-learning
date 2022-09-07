class KnnClassifier(object):
    def __init__(self, train_data):
        super(Knearestneighbours, self).__init__()
        self.train_data = train_data

    def normalise_dataset(self, dataset):
        def normalise(feature):
            mean = np.mean(feature)
            std  = np.std(feature)
            return (feature - mean) / std
        
        return np.apply_along_axis(normalise, axis=0, arr=dataset)

    def predict_new_data(self, new_data, k):
        # For each data point,
        # calculate Euclidean distance to all other points
        # Take k nearest points, get their target values
        # take most common target value (for reg, this is "take mean")
        # Return that value
        raise NotImplementedError(':)')

# def knn(new_data, k):
#     predicted_labels = []
    
#     for point in new_data:
#         nearest_points = []
#         index = 0
#         for i in range(diabetes_X.shape[0]):
#             # Euclidean distance
#             dist_glucose = (diabetes_X[i,0]-point[0])**2
#             dist_bloodpressure = (diabetes_X[i,1]-point[1])**2
#             distance = np.sqrt(dist_glucose + dist_bloodpressure)

#             if len(nearest_points) != k:
#                 # Still populating the list of k nearest neighbours
#                 nearest_points.append((distance, diabetes_y[i]))
#                 nearest_points.sort(key=lambda neighbour: neighbour[0])
#             elif nearest_points[-1][0] > distance:
#                 # Current observed data point is closer than furthest away
#                 #   previously known one. Replace and re-sort.
#                 nearest_points[-1] = (distance, diabetes_y[i])
#                 nearest_points.sort(key=lambda neighbour: neighbour[0])
#             # Else: this point was not close enough to be relevant for knn

#         # As the target label is binary, we can shortcut the label decision process
#         # Note: 0.5 is rounded to 0 by Python
#         nearest_labels = np.array(nearest_points)[:,1]
#         most_common_label = round(np.mean(nearest_labels))
#         predicted_labels.append(most_common_label)
        
#     return predicted_labels