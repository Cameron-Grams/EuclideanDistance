# worksheet for euclidean distance function

a = np.array([1,1])
b = np.array([2,2])

long = math.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)
short = np.linalg.norm(a-b)

#-----------------------------------

indices = [20, 32, 45]
order = [1, 2, 0]

sort_obj = list(zip(indices, order))
composite_array = np.array(sort_obj, dtype=[('index', 'i4'), ('rank', 'i4')])

order = np.argsort(composite_array, order='rank')

ordered_indices = [n[0] for n in composite_array[order]]
# ordered_indices

#-------------------------------------

def euclidean_classifier(X_target, X_train=X_train, y_train=y_train, n = 3):
    # given a set of X samples with y lables return a list of predictions
    # list will be based on the average of the labels of
    # the n closest values in the sample X to each sample in the target

    # each sample from target
        # compared to each sample from known
        # capture known sample distance to target
        # capture class of known sample
        # rank order the known (distance, class) tuples by ascending distance
        # average the class value for the first n members of the ordered distances

    predictions = []

    for i in range(len(X_target)):
        vector = X_target[i]
        vector_values = []

        for j in range(len(X_train)):
            known_vector = X_train[j]
            distance = np.linalg.norm(vector - known_vector)
            known_class = y_train[j]
            vector_values.append((distance, known_class))

        distance_vector = np.array(vector_values, dtype=[('distance', np.float64), ('class', 'i2')])
        order = np.argsort(distance_vector, order='distance')
        distance_vector = distance_vector[order]
        nn = distance_vector[:n]
        class_sum = sum([i[1] for i in nn])
        unk_prediction = float(class_sum)/n
        unk_prediction = 1 if unk_prediction >= 0.5 else 0
        predictions.append(unk_prediction)

    return predictions



