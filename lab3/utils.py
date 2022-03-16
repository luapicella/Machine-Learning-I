def vcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return vector.reshape(shape0, 1)


def vrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return vector.reshape(1, shape1)

def centerDataset(dataset):
    return dataset - dataset.mean(axis=1)