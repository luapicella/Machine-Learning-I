def vcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return vector.reshape(shape0, 1)


def vrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return vector.reshape(1, shape1)

def mcol(v):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return v.reshape((v.size, 1))

def mrow(v):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return (v.reshape(1, v.size))

def centerDataset(dataset):
    return dataset - dataset.mean(axis=1)