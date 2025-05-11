from sklearn_crfsuite import CRF

def train_crf(X_train, y_train):
    crf = CRF(algorithm='lbfgs', max_iterations=100)
    crf.fit(X_train, y_train)
    return crf
