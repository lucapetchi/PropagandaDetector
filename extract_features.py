def word2features(doc, i):
    token = doc[i]
    features = {
        'bias': 1.0,
        'word.lower()': token.text.lower(),
        'word.isupper()': token.text.isupper(),
        'word.istitle()': token.text.istitle(),
        'word.isdigit()': token.text.isdigit(),
        'postag': token.pos_,
        'shape': token.shape_,
        'prefix': token.text[:3],
        'suffix': token.text[-3:],
    }

    if i > 0:
        token1 = doc[i - 1]
        features.update({
            '-1:word.lower()': token1.text.lower(),
            '-1:postag': token1.pos_,
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(doc) - 1:
        token1 = doc[i + 1]
        features.update({
            '+1:word.lower()': token1.text.lower(),
            '+1:postag': token1.pos_,
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

def extract_features_from_doc(doc):
    return [word2features(doc, i) for i in range(len(doc))]
