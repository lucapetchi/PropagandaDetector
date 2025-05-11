import os
import spacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn.model_selection import KFold
from data.load_data import load_article, load_labels_task1
from data.preprocess import char_spans_to_bio
from extract_features import extract_features_from_doc

nlp = spacy.load("en_core_web_sm")

# Load all training data
def load_all_data():
    articles_dir = "data/train-articles"
    labels_dir = "data/train-labels-task1-span-identification"
    X, y = [], []

    for fname in os.listdir(articles_dir):
        if not fname.endswith(".txt"):
            continue
        article_id = fname.replace("article", "").replace(".txt", "")
        label_path = os.path.join(labels_dir, f"article{article_id}.task1-SI.labels")
        article_path = os.path.join(articles_dir, fname)

        if not os.path.exists(label_path):
            continue

        try:
            text = load_article(article_path)
            spans = load_labels_task1(label_path)
            doc = nlp(text)
            token_tag_pairs = char_spans_to_bio(text, spans)

            X.append(extract_features_from_doc(doc))
            y.append([tag for _, tag in token_tag_pairs])
        except Exception as e:
            print(f"Error with {article_id}: {e}")
            continue
    return X, y

def main():
    X_all, y_all = load_all_data()
    print(f"Loaded {len(X_all)} articles.")

    from collections import Counter
    tag_counts = Counter(tag for seq in y_all for tag in seq)
    print("Label distribution:", tag_counts)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    scores = []

    for train_idx, test_idx in kf.split(X_all):
        X_train = [X_all[i] for i in train_idx]
        y_train = [y_all[i] for i in train_idx]
        X_test  = [X_all[i] for i in test_idx]
        y_test  = [y_all[i] for i in test_idx]

        crf = CRF(algorithm='lbfgs', max_iterations=100)
        crf.fit(X_train, y_train)
        y_pred = crf.predict(X_test)

        f1 = flat_f1_score(y_test, y_pred, average='weighted')
        print(f"Fold {fold} F1: {f1:.4f}")
        scores.append(f1)
        fold += 1

    avg_f1 = sum(scores) / len(scores)
    print(f"\n✅ Average F1 across 5 folds: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
