import os
import spacy
import joblib
from sklearn_crfsuite.metrics import flat_classification_report
from data.load_data import load_article, load_labels_task1
from data.preprocess import char_spans_to_bio
from extract_features import extract_features_from_doc

# Load SpaCy & model
nlp = spacy.load("en_core_web_sm")
crf = joblib.load("models/crf_span_detector.pkl")

def load_dataset(articles_dir, labels_dir):
    X, y_true = [], []
    for fname in os.listdir(articles_dir):
        if not fname.endswith(".txt"):
            continue
        article_id = fname.replace("article", "").replace(".txt", "")
        label_file = os.path.join(labels_dir, f"article{article_id}.task1-SI.labels")
        article_file = os.path.join(articles_dir, fname)


        if not os.path.exists(label_file):
            print(f"⚠️  Skipping {article_id} (label file missing)")
            continue

        try:
            text = load_article(article_file)
            spans = load_labels_task1(label_file)
            doc = nlp(text)
            token_tag_pairs = char_spans_to_bio(text, spans)
            X.append(extract_features_from_doc(doc))
            y_true.append([tag for _, tag in token_tag_pairs])
        except Exception as e:
            print(f"⚠️  Error processing article {article_id}: {e}")
            continue

    return X, y_true


def main():
    dev_articles = "data/dev-articles"
    dev_labels   = "data/dev-labels-task1-span-identification"

    print("Loading dev data…")
    X_dev, y_dev = load_dataset(dev_articles, dev_labels)

    print("Predicting…")
    y_pred = crf.predict(X_dev)

    print("Evaluation report:")
    print(flat_classification_report(
        y_dev, y_pred,
        labels=["B-PROP","I-PROP","O"],
        digits=4
    ))

if __name__ == "__main__":
    main()
