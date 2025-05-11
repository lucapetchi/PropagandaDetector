import os
import spacy
from data.load_data import load_article, load_labels_task1
from data.preprocess import char_spans_to_bio
from extract_features import extract_features_from_doc
from sklearn_crfsuite import CRF

nlp = spacy.load("en_core_web_sm")

def main():
    articles_dir = "data/train-articles"
    labels_dir = "data/train-labels-task1-span-identification"

    X_train = []
    y_train = []

    for fname in os.listdir(articles_dir):
        if not fname.endswith(".txt"):
            continue
        article_id = fname.replace("article", "").replace(".txt", "")
        article_path = os.path.join(articles_dir, fname)
        label_path = os.path.join(labels_dir, f"article{article_id}.task1-SI.labels")

        try:
            text = load_article(article_path)
            spans = load_labels_task1(label_path)
            doc = nlp(text)

            token_tag_pairs = char_spans_to_bio(text, spans)
            X_train.append(extract_features_from_doc(doc))
            y_train.append([tag for _, tag in token_tag_pairs])

        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            continue

    print(f"Loaded {len(X_train)} articles.")

    # Train the CRF model
    crf = CRF(algorithm='lbfgs', max_iterations=100)
    crf.fit(X_train, y_train)

    print("✅ Trained CRF on full dataset!")
    import joblib
    joblib.dump(crf, "models/crf_span_detector.pkl")
    print("Model saved to models/crf_span_detector.pkl")

if __name__ == "__main__":
    main()

