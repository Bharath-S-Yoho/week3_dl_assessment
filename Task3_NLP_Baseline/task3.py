import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def make_dataset():
    # 12 simple examples per template, with small variations to create 360 labelled texts.
    positive = [
        "I love this product", "excellent quality and fast delivery",
        "very happy with my purchase", "the service was wonderful",
        "this works perfectly", "great support from the team",
        "easy to use and reliable", "thank you for the quick help",
        "the item is fantastic", "friendly staff and good service",
        "I am satisfied with the result", "superb experience overall"
    ]
    negative = [
        "I hate this product", "terrible quality and slow delivery",
        "very unhappy with my purchase", "the service was awful",
        "this does not work", "poor support from the team",
        "hard to use and unreliable", "no one helped me",
        "the item is disappointing", "rude staff and bad service",
        "I am dissatisfied with the result", "worst experience overall"
    ]
    rows = []
    for label, templates in [("positive", positive), ("negative", negative)]:
        for i in range(15):
            for text in templates:
                # Add a small neutral phrase variation while preserving the label.
                suffixes = [" today", " for me", " overall", " so far", " really"]
                rows.append({"text": text + suffixes[i % len(suffixes)], "label": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = make_dataset()
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv("feedback_dataset.csv", index=False)
    print("Dataset rows:", len(df))
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print("Confusion matrix (label order: negative, positive):")
    print(confusion_matrix(y_test, predictions, labels=["negative", "positive"]))
    print(classification_report(y_test, predictions))
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Inference time for test set: {inference_time:.4f} seconds")

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    for class_index, class_name in enumerate(classifier.classes_):
      if len(classifier.classes_) == 2:
        word_scores = classifier.coef_[0]
        if class_name == "negative":
            word_scores = -word_scores
      else:
        word_scores = classifier.coef_[class_index]

    top_indices = np.argsort(word_scores)[-10:][::-1]
    print(
        f"10 important words for {class_name}:",
        feature_names[top_indices].tolist()
    )

    # GloVe vectors and the required nearest words / analogy / averaged sentence vectors.
    try:
        import gensim.downloader as api
        print("Loading GloVe vectors. First run may download a large file...")
        vectors = api.load("glove-wiki-gigaword-50")
        for word in ["king", "computer"]:
            print(f"5 nearest words to {word}:", vectors.most_similar(word, topn=5))
        print("king - man + woman =", vectors.most_similar(
            positive=["king", "woman"], negative=["man"], topn=1
        )[0][0])

        def sentence_vector(sentence):
            words = [word for word in clean_text(sentence).split() if word in vectors]
            if not words:
                return np.zeros(50)
            return np.mean([vectors[word] for word in words], axis=0)

        X_train_vec = np.vstack([sentence_vector(s) for s in X_train])
        X_test_vec = np.vstack([sentence_vector(s) for s in X_test])
        embed_model = LogisticRegression(max_iter=1000)
        embed_model.fit(X_train_vec, y_train)
        embed_predictions = embed_model.predict(X_test_vec)
        print("Averaged GloVe sentence-vector accuracy:",
              round(accuracy_score(y_test, embed_predictions), 4))
        print("TF-IDF accuracy:", round(accuracy_score(y_test, predictions), 4))
    except Exception as error:
        print("GloVe/Word2Vec step could not run:", error)
        print("Install compatible gensim/scipy and rerun with internet access.")


if __name__ == "__main__":
    main()
