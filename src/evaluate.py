import pandas as pd
from datasets import load_dataset

from agent import *

def compute_metrics(eval_pred):
    """
    Compute classification accuracy for evaluation predictions.
    :param eval_pred: A tuple `(logits, labels)`
    :return: A dictionary containing:
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct = np.sum(predictions == labels)
    accuracy_score = correct / len(labels)
    return {"accuracy": accuracy_score}

def print_misclassified(y_test, y_preds, dataset):
    """
    Display misclassified examples from a text classification dataset.
    :param y_test: The true class labels for the test set.
    :param y_preds: The predicted class labels from the model.
    :param dataset:  A Hugging Face `DatasetDict` containing at least a `"test"` split
            with a `"text"` field.
    """
    misclassified = []
    for i, (gt, pred) in enumerate(zip(y_test, y_preds)):
        if gt != pred:
            misclassified.append({
                "id": i,
                "prompt": dataset["test"]["text"][i],
                "ground_truth": gt,
                "prediction": pred
            })

    print("Total Misclassified:", len(misclassified))
    for i, item in enumerate(misclassified):
        print(f"\n{separator(f"Prompt {item['id']} Summary")}\nGround Truth Label: {item['ground_truth']} - Predicted Label: {item['prediction']} - Prompt Word Length: {count_words(item['prompt'])}", f"\n{separator("Prompt")}\n {item['prompt']}")
    print("\n", separator())


if __name__ == "__main__":
    # Download and save the safe-guard-prompt-injection dataset
    raw_dataset = load_dataset("xTRam1/safe-guard-prompt-injection", cache_dir=DATA_DIR)
    cleaned_dataset = raw_dataset.map(clean_text)

    print(f"Test/Train size: {len(cleaned_dataset['test'])}/{len(cleaned_dataset['train'])}")

    # Load these packages for evaluation of logistic reg. only
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    tfidf_dataset = cleaned_dataset.map(lambda x: {"text": preprocess_spacy(x["text"])})
    vectorizer = TfidfVectorizer(max_features=25000)
    X_train = vectorizer.fit_transform(tfidf_dataset["train"]["text"])
    X_test = vectorizer.transform(tfidf_dataset["test"]["text"])
    y_train = cleaned_dataset["train"]["label"]
    y_test = cleaned_dataset["test"]["label"]
    save_with_pickle(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Evaluate Logistic Regression
    y_pred = clf.predict(X_test)
    print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))

    misclassified = []
    for i, (gt, pred) in enumerate(zip(y_test, y_pred)):
        if gt != pred:
            misclassified.append({
                "prompt": cleaned_dataset["test"]["text"][i],
                "ground_truth": gt,
                "prediction": pred
            })

    # # Display Mismatched Labels for baseline logistic model
    # print("Total Misclassified:", len(misclassified))
    # for i, item in enumerate(misclassified):
    #     print(
    #         f"\n{separator(f"Prompt {i + 1} Summary")}\nGround Truth Label: {item['ground_truth']} - Predicted Label: {item['prediction']} - Prompt Word Length: {count_words(item['prompt'])}",
    #         f"\n{separator("Prompt")}\n {item['prompt']}")
    # print("\n", separator())

    # Prepare dataset and DistilBERT classifier
    raw_train_valid = cleaned_dataset["train"].train_test_split(test_size=VALIDATION_SPLIT, seed=SEED)
    dataset_split = {
        "train": raw_train_valid["train"],
        "validation": raw_train_valid["test"],
        "test": cleaned_dataset["test"]
    }
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', cache_dir=MODEL_DIR)
    bert_tokenized_dataset = {
        split: dataset_split[split].map(lambda x: bert_tokenizer(x['text'], truncation=True, max_length=512),
                                        batched=True) for split in dataset_split}

    bert_train_dataset = bert_tokenized_dataset["train"]
    bert_val_dataset = bert_tokenized_dataset["validation"]
    bert_test_dataset = bert_tokenized_dataset["test"]

    # Classification Stage
    bert_model_path = f"{MODEL_DIR}/distilbert-finetuned-psa-5eps"
    # train_distilbert_model(bert_train_dataset, bert_val_dataset, bert_tokenizer, compute_metrics, model_path=bert_model_path, model_dir=MODEL_DIR, epoches=5)
    bert_model = fetch_distilbert_model(bert_model_path)

    # Run DistilBERT classifier prediction
    sliding_y_preds, sliding_y_probs = distilbert_predict(bert_test_dataset, bert_tokenizer, bert_model)

    # Show first 10 examples
    for i in range(10):
        print(f"Text: {cleaned_dataset['test'][i]['text'][:60]}...")
        print(f"True Label: {cleaned_dataset['test'][i]['label']}, Predicted: {sliding_y_preds[i]}")
        print("-" * 50)

    print("DistilBERT Classification Report:\n", classification_report(cleaned_dataset["test"]["label"], sliding_y_preds, target_names=["negative", "positive"]))

    # Fetch saved results
    result_df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    result_df[result_df["confidence"] < 0.6][['ground_truth', 'label', 'confidence', 'prompt']]

    # Plot confidence distribution
    result_df["confidence"].plot(kind="hist")

    print(f"Fallback Use {len(result_df[result_df["fallback_used"]])} out of {len(result_df)}")