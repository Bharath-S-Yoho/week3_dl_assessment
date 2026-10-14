from transformers import AutoTokenizer, pipeline

MODEL_NAME = "distilbert-base-uncased"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentences = [
        "I love this product.",
        "The delivery was late and the box was damaged."
    ]
    for sentence in sentences:
        result = tokenizer(sentence)
        print("\nSentence:", sentence)
        print("Tokens:", tokenizer.convert_ids_to_tokens(result["input_ids"]))
        print("Input IDs:", result["input_ids"])
        print("Attention mask:", result["attention_mask"])

    sentiment = pipeline("sentiment-analysis")
    ner = pipeline("ner", aggregation_strategy="simple")
    sentiment_sentences = [
        "I really like this service.",
        "The product is disappointing.",
        "The delivery arrived on time."
    ]
    ner_sentences = [
        "Sundar works at Google in California.",
        "Mary visited Paris last Monday.",
        "Microsoft opened an office in London."
    ]
    print("\nSentiment-analysis pipeline:")
    for sentence in sentiment_sentences:
        print(sentence, "->", sentiment(sentence))
    print("\nNER pipeline:")
    for sentence in ner_sentences:
        print(sentence, "->", ner(sentence))

if __name__ == "__main__":
    main()
