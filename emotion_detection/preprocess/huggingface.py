


def huggingface_proceed(train_dataset, val_dataset, test_dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["combined_text"], padding=True, truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset = train_dataset.remove_columns(["combined_text"])
    val_dataset = val_dataset.remove_columns(["combined_text"])
    test_dataset = test_dataset.remove_columns(["combined_text"])

    train_dataset = train_dataset.rename_column("emotion", "labels")
    val_dataset = val_dataset.rename_column("emotion", "labels")
    test_dataset = test_dataset.rename_column("emotion", "labels")


    # Set dataset format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, val_dataset, test_dataset