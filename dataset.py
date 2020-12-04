import json

from torch.utils.data import DataLoader


def load_dataset(batch_size):
    train_data = open("dataset/train.jsonl", "r").read().splitlines()

    train_set = []
    for data in train_data:
        data = json.loads(data)
        article_original = data["article_original"]
        article_original = [a.replace("\n", " ") for a in article_original]
        article_original = " ".join(article_original)
        abstractive = data["abstractive"].replace("\n", " ")
        train_set.append((article_original, abstractive))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    return train_loader
