from datasets import load_dataset
from collections import Counter

RANDOM_SEED = 42

def load_and_prepare_data(seed: int = RANDOM_SEED):
    
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    ds = dataset["train"]

    
    ds = ds.filter(lambda ex: ex["language"] == "en")

   
    target_queues = [
        "Technical Support",
        "Customer Service",
        "Billing and Payments",
        "Sales and Pre-Sales",
        "General Inquiry",
    ]
    ds = ds.filter(lambda ex: ex["queue"] in target_queues)

    
    ds = ds.shuffle(seed=seed)
    train_test = ds.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    train_ds = train_test["train"]
    val_ds = test_valid["train"]
    test_ds = test_valid["test"]

    
    label_list = sorted(list(set(train_ds["queue"])))
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    
    def build_text(ex):
        return {"text": f"Subject: {ex['subject']}\n\nBody: {ex['body']}"}

    train_ds = train_ds.map(build_text)
    val_ds = val_ds.map(build_text)
    test_ds = test_ds.map(build_text)

    def map_label(ex):
        return {"label": label2id[ex["queue"]]}

    train_ds = train_ds.map(map_label)
    val_ds = val_ds.map(map_label)
    test_ds = test_ds.map(map_label)

    
    print("Label distribution (train):")
    print(Counter(train_ds["queue"]))

    return train_ds, val_ds, test_ds, label_list, label2id, id2label


if __name__ == "__main__":
    train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
    print("Done!")
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))
    print("Labels:", label2id)