from signifyai.collect import CollectConfig, run_collection


if __name__ == "__main__":
    # Edit the label each time you collect a new class.
    cfg = CollectConfig(label="hello", samples=250)
    run_collection(cfg)
