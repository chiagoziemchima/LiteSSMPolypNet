# Data and partitioning notes

The supplied notebook expects a dataset root with `train_set/images`, `train_set/masks`, and the five named test-set folders documented in the README.

In the supplied loader, image-mask pairs are assembled from matching filenames. The notebook then uses `train_test_split` on pairs from `train_set` with `test_size=0.2`, `random_state=42`, and `shuffle=True` to create training and validation subsets. Test datasets are read from `test_set` separately.

The code package does not contain the datasets or an image-level split manifest. Researchers should download the datasets from their official sources and preserve their published partitions.
