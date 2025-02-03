from data import testDataset
if __name__ == "__main__":
    # have one parameter, as the parameter to testDataset
    import argparse
    parser = argparse.ArgumentParser()
    # directly use the first parameter as the parameter to testDataset
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    testDataset(args.data_dir)