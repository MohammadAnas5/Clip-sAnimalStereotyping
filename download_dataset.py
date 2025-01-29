import kagglehub

# Download the latest version of the dataset
def download_dataset(dataset_name="anas123siddiqui/animals"):
    path = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    download_dataset()
