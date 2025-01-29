import subprocess

if __name__ == "__main__":
    # Run all scripts in order
    subprocess.run(["python", "download_dataset.py"])
    subprocess.run(["python", "preprocess_dataset.py"])
    subprocess.run(["python", "clean_dataset.py"])
    subprocess.run(["python", "stereotype_detection.py"])
