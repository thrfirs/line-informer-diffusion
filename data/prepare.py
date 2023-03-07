import pandas
import os
import json
import shutil


raw_dataset_dir = "./raw/"
prepared_dataset_dir = "./prepared/"


def prepare_file(file_path: str) -> str:
    pd = pandas.read_excel(file_path)
    data = pd.to_numpy()
    line = data.tolist()[:5000]
    start_x, start_y, start_z = line[0]
    second_x, second_y, second_z = line[1]
    dist = ((second_x - start_x) ** 2 + (second_y - start_y) ** 2 + (second_z - start_z) ** 2) ** 0.5
    j = {"line": line, "base_params": [dist]}
    prepared_file_path = os.path.join(prepared_dataset_dir, os.path.basename(file_path) + ".json")
    with open(prepared_file_path, "w") as f:
        json.dump(j, f, separators=(',', ':'))
    return prepared_file_path


def main():
    if os.path.exists(prepared_dataset_dir):
        shutil.rmtree(prepared_dataset_dir)
    os.makedirs(prepared_dataset_dir)

    prepared_paths = []
    for root, dirs, files in os.walk(raw_dataset_dir):
        for file in files:
            if file.endswith(".xlsx"):
                prepared_paths.append(prepare_file(os.path.join(root, file)))
    
    with open("dataset.txt", "w") as f:
        for path in prepared_paths:
            abs_path = os.path.abspath(path)
            print(abs_path, file=f)


if __name__ == "__main__":
    main()
