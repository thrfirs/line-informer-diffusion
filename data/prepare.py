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
    x_span = max([x for x, y, z in line]) - min([x for x, y, z in line])
    y_span = max([y for x, y, z in line]) - min([y for x, y, z in line])
    z_span = max([z for x, y, z in line]) - min([z for x, y, z in line])
    start_x, start_y, start_z = line[0]
    j = {"line": line, "params": [x_span, y_span, z_span, start_x, start_y, start_z]}
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
            print(path, file=f)


if __name__ == "__main__":
    main()
