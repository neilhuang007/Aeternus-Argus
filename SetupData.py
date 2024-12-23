import os

import requests
import tqdm
import zipfile

vpndatasets = [
    "http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20A1-ARFF.zip",
    "http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20A2-ARFF.zip",
    "http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/CSVs/Scenario%20B-ARFF.zip"
]



def downloadvpndataset(dataset):
    # Create a folder to store the dataset
    os.makedirs("vpndata", exist_ok=True)

    #clear what we have 
    # Progress bar for the total number of files
    total_progress = tqdm.tqdm(total=len(vpndatasets), desc="Total files")

    for dataset in vpndatasets:
        response = requests.get(dataset, stream=True)
        total_bytes = int(response.headers.get('content-length', 0))
        downloaded_bytes = 0
        # Progress bar for the current file
        file_progress = tqdm.tqdm(total=total_bytes, desc=dataset.split('/')[-1], unit='B', unit_scale=True, unit_divisor=1024,)

        with open(os.path.join("vpndata", dataset.split('/')[-1]), "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)

        # unzip all of the files and delete the zip files
        with zipfile.ZipFile(os.path.join("vpndata", dataset.split('/')[-1]), 'r') as zip_ref:
            zip_ref.extractall("vpndata")

        os.remove(os.path.join("vpndata", dataset.split('/')[-1]))

        file_progress.close()
        total_progress.update(1)

    total_progress.close()

def converttocsv(path_to_directory):
    files = [arff for arff in os.listdir(path_to_directory) if arff.endswith(".arff")]

    def toCsv(content):
        data = False
        header = ""
        newContent = []
        for line in content:
            if not data:
                if "@attribute" in line:
                    attri = line.split()
                    columnName = attri[attri.index("@attribute") + 1]
                    header = header + columnName + ","
                elif "@data" in line:
                    data = True
                    header = header[:-1]
                    header += '\n'
                    newContent.append(header)
            else:
                newContent.append(line)
        return newContent

    # Main loop for reading and writing files
    for zzzz, file in enumerate(files):
        with open(path_to_directory + file, "r") as inFile:
            content = inFile.readlines()
            name, ext = os.path.splitext(inFile.name)
            new = toCsv(content)
            with open(name + ".csv", "w") as outFile:
                outFile.writelines(new)


downloadvpndataset(vpndatasets)
# converttocsv("./vpndata")
