import os

from scipy.io import arff
import pandas as pd

def preprocess_arff(filepath):
    """
    Some ARFF files have multi-line @attribute definitions;
    this ensures each attribute is on a single line.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    with open(filepath, 'w') as file:
        for line in lines:
            if line.strip().startswith('@attribute'):
                file.write(line.replace('\n', ' ').replace('\r', ''))
            else:
                file.write(line)

def load_scenario_data(folder_path, scenario_label):
    frames = []
    """
    Loads ARFF files from `folder_path`, applies scenario-specific logic to
    determine the VPN label, then appends the data to the global `frames` list.

    :param folder_path: Directory with .arff files
    :param scenario_label: One of {"A1", "A2", "B"} indicating how to label VPN
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.arff'):
                # Skip AllinOne files
                if 'AllinOne' in file:
                    continue

                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file_path}")

                # Fix potential multi-line attributes
                preprocess_arff(file_path)
                # Load ARFF
                try:
                    data, meta = arff.loadarff(file_path)
                    df_temp = pd.DataFrame(data)

                    # Decode any byte strings
                    for col in df_temp.columns:
                        if df_temp[col].dtype == 'object':
                            df_temp[col] = df_temp[col].apply(
                                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                            ) # this is needed because stuff stored in the arff files are stored as bytes and strings

                    # SCENARIO A1: class1 is already {Non-VPN, VPN}
                    if scenario_label == 'A1':
                        df_temp['class1'] = df_temp['class1'].map({'Non-VPN': 0, 'VPN': 1})

                    # SCENARIO A2: File name determines vpn or not, we ignore class 1 in this case
                    elif scenario_label == 'A2':
                        #file name check
                        if 'NO-VPN' in file.upper():
                            df_temp['class1'] = 0
                        else:
                            df_temp['class1'] = 1

                    # SCENARIO B: row-based but check if class1 starts with "VPN-"
                    elif scenario_label == 'B':
                        df_temp['class1'] = df_temp['class1'].apply(
                            lambda val: 1 if val.startswith('VPN-') else 0
                        )

                    print(f'rows of individual file: {len(df_temp)}')

                    frames.append(df_temp)
                except Exception as e:
                    print(f"Error processing file: {file_path}")
                    print(e)
                    continue
    return pd.concat(frames, ignore_index=True)

def load_scenario_data_multiclass(folder_path, scenario_label):
    frames = []
    """
    Loads ARFF files from `folder_path`, applies scenario-specific logic to
    determine the class label, then appends the data to the global `frames` list.

    :param folder_path: Directory with .arff files
    :param scenario_label: One of {"A1", "A2", "B"} indicating how to label the data
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.arff'):
                # Skip AllinOne files
                if 'AllinOne' in file:
                    continue

                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file_path}")

                # If you have a "preprocess_arff" function that fixes multi-line attributes, call it here
                # preprocess_arff(file_path)

                try:
                    # Load ARFF
                    data, meta = arff.loadarff(file_path)
                    df_temp = pd.DataFrame(data)

                    # Decode any byte strings
                    for col in df_temp.columns:
                        if df_temp[col].dtype == 'object':
                            df_temp[col] = df_temp[col].apply(
                                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                            )

                    # SCENARIO A1: class1 is already {Non-VPN, VPN}
                    if scenario_label == 'A1':
                        df_temp['class1'] = df_temp['class1'].map({'Non-VPN': 0, 'VPN': 1})

                    elif scenario_label == 'A2':
                        # If file name has 'NO-VPN', do nothing -> keep original (BROWSING, CHAT, etc.)
                        # Otherwise, prefix 'VPN-' to each label
                        if 'NO-VPN' in file.upper():
                            pass
                        else:
                            df_temp['class1'] = 'VPN-' + df_temp['class1'].astype(str)

                    elif scenario_label == 'B':
                        # For Scenario B, do nothing; the ARFF already has "BROWSING" or "VPN-BROWSING" etc.
                        pass

                    # Map classes to 0..13 for multi-class scenarios A2 and B
                    if scenario_label in ['A2', 'B']:
                        class_mapping = {
                            'BROWSING': 0, 'CHAT': 1, 'STREAMING': 2, 'MAIL': 3, 'VOIP': 4,
                            'P2P': 5, 'FT': 6,
                            'VPN-BROWSING': 7, 'VPN-CHAT': 8, 'VPN-STREAMING': 9, 'VPN-MAIL': 10,
                            'VPN-VOIP': 11, 'VPN-P2P': 12, 'VPN-FT': 13
                        }
                        df_temp['class1'] = df_temp['class1'].map(class_mapping)

                    print(f'rows of individual file: {len(df_temp)}')
                    frames.append(df_temp)
                except Exception as e:
                    print(f"Error processing file: {file_path}")
                    print(e)
                    continue
    return pd.concat(frames, ignore_index=True)
