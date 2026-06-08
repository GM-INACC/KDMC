import pandas as pd
import numpy as np

def extract_graph(csv_file_path: str, result_txt_file_path: str):

    df = pd.read_csv(csv_file_path)
    n = len(df.columns)
    column_mapping = {col: idx for idx, col in enumerate(df.columns)}

    adjacency_matrix = np.zeros((n, n), dtype=int)
    confidence_matrix = np.zeros((n, n), dtype=float)

    with open(result_txt_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().strip("(),")
            parts = line.split(",")
            if len(parts) == 3:
                source = parts[0].strip().replace("'", "")
                target = parts[1].strip().replace("'", "")
                confidence = float(parts[2].strip())
                if source in column_mapping and target in column_mapping:
                    s_idx = column_mapping[source]
                    t_idx = column_mapping[target]
                    adjacency_matrix[s_idx, t_idx] = 1
                    confidence_matrix[s_idx, t_idx] = confidence

    return adjacency_matrix, confidence_matrix

if __name__ == "__main__":
    csv_file_path = r'../realdata/data/rm_0.5_ASI_norm_1980_2018.csv'
    result_txt_file_path = r'./result.txt'
    init_adjacency, init_confidence = extract_graph(csv_file_path, result_txt_file_path)
    print("Test completed")
