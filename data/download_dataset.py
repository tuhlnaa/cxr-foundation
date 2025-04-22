"""
CXR Foundation Demo
This script demonstrates how to use ELIXR embeddings from chest X-ray images
to train a simple neural network for medical finding detection.
"""
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfFolder


class ELIXRDataLoader:
    """Handle data downloading and preparation for the ELIXR model."""
    
    def __init__(self, input_dir: str = "./dataset/input", hf_repo_id: str = "google/cxr-foundation"):
        """
        Initialize the DataLoader.
        
        Args:
            input_dir: Directory where downloaded data will be stored
            hf_repo_id: HuggingFace repository ID for the ELIXR model
        """
        self.hf_repo_id = hf_repo_id
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)


    def authenticate_huggingface(self) -> None:
        """Authenticate with HuggingFace if no token is available."""
        if HfFolder.get_token() is None:
            print("Please visit: https://huggingface.co/settings/tokens")
            login()
            print("Authentication completed")
        else:
            print("HuggingFace Token already set")
            

    def download_data(self) -> Dict[str, str]:
        """Download necessary data files from HuggingFace."""
        file_paths = {}
        
        # Download and prepare label data
        file_paths["labels"] = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename="labels.csv",
            subfolder="precomputed_embeddings",
            local_dir=self.input_dir
        )
        
        # Download image thumbnails
        file_paths["thumbnails"] = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename="thumbnails_id_to_webp.npz",
            subfolder="precomputed_embeddings",
            local_dir=self.input_dir
        )
        
        # Download precomputed embeddings
        file_paths["embeddings"] = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename="embeddings.npz",
            subfolder="precomputed_embeddings",
            local_dir=self.input_dir
        )
        
        return file_paths
    

    def prepare_dataset(
            self, 
            label_path: str, 
            diagnosis: str, 
            max_cases_per_category: int = 400
        ) -> pd.DataFrame:
        """
        Prepare a balanced dataset with the specified diagnosis.
        
        Args:
            label_path: Path to the labels CSV file
            diagnosis: The diagnosis to use for classification
            max_cases_per_category: Maximum number of cases per category (positive/negative)
            
        Returns:
            DataFrame with selected cases
        """
        # Read the full labels dataset
        full_labels_df = pd.read_csv(label_path)

        # Create a balanced dataset with equal positive and negative cases
        if max_cases_per_category == None:
            max_cases_per_category = len(full_labels_df)

        negative_cases = full_labels_df[full_labels_df[diagnosis] == 0][:max_cases_per_category]
        positive_cases = full_labels_df[full_labels_df[diagnosis] == 1][:max_cases_per_category]
        
        df_labels = pd.concat([negative_cases, positive_cases], ignore_index=True)
        print(f"Selected {max_cases_per_category} positive and {max_cases_per_category} negative cases")
        print(df_labels.head(), "\n")

        return df_labels


    def load_embeddings(self, embeddings_path: str, df_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Load embeddings for the selected cases.
        
        Args:
            embeddings_path: Path to the embeddings NPZ file
            df_labels: DataFrame with selected cases
            
        Returns:
            DataFrame with embeddings added
        """
        embeddings_file = np.load(embeddings_path)
        df_labels = df_labels.copy()
        print(f"Total embeddings available: {len(embeddings_file.files)}")
        
        # Track which image_ids have embeddings
        valid_ids = []
        embeddings_list = []
        
        for image_id in df_labels['image_id']:
            try:
                embeddings_list.append(embeddings_file[image_id])
                valid_ids.append(image_id)
            except KeyError:
                pass
        
        # Create a new DataFrame with only valid IDs
        df_with_embeddings = df_labels[df_labels['image_id'].isin(valid_ids)].copy()
        df_with_embeddings['embeddings'] = embeddings_list

        print(f"Embeddings shape: {np.array(embeddings_list[0]).shape}\n")
        print(f"Cases selected: {len(df_labels)}")
        print(f"Cases with available embeddings: {len(df_with_embeddings)}")
        
        embeddings_file.close()
        
        return df_with_embeddings


class EmbeddingsClassifier:
    """A simple neural network classifier for medical findings using ELIXR embeddings."""
    
    # Implementation would go here - this is a placeholder for the complete implementation
    # which would include PyTorch model definition, training, and evaluation functions
    
    pass


def main():
    """Main function to run the demo."""
    # Initialize data loader
    data_loader = ELIXRDataLoader()
    
    # Authenticate with HuggingFace
    data_loader.authenticate_huggingface()
    
    # Download data
    file_paths = data_loader.download_data()
    
    # Prepare dataset for specified diagnosis
    # @param ["AIRSPACE_OPACITY", "FRACTURE", "PNEUMOTHORAX", "CONSOLIDATION", "EFFUSION", "PULMONARY_EDEMA", "ATELECTASIS", "CARDIOMEGALY"]
    # @param {type:"slider", min:50, max:400, step:5}
    diagnosis = "FRACTURE"
    max_cases_per_category = 400  # 400 or None
    
    df_labels = data_loader.prepare_dataset(
        file_paths["labels"],
        diagnosis,
        max_cases_per_category
    )
    
    # Load embeddings
    df_with_embeddings = data_loader.load_embeddings(file_paths["embeddings"], df_labels)
    
    print("Data preparation completed.")
    print(f"DataFrame shape: {df_with_embeddings.shape}")
    print(df_with_embeddings.head(), "\n")


    # @title Separate into training, validation, and testing sets.
    # @param {type:"slider", min:0.05, max:0.8, step:0.05}
    TEST_SPLIT = 0.1 

    df_train, df_validate = train_test_split(df_labels, test_size=TEST_SPLIT)

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_validate)}")

    # The rest of the implementation would include model training and evaluation
    # which was not part of the provided code snippet


if __name__ == "__main__":
    main()

"""
Prerequisites: Log in huggingface to review the conditions and access this (model content)[https://huggingface.co/google/cxr-foundation].

HuggingFace Token already set
Selected 400 positive and 400 negative cases
       image_id  patient_id  case_id  split  AIRSPACE_OPACITY  ...  EFFUSION  PULMONARY_EDEMA  ATELECTASIS  CARDIOMEGALY        dicom_file
0  00015845_007       15845        7  train               1.0  ...       0.0              0.0          0.0           0.0  00015845_007.dcm
1  00008774_005        8774        5  train               1.0  ...       0.0              0.0          0.0           0.0  00008774_005.dcm
2  00022600_001       22600        1  train               1.0  ...       0.0              0.0          0.0           0.0  00022600_001.dcm
3  00017324_012       17324       12  train               1.0  ...       0.0              0.0          0.0           0.0  00017324_012.dcm
4  00021311_000       21311        0  train               1.0  ...       0.0              0.0          1.0           0.0  00021311_000.dcm

[5 rows x 13 columns]

Total embeddings available: 2737
Embeddings shape: (4096,)

Cases selected: 792
Cases with available embeddings: 792
Data preparation completed.
DataFrame shape: (792, 14)
       image_id  patient_id  case_id  split  ...  ATELECTASIS  CARDIOMEGALY        dicom_file                                         embeddings
0  00015845_007       15845        7  train  ...          0.0           0.0  00015845_007.dcm  [-0.17128147, 0.008595931, -0.0005963545, 0.06...
1  00008774_005        8774        5  train  ...          0.0           0.0  00008774_005.dcm  [-0.19136983, -0.06103782, -0.027953472, 0.070...
2  00022600_001       22600        1  train  ...          0.0           0.0  00022600_001.dcm  [-0.14782454, -0.030647462, -0.13165669, 0.003...
3  00017324_012       17324       12  train  ...          0.0           0.0  00017324_012.dcm  [-0.104966134, 0.021651793, -0.14169557, -0.00...
4  00021311_000       21311        0  train  ...          1.0           0.0  00021311_000.dcm  [-0.113549404, 0.03401351, -0.2252285, 0.20830...

[5 rows x 14 columns]

Training set size: 712
Validation set size: 80
"""