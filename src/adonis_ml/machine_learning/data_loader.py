import logging
import os
import pickle

import pandas as pd
from pandas.util import hash_pandas_object
import requests

requests.packages.urllib3.disable_warnings()

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config):
        self.config = config

        self._parse_args()

    def _parse_args(self):
        self.environment = self.config.environment
        self.hospital = self.config.hospital

        self.path_to_checkpoint = self.config.mlflow["path_to_checkpoint"]

    def load_adonis_feature_dataset(self):
        """
        Load and verify the Adonis feature dataset for a specific hospital.

        This function performs the following steps:
        1. Loads a feature dataset (Parquet file) corresponding to the given hospital.
        2. Verifies data integrity by checking that the hash of the loaded DataFrame 
        matches the stored hash (ensuring no corruption or mismatch).
        3. Loads the list of variables used in the dataset from a pickle file.

        Returns
        -------
        df : pd.DataFrame
            The verified feature dataset for the specified hospital, with valid `stay_id`s only.
        
        list_of_variables : list
            The list of variable names used in the dataset.

        Raises
        ------
        AssertionError
            If the computed hash of the loaded dataset does not match the stored hash,
            indicating potential data corruption or inconsistency.
        
        FileNotFoundError
            If any of the required files (dataset, hash, or list of variables) are missing.
        """
        
        # Define file paths
        parquet_path = os.path.join(self.path_to_checkpoint, f"df_cases_controls_{self.hospital}.parquet")
        hash_path = os.path.join(self.path_to_checkpoint, f"hash_{self.hospital}.pkl")
        variables_path = os.path.join(self.path_to_checkpoint, "list_variables.pkl")

        # 1. Load dataset
        logger.info("Loading dataset from: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded dataset shape: %s", df.shape)

        # 2. Verify data integrity
        with open(hash_path, "rb") as f:
            hash_correct = pickle.load(f)

        hash_to_check = hash_pandas_object(df, index=True).values
        assert (hash_correct == hash_to_check).all(), "Dataset hash mismatch — file may be corrupted or modified."
        logger.info("Hash check passed for hospital: %s", self.hospital)

        # Clean dataset
        df = df.dropna(subset=["stay_id"])
        logger.info("Dataset shape after dropna(stay_id): %s", df.shape)

        # 3. Load variable list
        with open(variables_path, "rb") as f:
            list_of_variables = pickle.load(f)

        return df, list_of_variables

    def load_demog(self):
        demog = pd.read_parquet(
            os.path.join(
                self.path_to_checkpoint, f"demographics.parquet"
            )
        )
        logger.info("Loaded demog shape: %s", demog.shape)
        return demog
    
    def load_fit_models(self):
        fit_models = {}

        for file_name in os.listdir(os.path.join(self.path_to_checkpoint)):
            if "sklearn" in file_name:
                with open(os.path.join(self.path_to_checkpoint, file_name), "rb") as f:
                    results_sklearn = pickle.load(f)
                model_type = file_name.split("_")[0]
                fit_models[model_type] = results_sklearn["fit_models"]

        return fit_models


