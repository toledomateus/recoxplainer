from typing import List, Optional
import numpy as np
import pandas as pd
import os


class DataReader:

    def __init__(self,
                 filepath_or_buffer: str,
                 sep: str,
                 names: list,
                 groups_filepath: List[str],
                 skiprows: int = 0,
                 ):
        
        self.filepath_or_buffer = filepath_or_buffer
        self.sep = sep
        self.names = names
        self.skiprows = skiprows
        self.groups_filepath = groups_filepath

        self._dataset = None
        self._num_user = None
        self._num_item = None
        self.dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = pd.read_csv(filepath_or_buffer=self.filepath_or_buffer,
                                        sep=self.sep,
                                        names=self.names,
                                        skiprows=self.skiprows,
                                        engine='python')
            self._num_item = int(self._dataset[['itemId']].nunique())
            self._num_user = int(self._dataset[['userId']].nunique())

        return self._dataset

    @dataset.setter
    def dataset(self, new_data):
        self._dataset = new_data

    def make_consecutive_ids_in_dataset(self):
        # TODO: create mapping function
        dataset = self.dataset.rename({
            "userId": "user_id",
            "itemId": "item_id"
        }, axis=1)

        user_id = dataset[['user_id']].drop_duplicates().reindex()
        num_user = len(user_id)

        user_id['userId'] = np.arange(num_user)
        self._dataset = pd.merge(
            dataset, user_id,
            on=['user_id'], how='left')

        item_id = dataset[['item_id']].drop_duplicates()
        num_item = len(item_id)
        item_id['itemId'] = np.arange(num_item)

        self._dataset = pd.merge(
            self._dataset, item_id,
            on=['item_id'], how='left')

        self.original_user_id = user_id.set_index('userId')
        self.original_item_id = item_id.set_index('itemId')
        self.new_user_id = user_id.set_index('user_id')
        self.new_item_id = item_id.set_index('item_id')

        self._dataset = self.dataset[
            ['userId', 'itemId', 'rating', 'timestamp']
        ]

        self._dataset.userId = [int(i) for i in self._dataset.userId]
        self._dataset.itemId = [int(i) for i in self._dataset.itemId]

    def binarize(self, binary_threshold=1):
        """binarize into 0 or 1, imlicit feedback"""

        self._dataset[self._dataset['rating'] > binary_threshold].rating = 1
        self._dataset[self._dataset['rating'] <= binary_threshold].rating = 0

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def get_original_user_id(self, u):
        if isinstance(u, int):
            return self.original_user_id.loc[u].user_id

        return list(self.original_user_id.loc[u].user_id)

    def get_original_item_id(self, i):
        if isinstance(i, int):
            return self.original_item_id.loc[i].item_id

        return list(self.original_item_id.loc[i].item_id)

    def get_new_user_id(self, u):
        if isinstance(u, int):
            return self.new_user_id.loc[u].userId

        return list(self.new_user_id.loc[u].userId)

    def get_new_item_id(self, i):
        if isinstance(i, int):
            return self.new_item_id.loc[i].itemId

        return list(self.new_item_id.loc[i].itemId)

    def _get_group_filepath(self, filename: str) -> Optional[str]:
        """
        Get a specific group file path by matching the filename.

        Args:
            filename (str): The name of the file to search for.

        Returns:
            str: The matched file path.

        Raises:
            ValueError: Error: File does not exist
            ValueError: No file found containing '{filename}' in its name.
        """
        for path in self.groups_filepath:
            if filename in path:  # Check if filename is part of the path
                filepath = os.path.abspath(path)
                if os.path.exists(filepath):
                    return filepath
                else:
                    raise ValueError(f"Error: File does not exist: {filepath}")

        raise ValueError(f"Error: No file found containing '{filename}' in its name.")

    def read_groups(self, filename: str) -> List[str]:
        """
        Method to read group IDs from a specified file.

        Args:
            filepath (str): Path to the file containing group IDs.

        Returns:
            List of group IDs.
        """
        if not filename:
            raise ValueError("Groups path not specified in configuration")

        filepath = self._get_group_filepath(filename)

        with open(filepath, "r") as f:
            groups = [x.strip() for x in f.readlines()]
        return groups
            
    def parse_group_members(self, group: str) -> List[int]:
        """
        Parse group ID to get member IDs.

        Args:
            group: Group ID string

        Returns:
            List of member IDs
        """
        group = group.strip()
        members = group.split('_')
        return [int(m) for m in members]
    
    def get_items_for_group_recommendation(self, data: pd.DataFrame, item_ids: np.ndarray, group: List[int]) -> np.ndarray:
        """
        Get items for group recommendation (those not interacted with by any group member).

        Args:
            data: DataFrame with interaction data
            item_ids: Array of all item IDs
            group: List of group member IDs

        Returns:
            Array of item IDs not interacted with by any group member
        """
        item_ids_group = data.loc[data.userId.isin(group), "itemId"]
        return np.setdiff1d(item_ids, item_ids_group)