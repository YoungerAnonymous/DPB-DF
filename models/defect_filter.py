import os
import pickle
import numpy as np
from .utils import create_index, patch_vote


class DefectFilter:
    def __init__(self, k, device, threshold=5):
        self.k = k
        self.device = device
        self.normal_patches = None
        self.threshold = threshold

    def run(self, embeddings: np.ndarray) -> np.ndarray:
        """Filter pseudo-defect patches in defect images.

        Args:
            embeddings: [N x D]
        """
        assert (self.normal_patches is not None), "Please load normal_features first"

        k = self.k
        index_space = create_index(self.normal_patches.shape[-1], self.device)
        index_space.add_with_ids(self.normal_patches, np.zeros(len(self.normal_patches), dtype=int))
        index_space.add_with_ids(embeddings, np.ones(len(embeddings), dtype=int))
        _, I = index_space.search(embeddings, k + 1)
        I = patch_vote(I[:, 1:])
        defects = embeddings[I == 1]

        # Try to adjust k_f if the number of defect patches is very low.
        while len(defects) < self.threshold:
            k -= 1
            if k < 0:
                raise ValueError("Running k_f is less than or equals 0!")
            _, I = index_space.search(embeddings, k + 1)
            I = patch_vote(I[:, 1:])
            defects = embeddings[I == 1]
        return defects
    
    def add(self, embeddings: np.ndarray):
        # Storing normal patches.
        if self.normal_patches is None:
            self.normal_patches = embeddings
        else:
            self.normal_patches = np.concatenate([self.normal_patches, embeddings])
            
    
    def load_from(self, root_path):
        with open(self.get_file_path(root_path), 'rb') as f:
            self.normal_patches = pickle.load(f)
    
    def save(self, root_path):
        with open(self.get_file_path(root_path), 'wb') as f:
            pickle.dump(self.normal_patches, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_file_path(root_path):
        return os.path.join(root_path, "criterion.pkl")