import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2
import tqdm
from .defect_filter import DefectFilter
from .utils import create_index, patch_vote
from .extractor import *


class DPB:
    def __init__(
            self, 
            k_e: int=5, 
            k_f: int=40, 
            d_n: float=0.1, 
            d_d: float=0.5, 
            lambda_n=0.998,
            lambda_d=1,
            input_shape=(3, 224, 224),
            device=torch.device("cpu"),
            layers_to_extract_from=("layer2", "layer3"),
        ):
        """
        DPB-DF defect detection class.

        Args:
            k1: The number of searching the nearest neighbors in patch-level weighted vote.
            k2: The number of searching the nearest neighbors in defect filter.
            dsr_n: Down-sampling rate for normal images.
            dsr_d: Down-sampling rate for defect images.
            input_shape: The shape of training or test images.
            device: torch.device
            layers_to_extract_from: The layers where get the feature map.
        """
        self.k_e = k_e
        self.k_f = k_f
        self.d_n = d_n
        self.d_d = d_d
        self.lambda_n =lambda_n
        self.lambda_d = lambda_d
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device

        self.banks = defaultdict(list)
        self.backbone = wide_resnet50_2(pretrained=True)
        self.patch_maker = PatchMaker(3, stride=1)
        self.patch_aggregation = torch.nn.ModuleDict({})
        
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, layers_to_extract_from, self.device
        )
        self.patch_aggregation["feature_aggregator"] = feature_aggregator

        if isinstance(input_shape, int):
            input_shape = (3, input_shape, input_shape)
        feature_dimension = feature_aggregator.feature_dimensions(input_shape)
        preprocessing = Preprocessing(feature_dimension, 1024)
        self.patch_aggregation["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(1024, self.device)
        self.patch_aggregation["preadapt_aggregator"] = preadapt_aggregator

        self.downsampler_n = ApproximateGreedyCoresetSampler(self.d_n, self.device)
        self.downsampler_d = ApproximateGreedyCoresetSampler(self.d_d, self.device)

        self.defect_filter = DefectFilter(self.k_f, self.device)

    def pretrain(self, normal_patches):
        self.defect_filter.add(normal_patches)
        normal_patches = self.downsampler_n.run(normal_patches)
        self.banks[0] += normal_patches.tolist()

    def update(self, defect_data: torch.utils.data.DataLoader):
        """Incrementally updates the patch bank."""
        patches, labels = self.to_patches(defect_data)
        categories = np.unique(labels)
        if np.any(categories == 0):
            normal_patches = patches[labels==0]
            self.pretrain(normal_patches)
            
        for category in categories[categories!=0]:
            unknown_patches = patches[labels==category]
            coresamp_patches = self.downsampler_d.run(unknown_patches)
            defect_patches = self.defect_filter.run(coresamp_patches)
            self.banks[category] += defect_patches.tolist()
    
    def predict(self, test_images):
        assert len(test_images.shape) == 4, "The image shape must be (b, c, h, w)."
        _ = self.patch_aggregation.eval()
        batchsize = test_images.shape[0]
        embeddings = self._image_to_features(test_images)
        preds_p = self.elatsic_weighted_feedback(np.asarray(embeddings))
        preds_p = self.patch_maker.unpatch_scores(preds_p, batchsize)
        preds_i = patch_vote(preds_p, exception=0)
        return preds_i.tolist()

    def evaluate(self, test_data: torch.utils.data.DataLoader):
        preds = []
        labels = []
        with tqdm.tqdm(test_data, desc="Inferring...", leave=False, position=1) as data_iterator:
            for image, label in data_iterator:
                preds.extend(self.predict(image))
                labels.extend(label)
        return preds, labels

    @property
    def lambda_values(self):
        return [self.lambda_n] + [self.lambda_d] * (len(self.banks) - 1)
    
    def elatsic_weighted_feedback(self, query_embeddings: np.ndarray):
        nearest_dst = []
        for category, patches in self.banks.items():
            patches = np.array(patches).astype("float32")
            index_space = create_index(query_embeddings.shape[-1], self.device)
            index_space.add_with_ids(patches, np.full(len(patches), category, dtype="int64"))
            nearest_dst.append(index_space.search(query_embeddings, self.k_e)[0])
            del index_space
        nearest_dst = np.stack(nearest_dst, axis=1)
        scores = np.sum(np.expand_dims(self.lambda_values, (0, 2)) / nearest_dst, axis=-1)
        return np.argmax(scores, axis=-1)

    def _image_to_features(self, input_image):
        with torch.no_grad():
            input_image = input_image.to(torch.float).to(self.device)
            return self._embed(input_image)

    def to_patches(self, data: torch.utils.data.DataLoader):
        patches = []
        labels = []
        _ = self.patch_aggregation.eval()
        with tqdm.tqdm(
            data, desc="Transforming images to patch embeddings...", position=1, leave=False
        ) as data_iterator:
            for image, label in data_iterator:
                patch = self._image_to_features(image)
                patches.append(patch)
                assert len(patch) % len(label) == 0
                labels.append(label.repeat_interleave(len(patch) // len(label)).cpu().numpy())
        return np.concatenate(patches, axis=0), np.concatenate(labels, axis=0)

    @staticmethod
    def get_file_path(root_path):
        return os.path.join(root_path, "patch_bank.pkl")

    def save(self, save_path, save_df=True):
        with open(self.get_file_path(save_path), "wb") as f:
            pickle.dump(self.banks, f, pickle.HIGHEST_PROTOCOL)
        if save_df:
            self.defect_filter.save(save_path)

    def load(self, load_path, load_df=True):
        with open(self.get_file_path(load_path), "rb") as f:
            self.banks = pickle.load(f)
        if load_df:
            self.defect_filter.load_from(load_path)


    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.patch_aggregation["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.patch_aggregation["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        # output = [bs * num_of_patch, f]
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.patch_aggregation["preprocessing"](features)
        features = self.patch_aggregation["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    