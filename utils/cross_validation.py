from sklearn.model_selection import KFold
import numpy as np
import torch

class KFoldCV:
    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def get_fold_indices(self, dataset_size):
        return list(self.kf.split(np.arange(dataset_size)))

    @staticmethod
    def create_fold_dataloaders(dataset, train_idx, val_idx, train_batch_size,
                                val_batch_size, num_workers):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_subsampler,
            num_workers=num_workers
        )

        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=val_batch_size,
            sampler=val_subsampler,
            num_workers=num_workers
        )

        return train_loader, val_loader


if __name__ == "__main__":
    fold1 = KFoldCV()
    # 5 different 80-20 splits
    indices = fold1.get_fold_indices(dataset_size=1000)
    # each split has 800 train, 200 val indices

    for index in indices:
        print(f"current index : {index}")
