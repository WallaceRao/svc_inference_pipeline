import math
import random

from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch.utils.data import ConcatDataset


class BatchSampler(Sampler):
    """Base sampler providing sample indices for a concatenated dataset.
    Args:
        concat_dataset (ConcatDataset): a concatenated dataset consisting of all datasets
        dataset_list (list): a list of datasets to be concatenated
        shuffle (bool): whether to shuffle the batch indices or not (batch-level shuffle, which keeps the contents of each batch the same, but change the order of batches)
        aggregates_samples_of_similar_durations (bool): whether to sample the data with the similar duration or not
        drop_last (bool): whether to drop the last incomplete batch or not
        holistic_shuffle (bool): whether to shuffle the whole dataset or not, exclusive with shuffle and duration_alike
    Usage:
        There are two common usage of this class:
        1. Blend the samples from different datasets together, and sample them at random totally.
        (shuffle = False, aggregates_samples_of_similar_durations = False, drop_last = True/False, holistic_shuffle = True)
        Example:
            >>> list(BatchSampler(ConcatDataset([0, 1, ..., 8, 9], [10, 11, ..., 18, 19], [20, 21, ..., 28, 29]])))
            [[21, 28, 18], [6, 17, 7], [22, 15, 12], [0, 10, 27], [25, 8, 5], [16, 2, 23], [1, 26, 20], [3, 4, 24], [11, 14, 29], [9, 13, 19]]
        2. Aggregate samples having similar duration in the same batch, and shuffle the order of batches.
        (shuffle = True, aggregates_samples_of_similar_durations = True, drop_last = True/False, holistic_shuffle = False)
        Example:
            >>> list(BatchSampler(ConcatDataset([0, 1, ..., 8, 9], [10, 11, ..., 18, 19], [20, 21, ..., 28, 29]])))
            [[26, 27, 28], [0, 1, 2], [13, 14, 15], [10, 11, 12], [20, 21, 22], [16, 17, 18], [23, 24, 25], [6, 7, 8], [3, 4, 5]]
    """

    def __init__(
        self,
        cfg,
        concat_dataset,
        dataset_list,
        shuffle=True,
        aggregates_samples_of_similar_durations=True,
        drop_last=True,
        holistic_shuffle=False,
    ):
        self.dataset = concat_dataset  # this is a ConcatDataset
        self.dataset_list = dataset_list
        self.batch_size = cfg.train.batch_size

        self.aggregates_samples_of_similar_durations = (
            aggregates_samples_of_similar_durations
        )
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.holistic_shuffle = holistic_shuffle

        # holistic_shuffle and (shuffle, duration_alike) cannot be True at the same time
        assert not (
            self.holistic_shuffle
            and (self.shuffle or self.aggregates_samples_of_similar_durations)
        ), "holistic_shuffle and (shuffle, duration_alike) cannot be True at the same time"

        self.number_of_datasets = len(cfg.dataset)

        dataset_lens = [len(cur_dataset) for cur_dataset in concat_dataset.datasets]

        if not self.holistic_shuffle:
            self.batch_num_list = [
                math.ceil(cur_dataset_len / self.batch_size)
                if not self.drop_last
                else math.floor(cur_dataset_len / self.batch_size)
                for cur_dataset_len in dataset_lens
            ]
        else:
            self.batch_num = (
                math.ceil(sum(dataset_lens) / self.batch_size)
                if not self.drop_last
                else math.floor(sum(dataset_lens) / self.batch_size)
            )

    def __len__(self):
        # return the number of batches in an epoch
        return (
            sum(self.batch_num_list) if not self.holistic_shuffle else self.batch_num
        )  # sum the number of batches in different datasets

    def __iter__(self):
        if not self.holistic_shuffle:
            samplers_list = []  # this is a list of samplers for different datasets
            sampler_iterators = []  # this is a list of iterators for different samplers
            # get the samplers for different datasets
            for dataset_idx in range(self.number_of_datasets):
                cur_dataset = self.dataset_list[dataset_idx]
                if self.aggregates_samples_of_similar_durations:
                    sampler = SequentialSampler(cur_dataset)
                else:
                    sampler = RandomSampler(cur_dataset)
                samplers_list.append(sampler)
                cur_sampler_iterator = sampler.__iter__()
                sampler_iterators.append(cur_sampler_iterator)

            # get the initial indices of different datasets like [0, 100, 200, 300, 400]
            init_indices = [0] + self.dataset.cumulative_sizes[:-1]

            output_batches_list = []
            for dataset_idx in range(self.number_of_datasets):
                for _ in range(self.batch_num_list[dataset_idx]):
                    cur_batch = []
                    for _ in range(self.batch_size):
                        try:
                            cur_sample_idx = next(sampler_iterators[dataset_idx])
                            cur_sample = cur_sample_idx + init_indices[dataset_idx]
                            cur_batch.append(cur_sample)
                        except StopIteration:
                            break
                            # sampler_iterators[dataset_idx] = samplers_list[
                            #     dataset_idx
                            # ].__iter__()
                            # cur_sample_idx = next(sampler_iterators[dataset_idx])
                            # cur_sample = cur_sample_idx + init_indices[dataset_idx]
                            # cur_batch.append(cur_sample)
                    output_batches_list.append(cur_batch)

            if self.shuffle:
                random.shuffle(output_batches_list)
            assert (
                len(output_batches_list) == self.__len__()
            ), "expected {}, got {}".format(self.__len__(), len(output_batches_list))
            return iter(output_batches_list)
        else:
            sample_list = list(RandomSampler(self.dataset))
            output_batches_list = []
            for _ in range(self.batch_num):
                cur_batch = []
                for _ in range(self.batch_size):
                    try:
                        cur_sample = sample_list.pop()
                        cur_batch.append(cur_sample)
                    except IndexError:
                        break
                output_batches_list.append(cur_batch)
            assert (
                len(output_batches_list) == self.__len__()
            ), "expected {}, got {}".format(self.__len__(), len(output_batches_list))
            # print(output_batches_list)
            return iter(output_batches_list)
