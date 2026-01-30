import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import webdataset as wd
import webdataset.utils as utils
from webdataset.pytorch import IterableDataset
from webdataset.shardlists import expand_source, split_by_node


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of URLs with resampling."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        seed=0,
        worker_seed=None,
        deterministic=False,
        max_urls=int(1e6),
        empty_check=True,
    ):
        """Initialize the ResampledShards.

        Args:
            urls: A list of URLs as a Python list or brace notation string.
            nshards (int): The number of shards to yield. Defaults to sys.maxsize.
            seed (int): The seed for random number generation.
            worker_seed (Callable or None): A function to generate worker-specific seeds.
            deterministic (bool): Whether to use deterministic sampling.
            max_urls (int): Maximum number of URLs to consider.
            empty_check (bool): Whether to check for empty URL list.

        Raises:
            ValueError: If empty_check is True and no shards are found.
        """
        super().__init__()
        self.urls = expand_source(urls, max_urls)
        if empty_check:
            if len(self.urls) == 0:
                raise ValueError(
                    "empty_check=True, but no shards found in ResampledShards"
                )
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.worker_seed = (
            utils.pytorch_worker_seed if worker_seed is None else worker_seed
        )
        self.deterministic = deterministic
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        self.epoch += 1
        seed = utils.make_seed(self.seed)
        if os.environ.get("WDS_SHOW_SEED", "0") == "1":
            print(f"# ResampledShards seed {seed}")

        self.rng = random.Random(seed)
        urls = deepcopy(self.urls)
        self.rng.shuffle(urls)

        if len(urls) == 0:
            raise ValueError("empty_check=True, but no shards found in ResampledShards")
        for _ in range(self.nshards):
            self.rng.shuffle(urls)

            for url in urls:
                yield dict(url=url)


class WebDatasetBuilder:
    def __init__(
        self,
        dataset_path: str,
    ):
        self.shard_paths = []
        self.shard_paths.extend([str(p) for p in Path(dataset_path).rglob("*.tar")])

    def build(
        self,
        batch_size: int = 1,
        shuffle_buffer: int = 3000,
        make_sample_fn=None,
        collate_fn=None,
    ):
        """Build a WebDataset pipeline for the requested stage.

        Args:
            batch_size (int, optional): Number of samples to combine into each
                batch. Defaults to 1.

        Returns:
            IterableDataset: A WebDataset iterable that yields batches of
                preprocessed samples for the specified stage.

        Raises:
            Exception: Propagates any exceptions raised by the underlying
                WebDataset library or file I/O operations (for example, when
                shard files are missing or corrupt).
        """
        webdataset_db = wd.WebDataset(
            self.shard_paths,
            nodesplitter=split_by_node,
            resampled=True,
            shardshuffle=False,
        )
        webdataset_db.pipeline[0] = ResampledShards(
            urls=webdataset_db.pipeline[0].urls,
            nshards=webdataset_db.pipeline[0].nshards,
            seed=webdataset_db.pipeline[0].seed,
            worker_seed=webdataset_db.pipeline[0].worker_seed,
            deterministic=webdataset_db.pipeline[0].deterministic,
        )
        webdataset_db = (
            webdataset_db.shuffle(shuffle_buffer)
            .decode()
            .map(make_sample_fn)
            .batched(batch_size, collation_fn=collate_fn)
        )
        webdataset_db.batch_size = batch_size

        return webdataset_db
