import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import webdataset as wd
import webdataset.utils as utils
from webdataset.pytorch import IterableDataset
from webdataset.shardlists import expand_source, split_by_node


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of URLs with resampling."""

    def __init__(
        self,
        urls: Union[str, List[str]],
        nshards: int = sys.maxsize,
        seed: int = 0,
        worker_seed: Optional[Callable[[], int]] = None,
        max_urls: int = int(1e6),
        empty_check: bool = True,
    ) -> None:
        """Initialize the ResampledShards.

        Args:
            urls: A list of URLs as a Python list or brace notation string.
            nshards (int): The number of shards to yield. Defaults to sys.maxsize.
            seed (int): The seed for random number generation.
            worker_seed (Callable or None): A function to generate worker-specific seeds.
            max_urls (int): Maximum number of URLs to consider.
            empty_check (bool): Whether to check for empty URL list.

        Raises:
            ValueError: If empty_check is True and no shards are found.
        """
        super().__init__()
        self.urls: List[str] = expand_source(urls, max_urls)
        if empty_check:
            if len(self.urls) == 0:
                raise ValueError("empty_check=True, but no shards found in ResampledShards")
        assert isinstance(self.urls[0], str)
        self.nshards: int = nshards
        self.worker_seed: Callable[[], int] = utils.pytorch_worker_seed if worker_seed is None else worker_seed
        self.seed: int = seed
        self.epoch: int = -1

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        self.epoch += 1
        seed = utils.make_seed(self.seed, self.epoch)
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
    """A builder class for creating WebDataset pipelines.

    This class provides a fluent interface for constructing WebDataset pipelines
    with support for multiple data sources and custom preprocessing.

    Args:
        dataset_path: Either a single path string or list of paths to tar files or
            directories containing tar files.
    """

    def __init__(
        self,
        dataset_path: Union[str, List[str]],
    ) -> None:
        self.shard_paths: List[str] = self._expand_paths(dataset_path)
        assert len(self.shard_paths) > 0, f"No .tar files found in {dataset_path}"
        self.joint_url_mapping_fns: List[Callable[[str], str]] = []

    def _expand_paths(self, dataset_path: Union[str, List[str]]) -> List[str]:
        """Expand dataset paths to a list of tar file paths.

        Args:
            dataset_path: Either a single path string (which can be a tar file or directory)
                or a list of such paths.

        Returns:
            A list of absolute paths to tar files.
        """
        if isinstance(dataset_path, list):
            for path in dataset_path:
                if path.endswith(".tar"):
                    return expand_source(path)
                else:
                    return [str(p) for p in Path(path).rglob("*.tar")]
        else:
            if dataset_path.endswith(".tar"):
                return expand_source(dataset_path)
            else:
                return [str(p) for p in Path(dataset_path).rglob("*.tar")]

    def join(self, url_mapping_fn: Callable[[str], str]) -> "WebDatasetBuilder":
        """Add a URL mapping function to join additional data sources.

        This method allows you to join multiple WebDataset sources by providing
        a function that maps URLs from the main dataset to URLs in additional datasets.

        Args:
            url_mapping_fn: A callable that takes a URL string and returns a corresponding
                URL string for an additional data source.

        Returns:
            Self, for method chaining.
        """
        self.joint_url_mapping_fns.append(url_mapping_fn)
        return self

    def build(
        self,
        batch_size: int = 1,
        shuffle_buffer: int = 3000,
        make_sample_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> IterableDataset:
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
        )

        if len(self.joint_url_mapping_fns) > 0:

            def join_samples(src: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
                """Join samples from multiple data sources.

                Args:
                    src: Iterator over samples from the main dataset.

                Yields:
                    Samples with data merged from additional sources.
                """
                current_main_url: Optional[str] = None
                additional_srcs: List[Iterator[Dict[str, Any]]] = []

                for sample in src:
                    if current_main_url != sample["__url__"]:
                        current_main_url = sample["__url__"]
                        additional_srcs = []

                        for url_mapping_fn in self.joint_url_mapping_fns:
                            additional_url = url_mapping_fn(sample["__url__"])

                            additional_src = wd.WebDataset(
                                additional_url,
                                shardshuffle=False,
                                nodesplitter=None,
                                workersplitter=None,
                            )
                            additional_srcs.append(iter(additional_src))

                    try:
                        for additional_src in additional_srcs:
                            additional_sample = next(additional_src)
                            assert sample["__key__"] == additional_sample["__key__"], (
                                f"Key mismatch between main sample {sample['__key__']} and "
                                f"additional sample {additional_sample['__key__']}"
                            )
                            sample.update(additional_sample)
                    except StopIteration:
                        current_main_url = sample["__url__"]
                        additional_srcs = []
                        for url_mapping_fn in self.joint_url_mapping_fns:
                            additional_url = url_mapping_fn(sample["__url__"])

                            additional_src = wd.WebDataset(
                                additional_url,
                                shardshuffle=False,
                                nodesplitter=None,
                                workersplitter=None,
                            )
                            additional_srcs.append(iter(additional_src))

                        for additional_src in additional_srcs:
                            additional_sample = next(additional_src)
                            assert sample["__key__"] == additional_sample["__key__"], (
                                f"Key mismatch between main sample {sample['__key__']} and "
                                f"additional sample {additional_sample['__key__']}"
                            )
                            sample.update(additional_sample)
                    except Exception as e:
                        raise e
                    yield sample

            webdataset_db = webdataset_db.compose(join_samples)

        if shuffle_buffer > 0:
            webdataset_db = webdataset_db.shuffle(shuffle_buffer)

        webdataset_db = webdataset_db.decode().map(make_sample_fn).batched(batch_size, collation_fn=collate_fn)
        webdataset_db.batch_size = batch_size

        return webdataset_db
