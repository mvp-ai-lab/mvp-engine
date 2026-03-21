"""Test cases for the logging system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from mvp_engine.utils.log.metric import Metric, MetricAggregator


class TestMetric:
    """Test cases for the Metric class."""

    def test_update_with_int(self):
        """Test updating metric with integer values."""
        metric = Metric(accumulation_size=5)
        metric.update(10)
        metric.update(20)
        assert metric.mean() == 15.0

    def test_update_with_float(self):
        """Test updating metric with float values."""
        metric = Metric(accumulation_size=5)
        metric.update(1.5)
        metric.update(2.5)
        metric.update(3.0)
        assert metric.mean() == pytest.approx(2.333, rel=0.01)

    def test_update_with_tensor(self):
        """Test updating metric with torch tensor values."""
        metric = Metric(accumulation_size=5)
        metric.update(torch.tensor(5.0))
        metric.update(torch.tensor(10.0))
        assert metric.mean() == 7.5

    def test_update_with_string(self):
        """Test updating metric with string values."""
        metric = Metric(accumulation_size=5)
        metric.update("first")
        metric.update("second")
        metric.update("last")
        assert metric.mean() == "last"
        assert metric.sum() == "last"

    def test_update_with_none(self):
        """Test updating metric with None values."""
        metric = Metric(accumulation_size=5)
        metric.update(10)
        metric.update(None)
        metric.update(20)
        # None values should be ignored in aggregation
        assert metric.mean() == 15.0

    def test_accumulation_size_limit(self):
        """Test that buffer respects accumulation size."""
        metric = Metric(accumulation_size=3)
        for i in range(10):
            metric.update(float(i))
        # Should only keep last 3 values: 7, 8, 9
        assert metric.mean() == 8.0

    def test_sum(self):
        """Test sum aggregation."""
        metric = Metric(accumulation_size=5)
        metric.update(1.0)
        metric.update(2.0)
        metric.update(3.0)
        assert metric.sum() == 6.0

    def test_nan_handling_with_support(self):
        """Test NaN handling when support_nan=True."""
        metric = Metric(accumulation_size=5, support_nan=True)
        metric.update(1.0)
        metric.update(float("nan"))
        metric.update(2.0)
        # NaN should be included, result will be nan
        result = metric.mean()
        assert torch.isnan(torch.tensor(result)).item()

    def test_nan_handling_without_support(self):
        """Test NaN handling when support_nan=False."""
        metric = Metric(accumulation_size=5, support_nan=False)
        metric.update(1.0)
        metric.update(float("nan"))  # Should be replaced with last value (1.0)
        metric.update(2.0)
        assert metric.mean() == pytest.approx(1.333, rel=0.01)

    def test_clear(self):
        """Test clearing the metric buffer."""
        metric = Metric(accumulation_size=5)
        metric.update(10)
        metric.update(20)
        metric.clear()
        assert metric.mean() == 0.0

    def test_empty_buffer(self):
        """Test behavior with empty buffer."""
        metric = Metric(accumulation_size=5)
        assert metric.mean() == 0.0
        assert metric.sum() == 0.0

    def test_invalid_type(self):
        """Test that invalid types raise TypeError."""
        metric = Metric(accumulation_size=5)
        with pytest.raises(TypeError):
            metric.update([1, 2, 3])


class TestMetricAggregator:
    """Test cases for the MetricAggregator class."""

    def test_add_metric(self):
        """Test adding a metric."""
        aggregator = MetricAggregator(default_interval=5)
        aggregator.add("loss")
        assert "loss" in aggregator._metrics

    def test_add_duplicate_metric(self):
        """Test that adding duplicate metric is ignored."""
        aggregator = MetricAggregator(default_interval=5)
        aggregator.add("loss", interval=10)
        aggregator.add("loss", interval=20)  # Should be ignored
        assert aggregator._metrics["loss"]["interval"] == 10

    def test_update_metrics(self):
        """Test updating metrics."""
        aggregator = MetricAggregator(default_interval=5)
        aggregator.update({"loss": 0.5, "accuracy": 0.9})
        assert "loss" in aggregator._metrics
        assert "accuracy" in aggregator._metrics

    def test_collect_before_interval(self):
        """Test that collect returns empty before interval is reached."""
        aggregator = MetricAggregator(default_interval=5)
        aggregator.update({"loss": 0.5})
        aggregator.update({"loss": 0.4})
        collected = aggregator.collect(["loss"])
        assert collected == {}

    def test_collect_at_interval(self):
        """Test that collect returns values when interval is reached."""
        aggregator = MetricAggregator(default_interval=3)
        for i in range(3):
            aggregator.update({"loss": float(i)})
        collected = aggregator.collect(["loss"])
        assert "loss" in collected
        assert collected["loss"] == pytest.approx(1.0, rel=0.01)  # mean of 0, 1, 2

    def test_collect_all(self):
        """Test collecting all metrics."""
        aggregator = MetricAggregator(default_interval=5)
        aggregator.update({"loss": 0.5, "accuracy": 0.9})
        collected = aggregator.collect_all()
        assert "loss" in collected
        assert "accuracy" in collected

    def test_string_metric(self):
        """Test string metric handling."""
        aggregator = MetricAggregator(default_interval=2)
        aggregator.update({"eta": "1:30:00"})
        aggregator.update({"eta": "1:25:00"})
        collected = aggregator.collect(["eta"])
        assert collected["eta"] == "1:25:00"

    def test_mixed_metrics(self):
        """Test mixed numeric and string metrics."""
        aggregator = MetricAggregator(default_interval=2)
        aggregator.update({"loss": 0.5, "eta": "1:30:00"})
        aggregator.update({"loss": 0.3, "eta": "1:25:00"})
        collected = aggregator.collect(["loss", "eta"])
        assert collected["loss"] == pytest.approx(0.4, rel=0.01)
        assert collected["eta"] == "1:25:00"


class TestLogger:
    """Test cases for the Logger class."""

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_logger_init(self, mock_world_size):
        """Test logger initialization."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend], interval=10)
        assert logger.step == 0
        assert len(logger.backends) == 1

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_log_config(self, mock_world_size):
        """Test logging configuration."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend])
        config = {"lr": 0.001, "batch_size": 32}
        logger.log_config(config)
        mock_backend.log_config.assert_called_once_with(config)

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_log_metrics(self, mock_world_size):
        """Test logging metrics."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend], interval=2)
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.log_metrics({"loss": 0.3}, step=2)
        # Should be called after interval is reached
        assert mock_backend.log_metrics.called

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_info_warning_error(self, mock_world_size):
        """Test info, warning, and error logging."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend])

        logger.info("info message")
        mock_backend.info.assert_called_once_with("info message")

        logger.warning("warning message")
        mock_backend.warning.assert_called_once_with("warning message")

        logger.error("error message")
        mock_backend.error.assert_called_once_with("error message")

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_logger_level_filters_messages(self, mock_world_size):
        """Test logger-level filtering for debug/info/warn/error methods."""
        from mvp_engine.utils.log.logger import Logger, LogLevel

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend], level=LogLevel.WARNING)

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

        mock_backend.debug.assert_not_called()
        mock_backend.info.assert_not_called()
        mock_backend.warning.assert_called_once_with("warning message")
        mock_backend.error.assert_called_once_with("error message")

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_destroy(self, mock_world_size):
        """Test logger destruction."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend])
        logger.destroy()
        mock_backend.destroy.assert_called_once()

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_add_metric(self, mock_world_size):
        """Test adding a single metric."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend])
        logger.add_metric("custom_metric", interval=10)
        assert "custom_metric" in logger.metrics._metrics

    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_add_metrics(self, mock_world_size):
        """Test adding multiple metrics."""
        from mvp_engine.utils.log.logger import Logger

        mock_backend = MagicMock()
        logger = Logger(backends=[mock_backend])
        logger.add_metrics(["metric1", "metric2", "metric3"])
        assert "metric1" in logger.metrics._metrics
        assert "metric2" in logger.metrics._metrics
        assert "metric3" in logger.metrics._metrics


class TestFileBackend:
    """Test cases for the FileBackend class."""

    @patch("mvp_engine.utils.log.backend.file.is_main_process", return_value=True)
    def test_file_backend_init(self, mock_is_main):
        """Test FileBackend initialization."""
        from mvp_engine.utils.log.backend.file import FileBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(id="test", path=Path(tmpdir))
            assert backend.enable
            assert backend.log_file is not None
            backend.destroy()

    @patch("mvp_engine.utils.log.backend.file.is_main_process", return_value=True)
    def test_file_backend_log_metrics(self, mock_is_main):
        """Test FileBackend log_metrics."""
        from mvp_engine.utils.log.backend.file import FileBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(id="test", path=Path(tmpdir))
            backend.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100, epoch=1)
            backend.destroy()

            log_file = Path(tmpdir) / "log_test.log"
            content = log_file.read_text()
            assert "loss" in content
            assert "accuracy" in content
            assert "Step" in content

    @patch("mvp_engine.utils.log.backend.file.is_main_process", return_value=True)
    def test_file_backend_info_warning_error(self, mock_is_main):
        """Test FileBackend info, warning, error logging."""
        from mvp_engine.utils.log.backend.file import FileBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(id="test", path=Path(tmpdir))
            backend.info("info message")
            backend.warning("warning message")
            backend.error("error message")
            backend.destroy()

            log_file = Path(tmpdir) / "log_test.log"
            content = log_file.read_text()
            assert "INFO" in content
            assert "WARN" in content
            assert "ERROR" in content

    @patch("mvp_engine.utils.log.backend.file.is_main_process", return_value=False)
    def test_file_backend_disabled_on_non_main_process(self, mock_is_main):
        """Test FileBackend is disabled on non-main process."""
        from mvp_engine.utils.log.backend.file import FileBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileBackend(id="test", path=Path(tmpdir))
            assert not backend.enable


class TestTerminalBackend:
    """Test cases for the TerminalBackend class."""

    @patch("mvp_engine.utils.log.backend.terminal.is_main_process", return_value=True)
    def test_terminal_backend_init(self, mock_is_main):
        """Test TerminalBackend initialization."""
        from mvp_engine.utils.log.backend.terminal import TerminalBackend

        backend = TerminalBackend(id="test")
        assert backend.enable
        assert backend.console is not None

    @patch("mvp_engine.utils.log.backend.terminal.is_main_process", return_value=True)
    def test_terminal_backend_log_metrics(self, mock_is_main):
        """Test TerminalBackend log_metrics."""
        from mvp_engine.utils.log.backend.terminal import TerminalBackend

        backend = TerminalBackend(id="test")
        backend.console = MagicMock()
        backend.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100, epoch=1)
        assert backend.console.print.called

    @patch("mvp_engine.utils.log.backend.terminal.is_main_process", return_value=True)
    def test_terminal_backend_info(self, mock_is_main):
        """Test TerminalBackend info logging."""
        from mvp_engine.utils.log.backend.terminal import TerminalBackend

        backend = TerminalBackend(id="test")
        backend.console = MagicMock()
        backend.info("test message")
        backend.console.print.assert_called()

    @patch("mvp_engine.utils.log.backend.terminal.is_main_process", return_value=False)
    def test_terminal_backend_disabled_on_non_main_process(self, mock_is_main):
        """Test TerminalBackend is disabled on non-main process."""
        from mvp_engine.utils.log.backend.terminal import TerminalBackend

        backend = TerminalBackend(id="test")
        assert not backend.enable


class TestWandbBackend:
    """Test cases for the WandbBackend class."""

    @patch("mvp_engine.utils.log.backend.wandb_log.is_main_process", return_value=True)
    def test_wandb_backend_init(self, mock_is_main):
        """Test WandbBackend initialization."""
        from mvp_engine.utils.log.backend.wandb_log import WandbBackend

        backend = WandbBackend(id="test_run_1", project="test_project")
        assert backend.enable
        backend.destroy()

    @patch("mvp_engine.utils.log.backend.wandb_log.is_main_process", return_value=True)
    def test_wandb_backend_log_metrics(self, mock_is_main):
        """Test WandbBackend log_metrics."""
        from mvp_engine.utils.log.backend.wandb_log import WandbBackend

        backend = WandbBackend(id="test_run_2", project="test_project")
        metrics = {"loss": 0.5, "accuracy": 0.9, "eta": "1h"}
        backend.log_metrics(metrics, step=100, epoch=2)
        backend.destroy()

    @patch("mvp_engine.utils.log.backend.wandb_log.is_main_process", return_value=False)
    def test_wandb_backend_disabled_on_non_main_process(self, mock_is_main):
        """Test WandbBackend is disabled on non-main process."""
        from mvp_engine.utils.log.backend.wandb_log import WandbBackend

        backend = WandbBackend(id="test_run_3", project="test_project")
        assert not backend.enable
        backend.log_metrics({"acc": 1.0}, step=1)
        backend.destroy()


class TestLoggerInitAndSimpleInfo:
    """Test cases for init_logger and simple_info helpers."""

    @patch.dict("os.environ", {"LOG_LEVEL": "error"}, clear=False)
    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_init_logger_reads_env_level(self, mock_world_size):
        """Test init_logger picks level from LOG_LEVEL when level arg is omitted."""
        from mvp_engine.utils.log import init_logger
        from mvp_engine.utils.log.logger import LogLevel

        mock_backend = MagicMock()
        logger = init_logger([mock_backend])

        assert logger.level == LogLevel.ERROR
        logger.destroy()

    @patch.dict("os.environ", {"LOG_LEVEL": "invalid_level"}, clear=False)
    @patch("mvp_engine.utils.log.Console")
    @patch("mvp_engine.utils.log.logger.get_world_size", return_value=1)
    def test_init_logger_invalid_env_level_falls_back_to_info(self, mock_world_size, mock_console_cls):
        """Test invalid LOG_LEVEL env value falls back to info and emits warning."""
        from mvp_engine.utils.log import init_logger
        from mvp_engine.utils.log.logger import LogLevel

        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        mock_backend = MagicMock()
        logger = init_logger([mock_backend])

        assert logger.level == LogLevel.INFO
        assert mock_console.print.called
        logger.destroy()

    @patch.dict("os.environ", {"LOG_LEVEL": "invalid_level"}, clear=False)
    @patch("mvp_engine.utils.log.Console")
    def test_simple_info_invalid_env_level_falls_back_to_info(self, mock_console_cls):
        """Test simple_info fallback handles invalid LOG_LEVEL by warning and using info."""
        from mvp_engine.utils.log import simple_info

        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        simple_info("visible", level="info")

        # First print is warning for invalid env, second is the actual info log.
        assert mock_console.print.call_count == 2

    @patch.dict("os.environ", {"LOG_LEVEL": "error"}, clear=False)
    @patch("mvp_engine.utils.log.Console")
    def test_simple_info_respects_env_without_logger(self, mock_console_cls):
        """Test simple_info fallback console output is filtered by LOG_LEVEL."""
        from mvp_engine.utils.log import simple_info

        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        simple_info("hidden", level="info")
        simple_info("shown", level="error")

        assert mock_console.print.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
