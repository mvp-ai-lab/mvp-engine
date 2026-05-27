"""Assertions for the Basic VLM model-compile skill."""

import ast
import inspect
import textwrap

from omegaconf import OmegaConf

from mvp_engine.kit.mllm import MLLMModelKit


def test_config_structure(config: OmegaConf):
    """Test that the recipe config exposes supported torch.compile options."""
    assert config.model.compile is not None, "Config must have a 'model.compile' section."
    assert config.model.compile.enabled in [True, False], "model.compile.enabled must be a bool."
    assert config.model.compile.backend in ["inductor", "aot_eager", "eager"], (
        "Compile backend must be 'inductor', 'aot_eager', or 'eager'."
    )
    assert config.model.compile.mode in [
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ], "Compile mode must be a supported torch.compile mode."


def test_engine_structure(engine_class: object):
    """Test that the engine prepares model compile through the model kit."""
    prepare_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    assert "self.config.model.compile.enabled" in prepare_source
    assert "self.config.model.compile.backend" in prepare_source
    assert "self.config.model.compile.mode" in prepare_source
    assert "self.model_kit.apply_model_compile" in prepare_source

    source = textwrap.dedent(inspect.getsource(MLLMModelKit.apply_model_compile))
    tree = ast.parse(source)

    compile_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
        ):
            compile_call = node
            break

    assert compile_call is not None, "MLLMModelKit.apply_model_compile must call model.compile(...)."

    keyword_values = {keyword.arg: ast.unparse(keyword.value) for keyword in compile_call.keywords}
    assert keyword_values.get("backend") == "backend", "model.compile must pass the backend argument."
    assert keyword_values.get("mode") == "mode", "model.compile must pass the mode argument."
