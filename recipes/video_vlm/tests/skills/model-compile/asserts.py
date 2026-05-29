import ast
import inspect
import textwrap

from omegaconf import OmegaConf


def test_config_structure(config: OmegaConf):
    assert config.model.compile is not None, "Config must have a 'model.compile' section."
    assert config.model.compile_backend in ["inductor", "torchdynamo"], (
        "Compile backend must be either 'inductor' or 'torchdynamo'."
    )
    assert config.model.compile_mode in ["default", "reduce-overhead"], (
        "Compile mode must be either 'default' or 'reduce-overhead'."
    )


def test_engine_structure(engine_class: object):
    """Test that the engine's prepare_model method calls model.compile with the correct arguments."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
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

    assert compile_call is not None, "prepare_model must call model.compile(...)."

    keyword_values = {keyword.arg: ast.unparse(keyword.value) for keyword in compile_call.keywords}
    assert keyword_values.get("backend") == "self.config.model.compile_backend", (
        "model.compile must pass backend=self.config.model.compile_backend."
    )
    assert keyword_values.get("mode") == "self.config.model.compile_mode", (
        "model.compile must pass mode=self.config.model.compile_mode."
    )
