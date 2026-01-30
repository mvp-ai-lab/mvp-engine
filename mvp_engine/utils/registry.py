from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """Simple registry mapping string keys to classes or callables.

    Typical usage:

        registry = Registry()

        @registry.register()
        class MyEngine: ...

        @registry.register('custom_name')
        def factory(...): ...

    The decorator supports either the ``@registry.register()`` form or the
    direct decoration ``@registry.register`` (no parentheses).
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self.mapper: Dict[str, Any] = {}

    def register(self, key: Optional[str] = None) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object.

        This function supports two usages:
        - ``@registry.register()`` (recommended)
        - ``@registry.register`` (convenience)

        Args:
            key: Optional name to register under. If omitted, the object's
                ``__name__`` is used.

        Returns:
            A decorator that registers the provided class/function and
            returns it unchanged.
        """

        # Support direct decoration: @registry.register
        if callable(key):  # type: ignore[arg-type]
            obj = key  # type: ignore[assignment]
            reg_key = obj.__name__
            self.mapper[reg_key] = obj
            return obj  # type: ignore[return-value]

        def decorator(class_or_func: T) -> T:
            reg_key = class_or_func.__name__ if key is None else key
            self.mapper[reg_key] = class_or_func
            return class_or_func

        return decorator

    def get(self, key: str) -> Any:
        """Retrieve a registered object by its name.

        Raises a ``KeyError`` when the key is not present.
        """
        if key not in self.mapper:
            raise KeyError(f"No object named '{key}' found!")
        return self.mapper[key]

    def __getitem__(self, key: str) -> Any:
        """Alias for ``get`` to allow ``registry[name]`` access."""
        return self.get(key)
