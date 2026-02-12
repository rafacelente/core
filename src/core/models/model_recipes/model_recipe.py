from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.model_config import CoreConfig
    from core.models.model import CoreModel


class ModelRecipe(ABC):
    """
    Base class for model recipes.

    Subclass and register to define new model architectures.

    Example:
        @ModelRecipe.register("my-model")
        class MyModelRecipe(ModelRecipe):
            n_layers = 4
            d_model = 256
            n_heads = 4

            def build_config(self, vocab_size, **kwargs):
                return CoreConfig(...)
    """

    _registry: Dict[str, Type["ModelRecipe"]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type["ModelRecipe"]], Type["ModelRecipe"]]:
        """
        Decorator to register a model recipe under the given name.

        Args:
            name: The name to register the recipe under (e.g. "gpt-tiny").

        Returns:
            A decorator that registers the class and returns it unchanged.
        """
        def decorator(recipe_cls: Type["ModelRecipe"]) -> Type["ModelRecipe"]:
            if not isinstance(recipe_cls, type):
                raise ValueError("ModelRecipe.register must be called with a class")
            if not issubclass(recipe_cls, ModelRecipe):
                raise ValueError(f"{recipe_cls.__name__} must extend ModelRecipe")
            if name in cls._registry:
                raise ValueError(f"Model recipe '{name}' is already registered")
            cls._registry[name] = recipe_cls
            return recipe_cls
        return decorator

    @classmethod
    def get_available_recipes(cls) -> List[str]:
        """
        Returns a list of the available model recipe names.
        """
        return list(cls._registry.keys())

    @classmethod
    def get_recipe(cls, name: str) -> "ModelRecipe":
        """
        Get an instance of a registered model recipe by name.

        Args:
            name: The registered recipe name.

        Returns:
            An instance of the requested ModelRecipe.

        Raises:
            ValueError: If the recipe name is not found in the registry.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Model recipe '{name}' not found. "
                f"Available recipes: {cls.get_available_recipes()}"
            )
        return cls._registry[name]()

    @abstractmethod
    def build_config(
        self,
        vocab_size: int,
        max_sequence_length: int = 2048,
        dropout: float = 0.0,
        transformer_type: str = "base",
        use_post_sdpa_gate: bool = False,
        gate_activation_type: str = "sigmoid",
        pad_token_id: int = 50256,
    ) -> "CoreConfig":
        """
        Build a CoreConfig for this model recipe.

        Each recipe defines how to assemble a CoreConfig from its
        architectural parameters and the provided runtime parameters.

        Args:
            vocab_size: The vocabulary size.
            max_sequence_length: Maximum sequence length.
            dropout: Dropout rate.
            transformer_type: Type of transformer ("base" or "normalized").
            use_post_sdpa_gate: Whether to use post-SDPA gating.
            gate_activation_type: Activation type for the gate.
            pad_token_id: Padding token ID.

        Returns:
            A CoreConfig instance.
        """
        ...

    def build(
        self,
        vocab_size: int,
        **kwargs,
    ) -> "CoreModel":
        """
        Build a model from this recipe.

        Convenience method that builds the config and then builds the model.

        Args:
            vocab_size: The vocabulary size.
            **kwargs: Additional arguments passed to build_config.

        Returns:
            A CoreModel instance.
        """
        config = self.build_config(vocab_size=vocab_size, **kwargs)
        return config.build()
