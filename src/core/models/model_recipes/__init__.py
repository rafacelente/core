from core.models.model_recipes.model_recipe import ModelRecipe

# Import concrete recipes to trigger registration of all built-in recipes
import core.models.model_recipes.model_recipes as _recipes  # noqa: F401

__all__ = ["ModelRecipe"]
