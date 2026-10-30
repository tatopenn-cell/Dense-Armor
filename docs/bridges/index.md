# Bridges: using Dense-Armor with other libraries

Dense-Armor estimates one sample at a time; scikit-learn estimates in a `fit` over the whole table. A bridge lets each side use the other without rewriting an algorithm.

## What a bridge is

A bridge does not replace an algorithm. It makes a Dense-Armor object usable where the other library expects its own interface, and the other way round, with a test that the numbers agree.

## Available bridges

| Library | Page | Extra |
|---|---|---|
| scikit-learn | [sklearn](sklearn.md) | `pip install dense-armor[sklearn]` |
| PyTorch | [torch](torch.md) | `pip install dense-armor[torch]` |
| river | `dense_armor.bridges.river` | `pip install dense-armor[river]` |

The imports of a bridge are optional: `import dense_armor` works without the other library installed, and the bridge module raises a `ModuleNotFoundError` that names the extra when the other library is missing.
