"""Relations between objects, as pure functions."""

from __future__ import annotations

from .objects import ArcObject


def adjacent(a: ArcObject, b: ArcObject) -> bool:
    """Two objects are adjacent if any two of their cells are 4-neighbours.

    Args:
        a: the first object.
        b: the second object.

    Returns:
        True if the objects share a 4-cell boundary.
    """
    for r, c in a.cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in b.cells:
                return True
    return False


def inside(a: ArcObject, b: ArcObject) -> bool:
    """``a`` is inside ``b``'s bbox (strictly).

    Args:
        a: the inner object.
        b: the outer object.

    Returns:
        True if the cells of ``a`` are a subset of the cells of ``b``.
    """
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return br0 <= ar0 and bc0 <= ac0 and ar1 <= br1 and ac1 <= bc1 and a is not b


def same_shape(a: ArcObject, b: ArcObject) -> bool:
    """Same area and same bbox shape.

    Args:
        a: the first object.
        b: the second object.

    Returns:
        True if the two objects have the same shape.
    """
    return a.area == b.area and (
        (a.bbox[2] - a.bbox[0], a.bbox[3] - a.bbox[1])
        == (b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1])
    )


def same_color(a: ArcObject, b: ArcObject) -> bool:
    """Check whether two objects share the same color.

    Args:
        a: the first object.
        b: the second object.

    Returns:
        True if the two objects have the same color.
    """
    return a.color is not None and a.color == b.color


def distance(a: ArcObject, b: ArcObject) -> float:
    """Smallest Chebyshev distance between any two cells.

    Args:
        a: the first object.
        b: the second object.

    Returns:
        the Chebyshev distance between the two bounding boxes.
    """
    best = float("inf")
    for r1, c1 in a.cells:
        for r2, c2 in b.cells:
            d = max(abs(r1 - r2), abs(c1 - c2))
            if d < best:
                best = d
    return best
