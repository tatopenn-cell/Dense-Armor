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
    """The bounding box of ``a`` is inside the bounding box of ``b``.

    The test is on the boxes, not on the cells: ``a`` may sit in a hole
    of ``b``. Equal boxes count as inside; an object is not inside itself.

    Args:
        a: the inner object.
        b: the outer object.

    Returns:
        True if the bbox of ``a`` is contained in the bbox of ``b`` and
        ``a is not b``.
    """
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return br0 <= ar0 and bc0 <= ac0 and ar1 <= br1 and ac1 <= bc1 and a is not b


def same_shape(a: ArcObject, b: ArcObject) -> bool:
    """Same normalised cell mask: the same shape wherever the object is.

    Args:
        a: the first object.
        b: the second object.

    Returns:
        True if the cells of the two objects coincide once each is moved
        to the top-left corner.
    """

    def mask(o: ArcObject) -> frozenset[tuple[int, int]]:
        r0, c0 = o.bbox[0], o.bbox[1]
        return frozenset((r - r0, c - c0) for r, c in o.cells)

    return mask(a) == mask(b)


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
        the smallest Chebyshev distance between a cell of ``a`` and a cell of ``b``.
    """
    best = float("inf")
    for r1, c1 in a.cells:
        for r2, c2 in b.cells:
            d = max(abs(r1 - r2), abs(c1 - c2))
            if d < best:
                best = d
    return best
