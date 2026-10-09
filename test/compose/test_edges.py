"""Edge paths of the compose transformers."""

from dense_armor.roles import Signal
from dense_armor.utility.compose import FuncTransformer, Select


def test_select_nothing_and_func_whole_and_missing_keys():
    sig = Signal(values=[1.0, 2.0], names=["a", "b"], units=["m", "s"], t=0.5)
    empty = Select(keys=("zz",)).transform_one(sig)
    assert empty.names == []
    whole = FuncTransformer(lambda d: {"s": d["a"] + d["b"]}, whole=True).transform_one(
        sig
    )
    assert whole.names == ["s"] and float(whole["s"]) == 3.0 and whole.t == 0.5
    f = FuncTransformer(lambda v: v * 10.0, keys=("a", "zz"))
    assert float(f.transform_one(sig)["a"]) == 10.0
    assert f.transform_one({"a": 1.0}) == {"a": 10.0}
