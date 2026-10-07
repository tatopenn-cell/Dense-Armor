# -*- coding: utf-8 -*-
import numpy as np
import pytest

from dense_armor.core.compiler import DynamicAICodegen, CMD_MAP


def test_compile_pipeline_riconosce_tutti_i_comandi_testuali():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(list(CMD_MAP.keys()))
    assert ops.shape == (len(CMD_MAP), 4)
    for name, idx in CMD_MAP.items():
        row = ops[list(CMD_MAP.keys()).index(name)]
        assert row[0] == float(idx)


def test_compile_pipeline_comando_sconosciuto_usa_identity():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["comando_inesistente"])
    assert ops[0, 0] == float(CMD_MAP["identity"])


def test_compile_pipeline_forma_a_tupla_passa_i_parametri():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline([("scale", 2.0, 0.5)])
    assert ops[0].tolist() == [float(CMD_MAP["scale"]), 2.0, 0.5, 0.0]


def test_run_dynamic_pipeline_relu_azzera_i_negativi():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["relu"])
    out = codegen.run_dynamic_pipeline(np.array([-2.0, 3.0, -0.5]), ops)
    np.testing.assert_allclose(np.array(out), [0.0, 3.0, 0.0])


def test_run_dynamic_pipeline_scale_applica_trasformazione_affine():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline([("scale", 2.0, 1.0)])
    out = codegen.run_dynamic_pipeline(np.array([1.0, 2.0]), ops)
    np.testing.assert_allclose(np.array(out), [3.0, 5.0])


def test_run_dynamic_pipeline_clip_rispetta_i_bordi():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline([("clip", -1.0, 1.0)])
    out = codegen.run_dynamic_pipeline(np.array([-5.0, 0.5, 5.0]), ops)
    np.testing.assert_allclose(np.array(out), [-1.0, 0.5, 1.0])


def test_run_dynamic_pipeline_l2_normalize_produce_norma_unitaria():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["l2_normalize"])
    out = codegen.run_dynamic_pipeline(np.array([3.0, 4.0]), ops)
    assert np.linalg.norm(np.array(out)) == pytest.approx(1.0, abs=1e-6)


def test_run_dynamic_pipeline_dropout_avanza_la_prng_key():
    codegen = DynamicAICodegen(seed=0)
    ops = codegen.compile_pipeline([("dropout", 0.5, 0.0)])
    key_before = codegen.base_key
    codegen.run_dynamic_pipeline(np.ones(100), ops)
    assert not np.array_equal(np.array(key_before), np.array(codegen.base_key))


def test_run_pipeline_with_chunking_coincide_col_run_diretto():
    codegen = DynamicAICodegen(seed=7)
    ops = codegen.compile_pipeline(["relu", "tanh", "sigmoid", "identity", "clip"])
    data = np.array([-2.0, 0.5, 3.0])

    codegen_a = DynamicAICodegen(seed=7)
    direct = codegen_a.run_dynamic_pipeline(data, ops)

    codegen_b = DynamicAICodegen(seed=7)
    chunked = codegen_b.run_pipeline_with_chunking(data, ops, chunk_size=2)

    np.testing.assert_allclose(np.array(direct), np.array(chunked), atol=1e-6)


def test_compute_gradients_ha_la_shape_dell_input():
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["scale"])
    grads = codegen.compute_gradients(np.array([1.0, 2.0, 3.0]), ops)
    assert grads.shape == (3,)
    assert np.all(np.isfinite(grads))


def test_save_and_load_compiled_pipeline_roundtrip(tmp_path):
    codegen = DynamicAICodegen()
    ops = codegen.compile_pipeline(["relu", "tanh"])
    filename = str(tmp_path / "recipe.npy")

    codegen.save_compiled_pipeline(ops, filename)
    loaded = codegen.load_compiled_pipeline(filename)

    np.testing.assert_array_equal(loaded, ops)


def test_load_compiled_pipeline_file_assente_solleva_errore(tmp_path):
    codegen = DynamicAICodegen()
    with pytest.raises(FileNotFoundError):
        codegen.load_compiled_pipeline(str(tmp_path / "non_esiste.npy"))
