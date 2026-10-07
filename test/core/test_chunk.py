# -*- coding: utf-8 -*-
import jax.numpy as jnp
import numpy as np

from dense_armor.core.chunk import ImageChunker
from dense_armor.core.compiler import DynamicAICodegen


def test_split_array_divide_in_chunk_della_dimensione_giusta():
    chunker = ImageChunker(chunk_size=3)
    data = np.arange(10)
    chunks = chunker.split_array(data)
    assert [len(c) for c in chunks] == [3, 3, 3, 1]


def test_merge_chunks_ricostruisce_array_1d():
    chunker = ImageChunker(chunk_size=4)
    data = np.arange(10, dtype=np.float32)
    chunks = chunker.split_array(data)
    merged = chunker.merge_chunks(chunks)
    np.testing.assert_array_equal(merged, data)


def test_merge_chunks_ricostruisce_array_2d():
    chunker = ImageChunker(chunk_size=2)
    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    chunks = chunker.split_array(data)
    merged = chunker.merge_chunks(chunks)
    np.testing.assert_array_equal(merged, data)


def test_merge_chunks_lista_vuota():
    chunker = ImageChunker()
    merged = chunker.merge_chunks([])
    assert merged.shape == (0,)


def test_execute_pipeline_chunked_usa_il_chunking_del_codegen():
    codegen = DynamicAICodegen(seed=1)
    ops = codegen.compile_pipeline(["relu", "relu"])
    chunker = ImageChunker(chunk_size=1)
    input_vector = np.array([-1.0, 2.0, -3.0])

    out = chunker.execute_pipeline_chunked(codegen, input_vector, list(ops))
    direct = codegen.run_dynamic_pipeline(input_vector, ops)
    np.testing.assert_allclose(np.array(out), np.array(direct), atol=1e-6)


def test_patch_and_scan_parameters_sostituisce_i_marcatori():
    template_ops = jnp.array([1.0, -1.0, 3.0, -1.0])
    dynamic_parameters = jnp.array([9.0, 8.0])

    patched = ImageChunker.patch_and_scan_parameters(template_ops, dynamic_parameters)
    patched = np.array(patched)

    np.testing.assert_allclose(patched[0], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(patched[1], [-1.0, -1.0, -1.0, 9.0])
    np.testing.assert_allclose(patched[2], [3.0, 3.0, 3.0, 3.0])
    np.testing.assert_allclose(patched[3], [-1.0, -1.0, -1.0, 8.0])
