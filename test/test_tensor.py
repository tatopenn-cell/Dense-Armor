# -*- coding: utf-8 -*-
import numpy as np
import pytest

from dense_armor.core.tensor import TensorVault


@pytest.mark.parametrize("name", ["invert", "identity", "edge_detector", "blend"])
def test_get_static_transform_nomi_validi(name):
    vault = TensorVault()
    result = vault.get_static_transform(name)
    assert np.array(result).size > 0


def test_get_static_transform_case_insensitive():
    vault = TensorVault()
    np.testing.assert_array_equal(
        np.array(vault.get_static_transform("IDENTITY")),
        np.array(vault.get_static_transform("identity")),
    )


def test_get_static_transform_nome_invalido_solleva_keyerror():
    vault = TensorVault()
    with pytest.raises(KeyError):
        vault.get_static_transform("non_esiste")


@pytest.mark.parametrize("name", ["scale_project", "amplify", "bias_shift"])
def test_get_parametric_transform_nomi_validi(name):
    vault = TensorVault()
    result = vault.get_parametric_transform(name, 0.5)
    assert np.array(result).size > 0


def test_get_parametric_transform_nome_invalido_solleva_keyerror():
    vault = TensorVault()
    with pytest.raises(KeyError):
        vault.get_parametric_transform("non_esiste", 1.0)


def test_get_parametric_transform_amplify_scala_la_diagonale():
    vault = TensorVault()
    result = np.array(vault.get_parametric_transform("amplify", 3.0))
    np.testing.assert_allclose(result, [[3.0, 0.0], [0.0, 3.0]])


def test_get_backend_info_ritorna_una_stringa_descrittiva():
    vault = TensorVault()
    info = vault.get_backend_info()
    assert isinstance(info, str)
    assert "bit" in info
