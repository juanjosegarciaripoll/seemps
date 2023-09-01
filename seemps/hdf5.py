"""Read and write MPS from and to HDF5 files.

HDF5 is a portable format to store arbitrary numerical data, collected
in a hierarchical structure of nested subgroups, whose leaves are actual
datasets (e.g., arrays, strings, numbers, etc).

This module provides functions to read and write :class:`~seemps.MPS` objects
from and to HDF5 files. Each quantum state is stored in a separate group,
with a name supplied by the user. If `g` is the HDF5 group, the format
consists of the following datasets and attributes:

- `g["length"]`, integer `N` with the size of the MPS.
- `g.attrs["type"]` is `"MPS"`
- `g.attrs["version"]` is 1 for this library.
- `g["MPS[0]"]`, `g["MPS[1]"]` and subsequent fields are datasets for each
  of the tensors.

SeeMPS uses the Python library `h5py <https://www.h5py.org/>`_ to read and write
states in these structured files. For instance, the following code creates a
file with a single MPS stored in the group `"state"`. Afterwards, it reads the
same state from the file.

.. highlight:: python
.. code-block:: python

    import h5py
    import seemps

    # Create a single file, overwriting any existing one
    with h5py.File("data.hdf5", "w") as file:
        seemps.hdf5.write_mps(file, "state" my_mps

    # Read the MPS from the same file, reopening it as read-only
    with h5py.File("data.hdf5", "r") as file
        seemps.hdf5.read_mps(file, "state")

"""
from __future__ import annotations
from typing import Union, Any
import h5py  # type: ignore
from .state import MPS


def _read_hdf5_item_as_path(
    item: Union[h5py.File, h5py.Group, h5py.Dataset], output: list[tuple[str, Any]]
) -> list[tuple[str, Any]]:
    for subitem in item.values():
        if isinstance(subitem, h5py.Dataset):
            output.append((subitem.name, subitem[()]))
        else:
            _read_hdf5_item_as_path(subitem, output)
    return output


def read_full_hdf5_as_paths(filename: str) -> dict[str, Any]:
    with h5py.File(filename, "r") as file:
        return {key: value for key, value in _read_hdf5_item_as_path(file, [])}


def _read_hdf5_item(item: Union[h5py.File, h5py.Group, h5py.Dataset]) -> dict:
    if isinstance(item, h5py.Dataset):
        return item[()]
    output: dict = {key: _read_hdf5_item(subitem) for key, subitem in item.items()}
    output["_attrs"] = list(item.attrs.items())
    return output


def read_full_hdf5(filename: str) -> dict:
    with h5py.File(filename, "r") as file:
        return _read_hdf5_item(file)


def write_mps(parent: Union[h5py.File, h5py.Group], name: str, M: MPS) -> None:
    """Write an MPS to an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPS
        The quantum state to save.

    Examples
    --------
    >>> import h5py
    >>> import seemps.state, seemps.hdf5
    >>> mps = seemps.state.random_mps(2, 10)
    >>> file = h5py.File("data.hdf5", "w")
    >>> seemps.hdf5.write_mps(file, "state", mps)
    >>> file.close()
    """
    g = parent.create_group(name)
    g.attrs["type"] = "MPS"
    g.attrs["version"] = 1
    g.create_dataset("length", data=len(M))
    for i, A in enumerate(M):
        g.create_dataset(f"MPS[{i}]", shape=A.shape, data=A)


def read_mps(parent: Union[h5py.File, h5py.Group], name: str) -> MPS:
    """Reand an MPS from an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPS
        The quantum state to save.

    Examples
    --------
    >>> import h5py
    >>> import seemps.state, seemps.hdf5
    >>> mps = seemps.state.random_mps(2, 10)
    >>> file = h5py.File("data.hdf5", "r")
    >>> mps = seemps.hdf5.read_mps(file, "state")
    >>> mps.physical_dimensions()
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> file.close()
    """
    g = parent[name]
    if g.attrs["type"] == "MPS" and g.attrs["version"] == 1:
        N = g["length"][()]
        # rlim = g["rlim"][()]
        # llim = g["llim"][()]
        v = [g[f"MPS[{i}]"][()] for i in range(N)]
    else:
        raise Exception(f"Unable to read MPS from HDF5 group {parent}")
    return MPS(v)
