from typing import Union, Any
import h5py  # type: ignore
from .state import MPS


def print_dict(d: dict, prefix="") -> None:
    for key, item in d.items():
        if isinstance(item, dict):
            print(prefix + key + ":")
            print_dict(item, prefix + " ")
        else:
            print(prefix + key + ": " + str(item))


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
    g = parent.create_group(name)
    g.attrs["type"] = "MPS"
    g.attrs["version"] = 1
    g.create_dataset("length", data=len(M))
    for i, A in enumerate(M):
        g.create_dataset(f"MPS[{i}]", shape=A.shape, data=A)


def read_mps(parent: Union[h5py.File, h5py.Group], name: str) -> MPS:
    g = parent[name]
    if g.attrs["type"] == "MPS" and g.attrs["version"] == 1:
        N = g["length"][()]
        # rlim = g["rlim"][()]
        # llim = g["llim"][()]
        v = [g[f"MPS[{i}]"][()] for i in range(N)]
    else:
        raise Exception(f"Unable to read MPS from HDF5 group {parent}")
    return MPS(v)
