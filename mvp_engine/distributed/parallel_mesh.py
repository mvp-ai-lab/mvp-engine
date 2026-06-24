"""Semantic accessors for torch distributed device meshes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from mvp_engine.utils.log import simple_info

DEFAULT_ROLE_DIMS: dict[str, tuple[str, ...]] = {
    "dp": ("replicate", "shard"),
    "ddp": ("replicate",),
    "fsdp": ("replicate", "shard"),
    "tp": ("tensor",),
    "sp": ("tensor",),
    "cp": ("context",),
}


@dataclass(frozen=True, slots=True)
class MeshRole:
    """Resolved process role inside a ``ParallelMesh``."""

    name: str
    dim_names: tuple[str, ...]
    mesh: DeviceMesh | None
    group: dist.ProcessGroup | None
    world_size: int
    rank: int
    active: bool


class ParallelMesh:
    """Semantic facade over a torch ``DeviceMesh``."""

    device_mesh: DeviceMesh
    dim_names: tuple[str, ...]
    global_rank: int
    global_world_size: int

    @classmethod
    def initialize(
        cls,
        device_type: str,
        mesh_cfg: Mapping[str, int],
        *,
        role_dims: Mapping[str, Sequence[str]] | None = None,
    ) -> "ParallelMesh":
        """Initialize a torch ``DeviceMesh`` and wrap it in ``ParallelMesh``."""
        mesh_cfg = dict(mesh_cfg)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        inferred_dim_names = [dim_name for dim_name, dim_size in mesh_cfg.items() if dim_size == -1]
        if len(inferred_dim_names) > 1:
            raise ValueError("Only one device mesh dimension can be inferred.")

        known_sizes = [int(dim_size) for dim_size in mesh_cfg.values() if dim_size != -1]
        known_product = math.prod(known_sizes) if known_sizes else 1

        if inferred_dim_names:
            inferred_dim_name = inferred_dim_names[0]
            if world_size % known_product != 0:
                raise ValueError(
                    f"Cannot infer mesh dimension {inferred_dim_name!r}: "
                    f"world_size={world_size} is not divisible by known mesh size {known_product}."
                )
            mesh_cfg[inferred_dim_name] = world_size // known_product

        mesh_shape = tuple(int(dim_size) for dim_size in mesh_cfg.values())
        mesh_dim_names = tuple(mesh_cfg.keys())
        mesh_world_size = math.prod(mesh_shape) if mesh_shape else 1
        if mesh_world_size != world_size:
            raise ValueError(
                f"Device mesh shape {mesh_shape} has size {mesh_world_size}, "
                f"but distributed world size is {world_size}."
            )

        simple_info(
            f"Device Mesh initializing: {' / '.join(mesh_dim_names)} ({' / '.join(str(x) for x in mesh_shape)})..."
        )

        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
        return cls(device_mesh, role_dims=role_dims)

    def __init__(
        self,
        device_mesh: DeviceMesh,
        *,
        role_dims: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Wrap an existing torch ``DeviceMesh`` with semantic mesh roles."""
        self.device_mesh = device_mesh
        self.dim_names = tuple(getattr(device_mesh, "mesh_dim_names", ()) or ())
        self.global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.global_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        self._role_dims = dict(DEFAULT_ROLE_DIMS)
        if role_dims is not None:
            self._role_dims.update({name: tuple(dim_names) for name, dim_names in role_dims.items()})

        self._roles: dict[str, MeshRole] = {}
        self._submeshes: dict[tuple[str, ...], DeviceMesh | None] = {}
        self._groups: dict[tuple[str, ...], dist.ProcessGroup | None] = {}

    @property
    def dp(self) -> MeshRole:
        """Data-parallel role."""
        return self.role("dp")

    @property
    def ddp(self) -> MeshRole:
        """DistributedDataParallel role."""
        return self.role("ddp")

    @property
    def fsdp(self) -> MeshRole:
        """Fully-sharded data-parallel role."""
        return self.role("fsdp")

    @property
    def tp(self) -> MeshRole:
        """Tensor-parallel role."""
        return self.role("tp")

    @property
    def sp(self) -> MeshRole:
        """Sequence-parallel role."""
        return self.role("sp")

    @property
    def cp(self) -> MeshRole:
        """Context-parallel role."""
        return self.role("cp")

    def role(self, name: str) -> MeshRole:
        """Return the resolved mesh role for ``name``."""
        if name in self._roles:
            return self._roles[name]

        dim_names = self._role_dim_names(name)
        mesh = self._submesh(dim_names)
        group = self._group(dim_names)
        world_size = self._world_size(dim_names)
        rank = dist.get_rank(group) if group is not None else 0
        role = MeshRole(
            name=name,
            dim_names=dim_names,
            mesh=mesh,
            group=group,
            world_size=world_size,
            rank=rank,
            active=world_size > 1,
        )
        self._roles[name] = role
        return role

    def submesh(self, name: str) -> DeviceMesh | None:
        """Return the submesh for a role or mesh dimension name."""
        return self.role(name).mesh

    def group(self, name: str) -> dist.ProcessGroup | None:
        """Return the process group for a role or mesh dimension name."""
        return self.role(name).group

    def world_size(self, name: str) -> int:
        """Return the world size for a role or mesh dimension name."""
        return self.role(name).world_size

    def rank(self, name: str) -> int:
        """Return the rank within a role or mesh dimension group."""
        return self.role(name).rank

    def active(self, name: str) -> bool:
        """Return whether a role or mesh dimension has more than one rank."""
        return self.role(name).active

    def has_dim(self, name: str) -> bool:
        """Return whether the wrapped mesh exposes ``name``."""
        return name in self.dim_names

    def dim_size(self, name: str) -> int:
        """Return the size of a concrete mesh dimension."""
        self._require_dim(name)
        return int(self.device_mesh[name].size())

    def dim_rank(self, name: str) -> int:
        """Return the current rank within a concrete mesh dimension."""
        self._require_dim(name)
        dim_world_size = self.dim_size(name)
        if dim_world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank(self.device_mesh[name].get_group())

    def summary(self) -> dict[str, Any]:
        """Return a compact serializable summary of mesh roles."""
        roles = {name: self.role(name) for name in DEFAULT_ROLE_DIMS}
        return {
            "dim_names": self.dim_names,
            "shape": tuple(getattr(self.device_mesh, "shape", ())),
            "roles": {
                name: {
                    "dim_names": role.dim_names,
                    "world_size": role.world_size,
                    "rank": role.rank,
                    "active": role.active,
                }
                for name, role in roles.items()
            },
        }

    def _role_dim_names(self, name: str) -> tuple[str, ...]:
        if name in self._role_dims:
            return tuple(dim_name for dim_name in self._role_dims[name] if self.has_dim(dim_name))
        if self.has_dim(name):
            return (name,)
        raise KeyError(f"Unknown mesh role or dimension: {name}")

    def _require_dim(self, name: str) -> None:
        if not self.has_dim(name):
            raise KeyError(f"Unknown mesh dimension: {name}")

    def _submesh(self, dim_names: tuple[str, ...]) -> DeviceMesh | None:
        if dim_names in self._submeshes:
            return self._submeshes[dim_names]
        if not dim_names:
            submesh = None
        elif len(dim_names) == 1:
            submesh = self.device_mesh[dim_names[0]]
        else:
            submesh = self.device_mesh[dim_names]
        self._submeshes[dim_names] = submesh
        return submesh

    def _group(self, dim_names: tuple[str, ...]) -> dist.ProcessGroup | None:
        if dim_names in self._groups:
            return self._groups[dim_names]
        world_size = self._world_size(dim_names)
        if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            self._groups[dim_names] = None
            return None
        mesh = self._submesh(dim_names)
        if mesh is None:
            self._groups[dim_names] = None
            return None
        if len(dim_names) == 1:
            group = mesh.get_group()
        else:
            group = mesh._flatten("_".join(dim_names)).get_group()
        self._groups[dim_names] = group
        return group

    def _world_size(self, dim_names: tuple[str, ...]) -> int:
        if not dim_names:
            return 1
        return math.prod(int(self.device_mesh[dim_name].size()) for dim_name in dim_names)
