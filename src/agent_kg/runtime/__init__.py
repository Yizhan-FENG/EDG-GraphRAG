"""Local shared-backbone role runtime with explicit sandbox boundaries."""

from .role_lora import RoleSandboxError, SharedQwenRoleRuntime

__all__ = ["RoleSandboxError", "SharedQwenRoleRuntime"]
