"""Project-specific exceptions.

Keeping the exception hierarchy small makes it possible for frontends and
backend plugins to distinguish user graph errors from compiler/runtime bugs.
"""


class MegaKernelError(Exception):
    """Base class for all public mega-infra errors."""


class ShapeError(MegaKernelError, ValueError):
    """A tensor shape or dtype constraint was not satisfied."""


class GraphValidationError(MegaKernelError, ValueError):
    """The semantic graph is malformed."""


class RegistryError(MegaKernelError, ValueError):
    """An op schema or kernel variant could not be registered/selected."""


class CompilationError(MegaKernelError, RuntimeError):
    """A valid graph could not be lowered into an executable program."""


class AbiError(MegaKernelError, ValueError):
    """A command buffer violates the runtime ABI."""


class RuntimeExecutionError(MegaKernelError, RuntimeError):
    """A compiled task failed in a runtime backend."""
