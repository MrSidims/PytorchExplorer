class IRGenerationError(Exception):
    """Raised when IR generation fails."""


class PytorchExecutionError(IRGenerationError):
    """Raised when user Pytorch code execution fails."""


class TritonCompilationError(IRGenerationError):
    """Raised when compiling Triton IR fails."""


class TritonExecutionError(IRGenerationError):
    """Raised when user Triton code execution fails."""


class CompilerPipelineError(IRGenerationError):
    """Raised when running compiler tools (opt, translate, etc) fails."""

    def __init__(self, msg):
        super().__init__(msg)
