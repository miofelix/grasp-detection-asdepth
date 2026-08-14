"""旧 ``defm_stackconv_depth`` 兼容快照；新模型使用 ``asdepth.models`` catalog。"""

from .variants.defm_stackconv import DeFMStackConvRGBDDepth

__all__ = ["DeFMStackConvRGBDDepth"]
