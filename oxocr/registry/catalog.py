"""内置 ModelSpec 清单聚合（纯数据，只有字符串与数字）。"""

from __future__ import annotations

from oxocr.models.anaksor_ocr.specs import ANAKSOR_SPECS as _ANAKSOR_SPECS
from oxocr.models.base import ModelSpec
from oxocr.models.duguang_ocr.specs import DUGUANG_SPECS as _DUGUANG_SPECS
from oxocr.models.indra_ocr.specs import INDRA_SPECS as _INDRA_SPECS
from oxocr.models.openthai_ocr.specs import OPENTHAI_SPECS as _OPENTHAI_SPECS
from oxocr.models.rapid_latex_ocr.specs import LATEX_OCR_SPECS as _LATEX_OCR_SPECS
from oxocr.models.pix2text.specs import MFR_SPECS as _MFR_SPECS
from oxocr.models.rapid_ocr.specs import SPECS as _PPOCR_SPECS

__all__ = ["BUILTIN_SPECS"]

BUILTIN_SPECS: dict[str, ModelSpec] = {
    **_PPOCR_SPECS,
    **_MFR_SPECS,
    **_DUGUANG_SPECS,
    **_ANAKSOR_SPECS,
    **_OPENTHAI_SPECS,
    **_INDRA_SPECS,
    **_LATEX_OCR_SPECS,
}
