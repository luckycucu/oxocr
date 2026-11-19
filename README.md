<div align="center">
  <img src="docs/images/logo.svg" alt="Logo" width="400" />
  <p>
    <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg?logo=opensourceinitiative&logoColor=white" alt="License" /></a>
    <img src="https://img.shields.io/badge/Python-3.10+-FFD43B.svg?logo=python&logoColor=3776AB" alt="Python" />
    <img src="https://img.shields.io/badge/Runtime-ONNX-005CED.svg?logo=onnx&logoColor=white" alt="ONNX Runtime" />
    <img src="https://img.shields.io/badge/Models-45-orange.svg" alt="Models" />
  </p>
  <p>
    <a href="README_zh.md">简体中文</a> ·
    <b>English</b>
  </p>
</div>

**Lightweight Python OCR library powered by ONNX Runtime** — 45 built-in models covering the PaddleOCR series, DuGuang OCR series, Indra OCR series, pix2text_mfr (formula recognition), and more.

- **Lightweight dependencies**: mainly numpy, Pillow and onnxruntime — **no OpenCV required**.
- **Layout output**: 4 layout modes (raw / line / para / layout); output plain text directly, or the raw recognized text-box data.
- **Visualization**: draw boxes and recognized text back onto the image (`oxocr.visual` / `run --vis`); auto-detects system fonts, no bundled font.
- **CLI**: Docker-style command design — simple, intuitive and easy to pick up.

## Quick Start

### Installation

```bash
pip install oxocr
```

Requires Python ≥ 3.10. Install from source: `pip install -e .` (dev/test dependencies: [docs/testing.md](docs/testing.md)).

### Model Download

Model weights are not shipped with the package. Download them before first use (ModelScope preferred, HuggingFace as fallback):

```bash
oxocr download ppocrv6_small     # saved to ~/.oxocr/models by default
```

You can also download from Python: `oxocr.download_model("ppocrv6_small")`.

All model weights are hosted in the following repositories for manual browsing or downloading:

- ModelScope (preferred in China): [mscoder/oxocr](https://modelscope.cn/models/mscoder/oxocr)
- HuggingFace (fallback): [Luoaho/oxocr](https://huggingface.co/Luoaho/oxocr)

Use `--source modelscope|huggingface` to pick a download source. On restricted networks, download files manually from the repositories above and place them in the model directory (default `~/.oxocr/models`, customizable via `OXOCR_MODEL_DIR`) for offline use.

### Recognition

```python
import oxocr

# One-liner text extraction (default: ppocrv6_small + para mode)
text = oxocr.readtext("shot.png")

# Full result: infer once, render multiple layouts
result = oxocr.ocr("shot.png", mode="para")
print(result.text)
print(result.as_text("layout"))          # switch layout without re-inference
for item in result.items:                 # boxes are always in original-image pixel coordinates
    print(item.text, item.score, item.box.as_rect())

# Reuse the engine (batch scenarios share a process-level resident session)
engine = oxocr.OcrEngine(model="ppocrv6_small", postprocess={"mode": "para"})
results = engine.batch(["a.png", "b.png"])
```

Specify a local offline model directory:

```python
text = oxocr.readtext("shot.png", model_dir="test/models")
```

### Visualization

Draw detection boxes and recognized text back onto the image:

```python
from oxocr.visual import draw_items, draw_ocr, draw_boxes
from PIL import Image

result = oxocr.ocr("shot.png")

vis = draw_items("shot.png", result.items)       # side-by-side: boxes | text (BGR ndarray)
Image.fromarray(vis[:, :, ::-1]).save("vis.png")

draw_ocr("shot.png", [it.box for it in result.items],
         [it.text for it in result.items])        # boxes + text panel with scores
draw_boxes("shot.png", [it.box for it in result.items])  # red boxes only
```

Or straight from the CLI:

```bash
oxocr run shot.png --vis vis.png                  # side-by-side visualization
oxocr run shot.png --vis vis.png --vis-mode box   # boxes only
```

Fonts are auto-detected from the system (CJK-first candidates); pass `font_path=` / `--font` to use any ttf/ttc — recommended for scripts like Thai or Devanagari that the default font may not cover. See [docs/cli.md](docs/cli.md) for all modes.

## Model List

Sorted by open-source release date.

| Model                          | Released   | Size (MB) | License           | Languages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------ | ---------- | --------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `anaksor_ocr`                | 2026-11-04 | 90.4      | Apache-2.0        | Khmer, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `openthai2.0_ocr`            | 2026-11-01 | 4.0       | Apache-2.0        | Thai, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `indra_ocr_cjk`              | 2026-10-19 | 28.7      | RAIL-M (modified) | Simplified Chinese, Traditional Chinese, Japanese, Korean                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `indra_ocr_latin`            | 2026-10-19 | 20.7      | RAIL-M (modified) | Latin, English, French, German, Italian, Spanish, Portuguese, Dutch, Polish, Romanian, Czech, Swedish, Norwegian, Danish, Finnish, Hungarian, Turkish, Vietnamese, Indonesian, Malay, Azerbaijani, Afrikaans, Bosnian, Croatian, Welsh, Estonian, Irish, Icelandic, Kurdish, Lithuanian, Latvian, Maltese, Maori, Occitan, Slovak, Slovenian, Albanian, Swahili, Tagalog, Uzbek, Latin, Serbian (Latin), Catalan, Basque, Galician, Luxembourgish, Romansh, Quechua, Hausa, Fula, Akan, Yoruba, Cyrillic, Greek |
| `indra_ocr_math`             | 2026-10-19 | 18.8      | RAIL-M (modified) | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `indra_ocr_rtl`              | 2026-10-19 | 5.5       | RAIL-M (modified) | Arabic, Persian, Urdu, Pashto, Kurdish, Hebrew                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `indra_ocr_world`            | 2026-10-19 | 17.4      | RAIL-M (modified) | Hindi, Thai, Bengali, Tamil, Telugu, Burmese, Khmer, Lao, Sinhala, Kannada, Armenian, Amharic                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `ppocrv6_medium`             | 2026-06-11 | 132.9     | Apache-2.0        | Simplified Chinese, Traditional Chinese, English, Japanese, French, German, Italian, Spanish, Portuguese, Dutch, Polish, Romanian, Czech, Swedish, Norwegian, Danish, Finnish, Hungarian, Turkish, Vietnamese, Indonesian, Malay, Azerbaijani, Afrikaans, Bosnian, Croatian, Welsh, Estonian, Irish, Icelandic, Kurdish, Lithuanian, Latvian, Maltese, Maori, Occitan, Slovak, Slovenian, Albanian, Swahili, Tagalog, Uzbek, Latin, Serbian (Latin), Catalan, Basque, Galician, Luxembourgish, Romansh, Quechua |
| `ppocrv6_small`              | 2026-06-11 | 30.3      | Apache-2.0        | Simplified Chinese, Traditional Chinese, English, Japanese, French, German, Italian, Spanish, Portuguese, Dutch, Polish, Romanian, Czech, Swedish, Norwegian, Danish, Finnish, Hungarian, Turkish, Vietnamese, Indonesian, Malay, Azerbaijani, Afrikaans, Bosnian, Croatian, Welsh, Estonian, Irish, Icelandic, Kurdish, Lithuanian, Latvian, Maltese, Maori, Occitan, Slovak, Slovenian, Albanian, Swahili, Tagalog, Uzbek, Latin, Serbian (Latin), Catalan, Basque, Galician, Luxembourgish, Romansh, Quechua |
| `ppocrv6_tiny`               | 2026-06-11 | 6.6       | Apache-2.0        | Simplified Chinese, Traditional Chinese, English, French, German, Italian, Spanish, Portuguese, Dutch, Polish, Romanian, Czech, Swedish, Norwegian, Danish, Finnish, Hungarian, Turkish, Vietnamese, Indonesian, Malay, Azerbaijani, Afrikaans, Bosnian, Croatian, Welsh, Estonian, Irish, Icelandic, Kurdish, Lithuanian, Latvian, Maltese, Maori, Occitan, Slovak, Slovenian, Albanian, Swahili, Tagalog, Uzbek, Latin, Serbian (Latin), Catalan, Basque, Galician, Luxembourgish, Romansh, Quechua           |
| `ppocrv5_arabic_mobile`      | 2025-10-16 | 13.2      | Apache-2.0        | Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv5_cyrillic_mobile`    | 2025-10-16 | 13.3      | Apache-2.0        | Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhaz, Adyghe, Kabardian, Avar, Dargwa, Ingush, Chechen, Lak, Lezgian, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Mari, Mordvin, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English                                                                                                                                                                                               |
| `ppocrv5_devanagari_mobile`  | 2025-10-16 | 13.1      | Apache-2.0        | Hindi, Marathi, Nepali, Bihari, Maithili, Old English, Bhojpuri, Magahi, Sadri, Newari, Konkani, Sanskrit, Haryanvi, English                                                                                                                                                                                                                                                                                                                                                                                    |
| `ppocrv5_eslav_mobile`       | 2025-10-16 | 13.1      | Apache-2.0        | Russian, Belarusian, Ukrainian, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv5_korean_mobile`      | 2025-10-16 | 18.4      | Apache-2.0        | Korean, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `ppocrv5_latin_mobile`       | 2025-10-16 | 13.1      | Apache-2.0        | French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua                                                       |
| `ppocrv5_ta_mobile`          | 2025-10-16 | 13.1      | Apache-2.0        | Tamil, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `ppocrv5_te_mobile`          | 2025-10-16 | 13.1      | Apache-2.0        | Telugu, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `ppocrv5_el_mobile`          | 2025-08-21 | 13.0      | Apache-2.0        | Greek, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `ppocrv5_en_mobile`          | 2025-08-21 | 13.1      | Apache-2.0        | English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv5_th_mobile`          | 2025-08-21 | 13.1      | Apache-2.0        | Thai, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `pix2text_mfr_1.5`           | 2025-07-07 | 114.1     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ppocrv5_ch_mobile`          | 2025-05-20 | 21.4      | Apache-2.0        | Simplified Chinese, Traditional Chinese, English, Japanese                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ppocrv5_ch_server`          | 2025-05-20 | 165.7     | Apache-2.0        | Simplified Chinese, Traditional Chinese, English, Japanese                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `pix2text_mfr`               | 2024-02-12 | 112.2     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `pix2text_mfr_quantized`     | 2024-02-12 | 50.8      | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ppocrv4_arabic_mobile`      | 2023-08-07 | 12.4      | Apache-2.0        | Arabic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `ppocrv4_ch_doc_server`      | 2023-08-07 | 199.2     | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_ch_mobile`          | 2023-08-07 | 15.4      | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_ch_server`          | 2023-08-07 | 195.0     | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_devanagari_mobile`  | 2023-08-07 | 12.4      | Apache-2.0        | Devanagari                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ppocrv4_en_mobile`          | 2023-08-07 | 12.4      | Apache-2.0        | English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv4_japan_mobile`       | 2023-08-07 | 14.4      | Apache-2.0        | Japanese                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv4_ka_mobile`          | 2023-08-07 | 12.4      | Apache-2.0        | Kannada                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv4_korean_mobile`      | 2023-08-07 | 28.0      | Apache-2.0        | Korean                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `ppocrv4_ta_mobile`          | 2023-08-07 | 26.4      | Apache-2.0        | Tamil                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ppocrv4_te_mobile`          | 2023-08-07 | 26.4      | Apache-2.0        | Telugu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `duguang_ocr_small`          | 2023-05-29 | 12.7      | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `duguang_ocr_large`          | 2023-02-06 | 119.8     | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `duguang_ocr_max`            | 2022-08-19 | 151.4     | Apache-2.0        | Simplified Chinese, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `rapid_latex_ocr`            | 2022-05-25 | 170.7     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ppocrv3_chinese_cht_mobile` | 2022-05-09 | 13.5      | Apache-2.0        | Traditional Chinese                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `ppocrv3_cyrillic_mobile`    | 2022-05-09 | 11.4      | Apache-2.0        | Cyrillic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv3_en_mobile`          | 2022-05-09 | 11.4      | Apache-2.0        | English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv3_latin_mobile`       | 2022-05-09 | 11.4      | Apache-2.0        | Latin, English                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

- `RAIL-M (modified)` is short for AI PUBS OPEN RAIL-M (modified).
- Models whose language column is `latex` are whole-image formula recognizers (formula image → LaTeX, no detection model).
- Use `oxocr inspect <name>` to view model details.
- `ppocr` models come from RapidOCR.

```python
oxocr.readtext("shot.png", model="ppocrv5_ch_server")
oxocr.readtext("shot.png", model="ppocrv4_korean_mobile")

# Orientation classification: three weights available, input shape switches automatically with the weight
oxocr.readtext("shot.png", use_cls=True,
               config={"model_options": {"params": {"cls_model": "textline_ori_server"}}})

# Formula recognition: whole formula image → LaTeX; postprocess skipped by default, LaTeX output as-is
oxocr.readtext("formula.png", model="pix2text_mfr")  # 1.0 original
oxocr.readtext("formula.png", model="pix2text_mfr_1.5")
oxocr.readtext("formula.png", model="pix2text_mfr_quantized")  # int8 encoder, faster
oxocr.readtext("formula.png", model="rapid_latex_ocr")  # LaTeX-OCR (pix2tex), built-in preprocessing, <1s per formula on CPU

# DuGuang OCR: three tiers for Chinese/English; max uses SegLink detection, small/large use DB detection
oxocr.readtext("shot.png", model="duguang_ocr_large")
oxocr.readtext("shot.png", model="duguang_ocr_max")

# Khmer OCR: Khmer + English/Latin; long lines auto-split and stitched, vertical/180° optional
oxocr.readtext("shot.png", model="anaksor_ocr")

# Thai OCR: Thai + English; full-page auto-rotation/deskew, column-aware reading order, ~0.4s/page on CPU
oxocr.readtext("shot.png", model="openthai2.0_ocr")

# Indra multilingual: pick the dedicated model by dominant language (each recognizer also reads Latin)
oxocr.readtext("shot.png", model="indra_ocr_cjk")     # CJK (Korean jamo auto-composed into syllables)
oxocr.readtext("shot.png", model="indra_ocr_rtl")     # Arabic/Hebrew (logical-order output)
oxocr.readtext("formula.png", model="indra_ocr_math")  # whole-image formula → LaTeX (no detection, same as MFR)
```

## CLI

Docker-style commands (`run` / `models` / `version`), see [docs/cli.md](docs/cli.md):

```bash
oxocr run shot.png -m ppocrv6_tiny -M layout  # recognize an image, output multi-column layout text
oxocr run shot.png --vis vis.png          # recognize and save a visualization image
oxocr models ls                 		# list models
oxocr models inspect ppocrv6_small		# inspect model details
oxocr -h --lang zh                 		# Chinese help (English by default)
```

## License

The code is open-sourced under the MIT License; model weights are not covered by this license and follow their respective license terms.
