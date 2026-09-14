#!/usr/bin/env python3
"""Собрать PDF научной работы из report.md.

    python scripts/build_report_pdf.py

Скрипт содержит минимальный конвертер Markdown того подмножества, которое
использует `report.md` (заголовки, абзацы, таблицы, списки, блоки формул,
цитаты, выделение). Внешние библиотеки конвертации намеренно НЕ используются:
проект заявляет минимальный набор зависимостей, и добавлять ещё одну ради
сборки одного документа нецелесообразно.

Для получения PDF используется установленный в системе браузер на движке
Chromium в режиме печати. Если браузер не найден, скрипт сохраняет HTML —
его можно открыть и напечатать в PDF вручную (Cmd+P -> Сохранить как PDF).
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_root, setup_logging

#: Возможные имена исполняемого файла браузера.
BROWSER_CANDIDATES = (
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


# --------------------------------------------------------------------------
# Разбор Markdown
# --------------------------------------------------------------------------
def _superscripts(text: str) -> str:
    """Записи вида 10^(-5) и 10^2 — в настоящие верхние индексы.

    Шаблон требует цифру перед знаком степени, поэтому идентификаторы вроде
    ``attack_type`` и имена файлов не затрагиваются.
    """
    text = re.sub(r"(\d)\^\(([^)]+)\)", r"\1<sup>\2</sup>", text)
    return re.sub(r"(\d)\^(\d+)", r"\1<sup>\2</sup>", text)


def _subscripts(text: str) -> str:
    """Записи вида r_1 и C_{L} — в нижние индексы (только внутри формул)."""
    text = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
    return re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", text)


def _inline(text: str) -> str:
    """Оформление внутри строки: код, полужирный, курсив, степени."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", out)
    return _superscripts(out)


def _table(rows: list[str]) -> str:
    """Собрать HTML-таблицу из строк Markdown."""
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], cells[2:]          # cells[1] — строка-разделитель
    parts = ["<table><thead><tr>"]
    parts += [f"<th>{_inline(c)}</th>" for c in header]
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def markdown_to_html(text: str) -> str:
    """Преобразовать используемое подмножество Markdown в HTML."""
    # front matter отбрасывается: он служит метаданными исходника
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]

    lines = text.split("\n")
    out: list[str] = []
    index = 0
    list_open: str | None = None

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append(f"</{list_open}>")
            list_open = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # блок формул ```math ... ```
        if stripped.startswith("```"):
            close_list()
            language = stripped[3:].strip()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            body = html.escape("\n".join(block))
            if language == "math":
                body = _subscripts(_superscripts(body))
                out.append(f'<div class="formula">{body}</div>')
            else:
                out.append(f'<div class="codeblock">{body}</div>')
            continue

        # таблица
        if stripped.startswith("|") and index + 1 < len(lines) and \
                re.match(r"^\|[\s:\-|]+\|$", lines[index + 1].strip()):
            close_list()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            out.append('<div class="tablewrap">' + _table(rows) + "</div>")
            continue

        if not stripped:
            close_list()
            index += 1
            continue

        if stripped == "<!-- pagebreak -->":
            close_list()
            out.append('<div class="pagebreak"></div>')
            index += 1
            continue

        if stripped == "---":
            close_list()
            out.append('<hr class="rule">')
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if stripped.startswith("> "):
            close_list()
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            out.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue

        ordered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        bullet = re.match(r"^[*-]\s+(.*)$", stripped)
        if ordered or bullet:
            want = "ol" if ordered else "ul"
            if list_open != want:
                close_list()
                out.append(f"<{want}>")
                list_open = want
            item = [(ordered or bullet).group(2 if ordered else 1)]
            index += 1
            # продолжение пункта: следующие непустые строки, не начинающие новый блок
            while index < len(lines) and lines[index].strip() and not re.match(
                r"^(#{1,4}\s|\||>|```|[*-]\s|\d+\.\s|---$|<!--)", lines[index].strip()
            ):
                item.append(lines[index].strip())
                index += 1
            out.append(f"<li>{_inline(' '.join(item))}</li>")
            continue

        # абзац: собираем подряд идущие строки
        close_list()
        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip() and \
                not re.match(r"^(#{1,4}\s|\||>|```|[*-]\s|\d+\.\s|---$)", lines[index].strip()):
            paragraph.append(lines[index].strip())
            index += 1
        out.append(f"<p>{_inline(' '.join(paragraph))}</p>")

    close_list()
    return "\n".join(out)


# --------------------------------------------------------------------------
# Оформление и сборка
# --------------------------------------------------------------------------
def build_html(body: str, fonts_css: str = "") -> str:
    """Обернуть содержимое в печатный документ формата A4."""
    font_block = f"<style>\n{fonts_css}\n</style>" if fonts_css else (
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Literata:opsz,wght@7..72,400;7..72,600;7..72,700&'
        'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Объяснимая математическая модель геометрической живости</title>
{font_block}
<style>
  @page {{ size: A4; margin: 20mm 18mm; }}
  *{{ box-sizing: border-box; }}
  html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{
    margin: 0; background: #fff; color: #16191F;
    font-family: "IBM Plex Sans", -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 10.8pt; line-height: 1.58;
  }}
  h1, h2, h3, h4 {{
    font-family: Literata, Georgia, serif; color: #0E1117;
    text-wrap: balance; margin: 0;
  }}
  h1 {{ font-size: 18pt; line-height: 1.22; margin-top: 26pt; margin-bottom: 10pt;
       padding-bottom: 6pt; border-bottom: 2px solid #C8CEDA; break-after: avoid; }}
  h1:first-of-type {{ margin-top: 0; }}
  h2 {{ font-size: 13.5pt; margin-top: 18pt; margin-bottom: 7pt; break-after: avoid; }}
  h3 {{ font-size: 11.5pt; margin-top: 14pt; margin-bottom: 5pt; break-after: avoid; }}
  p {{ margin: 0 0 8pt; text-align: justify; hyphens: auto; }}
  ul, ol {{ margin: 0 0 10pt; padding-left: 20pt; }}
  li {{ margin-bottom: 4pt; }}
  strong {{ font-weight: 600; }}
  code {{
    font-family: "IBM Plex Mono", monospace; font-size: .87em;
    background: #F2F4F8; border: 1px solid #DDE2EC; border-radius: 3px; padding: 0 3px;
  }}
  .formula sub {{ font-style: normal; font-size: .72em; }}
  .formula sup {{ font-style: normal; font-size: .72em; }}
  sup, sub {{ line-height: 0; }}
  .formula {{
    font-family: Literata, Georgia, serif; font-style: italic; font-size: 11pt;
    white-space: pre-wrap; text-align: center; line-height: 1.85;
    background: #F7F8FB; border-left: 2.5px solid #4A6FD4;
    padding: 11pt 14pt; margin: 10pt 0 12pt; break-inside: avoid;
  }}
  .codeblock {{
    font-family: "IBM Plex Mono", monospace; font-size: 9pt; white-space: pre-wrap;
    background: #F2F4F8; border: 1px solid #DDE2EC; border-radius: 4px;
    padding: 9pt 11pt; margin: 8pt 0 12pt; break-inside: avoid;
  }}
  .tablewrap {{ margin: 8pt 0 13pt; break-inside: avoid; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 9.6pt; }}
  th, td {{ border: 1px solid #D5DAE4; padding: 5pt 7pt; text-align: left;
            vertical-align: top; font-variant-numeric: tabular-nums; }}
  th {{ background: #EEF1F7; font-weight: 600; }}
  blockquote {{
    margin: 9pt 0 12pt; padding: 9pt 13pt; background: #FBF0DC;
    border-left: 3px solid #C98B2E; font-size: 10pt; break-inside: avoid;
  }}
  hr.rule {{ border: none; border-top: 1px solid #DDE2EC; margin: 16pt 0; }}
  .pagebreak {{ break-after: page; height: 0; }}
  /* титульный лист: крупный заголовок и воздух вокруг выходных данных */
  body > h1:first-child {{
    font-size: 22pt; border: none; padding-bottom: 0;
    margin-top: 34pt; margin-bottom: 20pt; text-align: center;
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def find_browser() -> str | None:
    """Найти браузер на движке Chromium для печати в PDF."""
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    bundled = sorted(Path("/opt/pw-browsers").glob("*/chrome-linux/chrome"))
    return str(bundled[0]) if bundled else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка PDF научной работы из report.md")
    parser.add_argument("--source", default="report.md")
    parser.add_argument("--output", default="report.pdf")
    parser.add_argument("--fonts-css", default="",
                        help="файл с локальными @font-face (необязательно)")
    args = parser.parse_args()

    logger = setup_logging()
    root = project_root()
    source = root / args.source
    if not source.exists():
        logger.error("Файл не найден: %s", source)
        return 2

    fonts_css = ""
    if args.fonts_css and Path(args.fonts_css).exists():
        fonts_css = Path(args.fonts_css).read_text(encoding="utf-8")

    document = build_html(markdown_to_html(source.read_text(encoding="utf-8")), fonts_css)
    html_path = root / (Path(args.output).stem + ".html")
    html_path.write_text(document, encoding="utf-8")
    logger.info("HTML собран: %s", html_path)

    browser = find_browser()
    if browser is None:
        print(f"\nБраузер на движке Chromium не найден. HTML сохранён: {html_path}")
        print("Откройте его и напечатайте в PDF (Cmd+P -> Сохранить как PDF).")
        return 0

    pdf_path = root / args.output
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", "--virtual-time-budget=20000",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )
    logger.info("PDF собран: %s", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
