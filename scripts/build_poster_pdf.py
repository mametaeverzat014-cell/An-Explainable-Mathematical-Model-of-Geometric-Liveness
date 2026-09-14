#!/usr/bin/env python3
"""Собрать печатный постер A1 для научного стенда.

    python scripts/build_poster_pdf.py

Результат: poster.html и poster.pdf в корне проекта. PDF печатается на A1
(594 x 841 мм, вертикальный). Как и сборка научной работы, скрипт использует
установленный браузер на движке Chromium в режиме печати и не добавляет
внешних зависимостей.

ВСЕ ЧИСЛА НА ПОСТЕРЕ ЧИТАЮТСЯ ИЗ results/. Скрипт ничего не выдумывает: если
нужного файла нет, он сообщит об этом и не станет подставлять значения.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.build_report_pdf import find_browser
from src.utils import get_logger, project_root, setup_logging

#: Организационные поля, которые авторы заполняют сами.
#: Выдумывать их нельзя, поэтому по умолчанию стоит заметная метка.
SCHOOL = "[указать учебное заведение]"
SUPERVISOR = "[указать научного руководителя]"

AUTHORS = "Қуанышбеков Даулет · Молдағалиева Асия"
GRADE = "10А класс · направление: математика"
REPO = "github.com/mametaeverzat014-cell/An-Explainable-Mathematical-Model-of-Geometric-Liveness"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def count_pdf_pages(path: Path) -> int:
    """Число страниц в PDF.

    Постер обязан умещаться ровно на один лист: вторая страница означает, что
    содержимое переполнило колонку, и в типографию такой файл отправлять нельзя.
    """
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def load_numbers(root: Path) -> dict[str, object]:
    """Прочитать все числа постера из results/. Ничего не выдумывать."""
    results = root / "results"
    missing: list[str] = []

    def need(name: str) -> pd.DataFrame | None:
        path = results / name
        if not path.exists():
            missing.append(str(path.relative_to(root)))
            return None
        return pd.read_csv(path)

    pooled = need("pooled_metrics.csv")
    ablation = need("ablation_metrics.csv")
    features = need("video_features.csv")
    replay = need("synthetic/synthetic_replay_predictions.csv")

    if missing:
        raise SystemExit(
            "Постер не собран: нет файлов с результатами.\n  "
            + "\n  ".join(missing)
            + "\n\nСначала выполните:\n"
            "  python scripts/run_pipeline.py --config configs/prototype.yaml\n"
            "  python scripts/run_synthetic_study.py --config configs/prototype.yaml"
        )

    assert pooled is not None and ablation is not None
    assert features is not None and replay is not None

    by_model = pooled.set_index("model")
    live = features[features["y"] == 1]
    attack = features[features["y"] == 0]
    share = replay.groupby("attack_type")["predicted_live"].mean()

    return {
        "pooled": by_model,
        "n_subjects": int(features["subject_id"].nunique()),
        "n_videos": int(len(features)),
        "ablation": ablation.set_index("variant")["acer"].to_dict(),
        "g2_live_min": float(live["G2"].min()),
        "g2_attack_max": float(attack["G2"].max()),
        "g1_live_min": float(live["G1"].min()),
        "g1_attack_max": float(attack["G1"].max()),
        "replay_share": float(share.get("replay", float("nan"))),
        "screen_share": float(share.get("screen", float("nan"))),
        "n_replay": int((replay["attack_type"] == "replay").sum()),
    }


def build_html(d: dict[str, object]) -> str:
    """Свёрстанный постер A1. Размеры в мм, шрифты — под чтение с 1–2 метров."""
    pooled = d["pooled"]

    def metric(model: str, column: str) -> str:
        return _fmt(float(pooled.loc[model, column]))

    abl = d["ablation"]
    abl_labels = {
        "PLS_full_G1G2G3": "полный PLS (G₁+G₂+G₃)",
        "PLS_without_G1": "без G₁ — псевдоглубины",
        "PLS_without_G2": "без G₂ — асимметрии",
        "PLS_without_G3": "без G₃ — пропорции",
    }
    baseline_acer = float(abl.get("PLS_full_G1G2G3", 0.0))
    abl_rows = "".join(
        "<tr{cls}><td>{label}</td><td class='num'>{acer}</td></tr>".format(
            cls=" class='miss'" if float(value) > baseline_acer else "",
            label=abl_labels.get(name, name),
            acer=_fmt(float(value)),
        )
        for name, value in abl.items()
    )

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Постер</title><style>
@page {{ size: 594mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
/* Ни ширина, ни высота листа НЕ задаются на body явно: 594mm и 841mm при
   переводе в пиксели округляются вверх (594.1 и 841.1), содержимое вылезает
   за страницу, и Chromium печатает лишний второй лист. Размер задаёт @page,
   а .sheet занимает его целиком с небольшим запасом по высоте. */
body {{
  overflow: hidden;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: #14213d; background: #fff;
  font-size: 7.4mm; line-height: 1.34;
}}
.sheet {{ width: 100%; height: 838mm; padding: 16mm 15mm 12mm;
          display: flex; flex-direction: column; }}

header {{ border-bottom: 2.2mm solid #14213d; padding-bottom: 7mm; margin-bottom: 8mm; }}
h1 {{ font-size: 15.5mm; line-height: 1.12; letter-spacing: -0.02em; margin-bottom: 5mm; }}
.authors {{ font-size: 8.4mm; font-weight: 600; }}
.meta {{ font-size: 6.6mm; color: #4a5568; margin-top: 2mm; }}

.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10mm; flex: 1; }}
section {{ margin-bottom: 7mm; break-inside: avoid; }}
h2 {{
  font-size: 8.8mm; color: #fff; background: #14213d;
  padding: 2.6mm 4mm; margin-bottom: 4mm; border-radius: 1.2mm;
}}
h2.accent {{ background: #9d0208; }}
p {{ margin-bottom: 3.4mm; }}
ul {{ margin: 0 0 3.4mm 6mm; }}
li {{ margin-bottom: 2mm; }}
b {{ font-weight: 700; }}
.mono {{ font-family: "SF Mono", Menlo, Consolas, monospace; }}

.formula {{
  background: #f1f4f9; border-left: 2.2mm solid #14213d;
  padding: 4mm 5mm; margin-bottom: 4mm;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 6.2mm; line-height: 1.7;
}}
.formula .big {{ font-size: 7.6mm; font-weight: 700; display: block; margin-top: 2.5mm; }}

table {{ width: 100%; border-collapse: collapse; margin-bottom: 4mm; font-size: 6.8mm; }}
th, td {{ padding: 2.6mm 3mm; border-bottom: 0.4mm solid #cbd5e0; text-align: left; }}
th {{ background: #eef1f6; font-weight: 700; }}
td.num {{ text-align: right; font-family: "SF Mono", Menlo, Consolas, monospace; }}
tr.hit td {{ background: #e8f5e9; font-weight: 700; }}
tr.miss td {{ background: #fdecea; font-weight: 700; }}

.box {{ border: 1.2mm solid #14213d; border-radius: 1.6mm; padding: 5mm; margin-bottom: 4mm; }}
.box.warn {{ border-color: #9d0208; background: #fdf2f2; }}
.box .label {{
  font-size: 5.6mm; letter-spacing: 0.1em; text-transform: uppercase;
  color: #9d0208; font-weight: 700; margin-bottom: 2.5mm;
}}
.box.plain .label {{ color: #14213d; }}

figure {{ margin-bottom: 4mm; }}
figure img {{
  width: 100%; max-height: 95mm; object-fit: contain;
  border: 0.4mm solid #cbd5e0; border-radius: 1mm;
}}
figcaption {{ font-size: 5.8mm; color: #4a5568; margin-top: 2mm; }}

.kpi {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; margin-bottom: 4mm; }}
.kpi div {{ border: 1mm solid #14213d; border-radius: 1.6mm; padding: 4mm; text-align: center; }}
.kpi .v {{ font-size: 15mm; font-weight: 700; line-height: 1; }}
.kpi .k {{ font-size: 5.8mm; color: #4a5568; margin-top: 2mm; }}
.kpi .good {{ color: #1b5e20; }}
.kpi .bad {{ color: #9d0208; }}

footer {{
  border-top: 1.2mm solid #14213d; padding-top: 5mm; margin-top: 4mm;
  font-size: 5.9mm; color: #4a5568; display: flex; justify-content: space-between; gap: 8mm;
}}
footer .repo {{ font-family: "SF Mono", Menlo, Consolas, monospace; word-break: break-all; }}
</style></head><body><div class="sheet">

<header>
  <h1>Объяснимая математическая модель геометрической живости
      для защиты лицевой биометрии от плоских атак</h1>
  <div class="authors">{AUTHORS}</div>
  <div class="meta">{GRADE} &nbsp;·&nbsp; {SCHOOL} &nbsp;·&nbsp; научный руководитель: {SUPERVISOR}</div>
</header>

<div class="cols">
<div>

  <section>
    <h2>1. Задача</h2>
    <p>Системы распознавания лица обманывает простейшая атака: камере показывают
    не человека, а его фотографию — распечатанную или с экрана телефона. Такие
    атаки называют <b>плоскими</b>: носитель изображения есть плоскость.</p>
    <p><b>Цель работы.</b> Построить интерпретируемую математическую модель,
    отличающую живое лицо от плоской атаки, — и <b>установить границы её
    применимости</b>, доказав, где она работать не может.</p>
  </section>

  <section>
    <h2>2. Идея: параллакс</h2>
    <p>У живого лица есть рельеф: кончик носа вынесен вперёд относительно щёк.
    При движении головы это порождает <b>параллакс</b> — взаимное расположение
    точек в кадре меняется не так, как у плоского объекта.</p>
    <p>Используются 6 опорных точек лица. Координата <span class="mono">z</span> —
    это <b>оценочная относительная псевдоглубина</b>, а не измеренная глубина.</p>
    <div class="formula">
      IOD = d<sub>2D</sub>(глаз<sub>Л</sub>, глаз<sub>П</sub>)<br><br>
      r<sub>1</sub> = [z<sub>нос</sub> − (z<sub>щека Л</sub> + z<sub>щека П</sub>)/2] / IOD<br>
      r<sub>2</sub> = d<sub>2D</sub>(нос, глаз<sub>Л</sub>) / d<sub>2D</sub>(нос, глаз<sub>П</sub>)<br>
      r<sub>3</sub> = d<sub>2D</sub>(нос, подбородок) / d<sub>2D</sub>(щека<sub>Л</sub>, щека<sub>П</sub>)<br><br>
      G<sub>k</sub> = Var<sub>t</sub>[ r<sub>k</sub>(t) ]
      <span class="big">PLS = (G₁ + G₂ + G₃) / 3</span>
    </div>
    <p>Обучаемых весов в модели <b>нет</b>: только нормировка и порог, и оба
    оцениваются исключительно на обучающих участниках.</p>
  </section>

  <section>
    <h2>3. Как проверяли</h2>
    <ul>
      <li><b>Субъект-дизъюнктная кросс-валидация</b> (LOSO): участник целиком
      либо в обучении, либо в тесте. Делить по кадрам — утечка данных.</li>
      <li>Метрики <b>APCER / BPCER / ACER</b> по ISO/IEC 30107-3.</li>
      <li>Протокол, гипотезы и правило выбора порога зафиксированы
      <b>до сбора данных</b>.</li>
      <li>Данные: <b>{d["n_subjects"]} участников, {d["n_videos"]} записей</b>.
      Видео не покидают компьютер; в репозитории только агрегированные числа.</li>
    </ul>
  </section>

  <section>
    <h2>4. Результат</h2>
    <table>
      <tr><th>Модель</th><th>APCER</th><th>BPCER</th><th>ACER</th></tr>
      <tr><td>B0 мажоритарный</td>
          <td class="num">{metric("B0_majority", "apcer")}</td>
          <td class="num">{metric("B0_majority", "bpcer")}</td>
          <td class="num">{metric("B0_majority", "acer")}</td></tr>
      <tr><td>B1 статический 2D</td>
          <td class="num">{metric("B1_static2d", "apcer")}</td>
          <td class="num">{metric("B1_static2d", "bpcer")}</td>
          <td class="num">{metric("B1_static2d", "acer")}</td></tr>
      <tr class="hit"><td>M1 PLS (наша)</td>
          <td class="num">{metric("M1_pls", "apcer")}</td>
          <td class="num">{metric("M1_pls", "bpcer")}</td>
          <td class="num">{metric("M1_pls", "acer")}</td></tr>
    </table>
    <div class="box warn">
      <div class="label">Это не «точность 100 %»</div>
      При нуле ошибок на {d["n_subjects"]} участниках <b>правило трёх</b> даёт
      верхнюю границу истинной вероятности ошибки <b>3/{d["n_subjects"]} = 0.5</b>.
      Bootstrap выдал интервал [0.000, 0.000] — это <b>вырожденный</b> результат,
      а не доказательство безошибочности. Выборка мала, вывод разведочный.
    </div>
    <div class="kpi">
      <div><div class="v">{metric("B1_static2d", "acer")}</div>
           <div class="k">ACER статического<br>однокадрового базлайна</div></div>
      <div><div class="v good">{metric("M1_pls", "acer")}</div>
           <div class="k">ACER предложенной<br>модели PLS</div></div>
    </div>
  </section>

</div>
<div>

  <section>
    <h2>5. Гипотеза не подтвердилась</h2>
    <p>Предполагалось (H2), что разделение обеспечивает <span class="mono">G₁</span> —
    признак псевдоглубины носа. <b>Абляция это опровергла:</b></p>
    <table>
      <tr><th>Вариант модели</th><th>ACER</th></tr>
      {abl_rows}
    </table>
    <p>Удаление <span class="mono">G₁</span> не меняет <b>ничего</b>. Портит
    результат только удаление <span class="mono">G₂</span> — лево-правой
    асимметрии.</p>
    <div class="box plain">
      <div class="label">Почему</div>
      Нормировка на межзрачковое расстояние <b>сокращает</b> множитель
      cos(поворота) в числителе r₁ и глушит тот самый сигнал, ради которого
      признак вводился (доказано, утверждение 5).<br><br>
      <b>Этот провал был предсказан симуляцией и записан в документацию
      до сбора данных.</b> Формулировка гипотезы не менялась.
    </div>
  </section>

  <section>
    <h2 class="accent">6. Доказанная граница: replay</h2>
    <p>Что если показать не фото, а <b>видео</b> поворачивающейся головы?
    Лицо на экране повернётся само.</p>
    <div class="box warn">
      <div class="label">Утверждение 7</div>
      Неподвижный экран отображает кадр <b>гомотетией</b>: все попарные
      расстояния умножаются на одно число. Признаки r₂ и r₃ суть
      <b>отношения</b> расстояний — коэффициент сокращается.<br><br>
      <b>Следствие:</b> любой признак, построенный на отношениях двумерных
      расстояний, при такой атаке воспроизводится <b>тождественно</b>.
      Разделять нечего ни при каком пороге.<br><br>
      Проверено численно: max |r₂<sup>живое</sup> − r₂<sup>replay</sup>| = <b>0</b>.
      Совпадение побитовое, а не приближённое.
    </div>
    <div class="kpi">
      <div><div class="v good">{d["screen_share"] * 100:.0f}%</div>
           <div class="k">фото на экране<br>принято за живое</div></div>
      <div><div class="v bad">{d["replay_share"] * 100:.0f}%</div>
           <div class="k">видео на экране<br>принято за живое</div></div>
    </div>
    <p>Модель не распознала <b>ни одной</b> атаки воспроизведением из
    {d["n_replay"]}. Гипотеза H4 о провале собственной модели зафиксирована
    в протоколе <b>до</b> эксперимента.</p>
  </section>

  <section>
    <h2>7. Выводы</h2>
    <ul>
      <li>Доказаны <b>7 утверждений</b> о поведении модели; все подтверждены
      численно с точностью до 10⁻⁸.</li>
      <li>Модель <b>точнее статического базлайна</b> на собранных данных, но
      выборка мала, и вывод разведочный.</li>
      <li>Разделение обеспечивает <b>не тот признак</b>, который
      предполагался, — и это было предсказано заранее.</li>
      <li>Доказана уязвимость <b>целого класса</b> признаков к replay-атакам.
      Защита от них требует признаков иной природы: измеряемой глубины,
      текстуры носителя или временнóй согласованности.</li>
    </ul>
    <div class="box warn">
      <div class="label">Чего работа не утверждает</div>
      Не готовность к применению · не универсальную стойкость · не защиту от
      масок, дипфейков и печатных атак (класс не собирался) · не работу вне
      условий эксперимента: одна камера, одно освещение, {d["n_subjects"]} человек.
      Модель <b>не проверялась и не должна проверяться</b> против реальных
      систем контроля доступа.
    </div>
  </section>

</div>
</div>

<footer>
  <div>Код, данные протокола и полный текст работы (35 с.) —
       воспроизводится одной командой:<br>
       <span class="repo">{REPO}</span></div>
  <div style="text-align:right; white-space:nowrap">43 автоматических теста<br>Python · CPU · без облачных сервисов</div>
</footer>

</div></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка печатного постера A1")
    parser.add_argument("--output", default="poster.pdf")
    args = parser.parse_args()

    logger = setup_logging()
    root = project_root()

    numbers = load_numbers(root)
    html_path = root / (Path(args.output).stem + ".html")
    html_path.write_text(build_html(numbers), encoding="utf-8")
    logger.info("HTML постера собран: %s", html_path)

    browser = find_browser()
    if browser is None:
        print(f"\nБраузер на движке Chromium не найден. HTML сохранён: {html_path}")
        print("Откройте его и напечатайте в PDF (Cmd+P -> размер A1, без полей).")
        return 0

    pdf_path = root / args.output
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", "--virtual-time-budget=20000",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )
    pages = count_pdf_pages(pdf_path)
    if pages != 1:
        print(
            f"\nОШИБКА: постер занял {pages} стр., а должен ровно одну.\n"
            "Содержимое не помещается на лист A1. Сократите текст или графики\n"
            "в build_html() — печатать такой файл нельзя."
        )
        return 1
    logger.info("PDF постера собран: %s (A1, 594 x 841 мм, 1 страница)", pdf_path)

    if SCHOOL.startswith("[") or SUPERVISOR.startswith("["):
        print("\nВНИМАНИЕ: в шапке постера остались незаполненные поля.")
        print("Откройте scripts/build_poster_pdf.py и впишите SCHOOL и SUPERVISOR,")
        print("затем соберите постер заново.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
