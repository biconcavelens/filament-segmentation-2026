"""Build the arXiv source package: comments stripped, only the files the paper uses, zipped flat.

    python paper/make_arxiv.py        # -> paper_arxiv/ and paper_arxiv.zip
"""
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper_arxiv"

src = (HERE / "main.tex").read_text(encoding="utf8")
lines = []
for line in src.split("\n"):
    line = re.sub(r"(?<!\\)%.*$", "", line).rstrip()  # drop comments, keep \%
    if line or (lines and lines[-1]):  # collapse blank runs left by removed comment lines
        lines.append(line)
tex = "\n".join(lines) + "\n"

figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
(OUT / "main.tex").write_text(tex, encoding="utf8")
for f in figures:
    shutil.copy(HERE / f, OUT / f)
with zipfile.ZipFile(OUT.parent / "paper_arxiv.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        z.write(p, p.name)
print("figures:", figures)
print("TODO left:", "TODO" in tex, "| % left:", len(re.findall(r"(?<!\\)%", tex)))
