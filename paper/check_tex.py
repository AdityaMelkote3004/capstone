"""Static checks for main.tex / refs.bib (no LaTeX compiler on this machine)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
BS = "\\"

cites = set(k.strip() for g in re.findall(re.escape(BS + "cite{") + r"([^}]*)\}", tex)
            for k in g.split(","))
keys = set(re.findall(re.escape(BS + "bibitem{") + r"([^}]+)\}", tex))
print("cited keys:", len(cites), "| bib entries:", len(keys))
print("cited but missing from bib:", sorted(cites - keys))
print("in bib but never cited:", sorted(keys - cites))

body = re.sub(r"(?<!\\)%.*", "", tex)
print("brace balance (0 = ok):", body.count("{") - body.count("}"))
for env in sorted(set(re.findall(re.escape(BS + "begin{") + r"([A-Za-z*]+)\}", body))):
    b = body.count(BS + "begin{" + env + "}")
    e = body.count(BS + "end{" + env + "}")
    print(f"env {env}: begin {b} end {e}", "" if b == e else "<-- UNBALANCED")
print("$ count even:", body.count("$") % 2 == 0)
for f in re.findall(re.escape(BS + "includegraphics") + r"\[[^\]]*\]\{([^}]+)\}", tex):
    print("figure", f, "exists:", os.path.exists(os.path.join(HERE, f)))
labels = set(re.findall(re.escape(BS + "label{") + r"([^}]+)\}", tex))
refs = set(re.findall(re.escape(BS + "ref{") + r"([^}]+)\}", tex))
print("refs without labels:", refs - labels, "| labels never referenced:", labels - refs)

# unescaped special characters outside math, tabular rows and commands
doc = body.split(BS + "begin{document}")[1]
for n, line in enumerate(doc.splitlines(), 1):
    text = re.sub(r"\$[^$]*\$", "", line)
    if "&" in text and BS + BS in text:
        continue  # table row
    for ch in "&_#":
        for m in re.finditer(re.escape(ch), text):
            if m.start() == 0 or text[m.start() - 1] != BS:
                if ch == "_" and (BS + "label" in text or BS + "ref" in text or BS + "cite" in text
                                  or BS + "includegraphics" in text):
                    continue
                print(f"unescaped {ch!r} in doc line {n}: {line.strip()[:90]}")
words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", doc).split())
print("approx words in document:", words)
