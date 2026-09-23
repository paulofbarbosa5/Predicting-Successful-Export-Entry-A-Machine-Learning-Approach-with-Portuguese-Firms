"""
find_rf_params.py -- procura em todo o projecto os hiperparametros
seleccionados que ainda faltam: Random Forest (max_depth, min_samples_leaf),
logit (C) e elastic net (C, l1_ratio).

Correr a partir de distance-to-export-python:

    python scripts\\find_rf_params.py

NAO use a redireccao ">" do PowerShell: nesta pasta do OneDrive falha.
O script grava o relatorio sozinho em

    scripts\\outputs\\diagnostics\\rf_params_audit.txt

Para gravar noutro sitio:

    python scripts\\find_rf_params.py --out C:\\temp\\rf_params.txt

Nao abre microdados: salta raw, processed, models e predictions, e so le
csv, json, txt, log, md e tex.
"""

import os
import re
import sys

SKIP = {
    ".git", "__pycache__", ".venv", "venv", "env", ".ipynb_checkpoints",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "raw", "processed", "models", "predictions",
}
EXTS = (".csv", ".json", ".txt", ".log", ".md", ".tex")
MAX_BYTES = 30_000_000

PAT = re.compile(
    r"min_samples_leaf|'max_depth'|\"max_depth\"|'l1_ratio'|\"l1_ratio\"|"
    r"random_forest.{0,200}?best_params|best_params.{0,200}?random_forest|"
    r"elastic_net.{0,200}?best_params|best_params.{0,200}?elastic_net|"
    r"\blogit\b.{0,200}?best_params",
    re.I | re.S,
)
LINE = re.compile(
    r"best_params|min_samples_leaf|max_depth|l1_ratio|n_estimators|"
    r"learning_rate|\bchosen\b|'C'|\"C\"",
    re.I,
)

LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def parse_args():
    argv = sys.argv[1:]
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    return (argv[0] if argv else None), out


def roots(roots_arg):
    if roots_arg:
        return [os.path.abspath(roots_arg)]
    here = os.path.abspath(".")
    return [here, os.path.dirname(here)]


def default_out():
    folder = os.path.join(os.path.abspath("."), "scripts", "outputs", "diagnostics")
    if not os.path.isdir(folder):
        folder = os.path.join(os.path.abspath("."), "outputs", "diagnostics")
    return os.path.join(folder, "rf_params_audit.txt")


def save(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(LINES) + "\n")
        print(f"\nRelatorio gravado em: {path}")
    except OSError as exc:
        print(f"\nNao foi possivel gravar o relatorio ({exc}).")
        print("O conteudo esta acima, no ecra.")


def main():
    roots_arg, out = parse_args()
    say("find_rf_params.py")
    say("raizes analisadas:")
    for r in roots(roots_arg):
        say(f"  {r}")

    seen = set()
    hits = 0
    for root in roots(roots_arg):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP]
            for name in sorted(filenames):
                if not name.lower().endswith(EXTS):
                    continue
                path = os.path.realpath(os.path.join(dirpath, name))
                if path in seen:
                    continue
                seen.add(path)
                try:
                    if os.path.getsize(path) > MAX_BYTES:
                        continue
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                if not PAT.search(text):
                    continue
                hits += 1
                say(f"\n=== {path}")
                if name.lower().endswith(".csv"):
                    say(f"    cabecalho: {text.splitlines()[0][:300]}")
                for i, line in enumerate(text.splitlines(), 1):
                    if LINE.search(line):
                        say(f"{i:5}: {line.strip()[:400]}")

    say(f"\n=== ficheiros com seleccoes gravadas: {hits} ===")
    if not hits:
        say("Nada encontrado. Os outputs de afinacao do Random Forest,")
        say("do logit e do elastic net nao ficaram nesta copia.")
        say("Corra entao, sem a flag --quick:")
        say("    python scripts\\07_run_main_models.py --seed 123")

    save(out or default_out())


if __name__ == "__main__":
    main()
