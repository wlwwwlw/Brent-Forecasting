"""Проверка исходного эксперимента и его сохранённых вычислительных файлов."""

import hashlib
from experiment import ROOT

ARCHIVED = {
    "src/run_final.py": "provenance/v1/run_final.py",
    "src/report_final.py": "provenance/v1/report_final.py",
}


def verify_file(name, expected, root=ROOT):
    path = root / ARCHIVED.get(name, name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Original experiment changed: {name}")


def verify_files(checksums, root=ROOT):
    for name, expected in checksums.items():
        verify_file(name, expected, root)
