import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args, timeout=600):
    return subprocess.run([sys.executable, "cli.py", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout)


def test_all_runs_without_exceptions():
    r = run("--all")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Traceback" not in r.stdout + r.stderr
    assert "Г4" in r.stdout


def test_task_with_plot(tmp_path):
    out = tmp_path / "g4.png"
    r = run("--task", "Г4", "--ivp", "theta(0)=1,theta'(0)=0", "--interval", "0", "10", "--plot", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists() and out.stat().st_size > 10_000


def test_bad_input_is_reported(tmp_path):
    r = run("y' = x +")
    assert r.returncode == 2
    assert "Ошибка ввода" in r.stdout


def test_menu_survives_bad_input():
    r = subprocess.run([sys.executable, "cli.py"], cwd=ROOT, input="7\n2\ny' = sinn(x)\n1\nZ9\n0\n",
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Нет такого пункта" in r.stdout
    assert "неизвестная функция" in r.stdout
    assert "задачи «Z9» нет" in r.stdout
    assert "До свидания" in r.stdout
