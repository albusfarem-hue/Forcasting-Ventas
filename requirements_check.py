import pathlib
import sys

print("=== REQUIREMENTS CHECK ===")
print("PYTHON:", sys.version)

p = pathlib.Path("requirements.txt")
print("REQ EXISTS:", p.exists())

if p.exists():
    txt = p.read_text(encoding="utf-8", errors="replace")
    print("REQ SIZE:", len(txt), "chars")
    print("REQ CONTENT START >>>")
    print(txt)
    print("<<< REQ CONTENT END")
else:
    print("requirements.txt not found in current working directory")