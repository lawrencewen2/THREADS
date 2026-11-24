"""总组合回撤红线回测：python scripts/astock_redline.py（读 data/research，几秒钟）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import redline  # noqa: E402

if __name__ == "__main__":
    res = redline.run()
    redline.write(res)
    print(redline.to_markdown(res))
