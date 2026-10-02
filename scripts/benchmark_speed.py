import sys
from embedding_bench.cli import main

raise SystemExit(main(["speed", *sys.argv[1:]]))

