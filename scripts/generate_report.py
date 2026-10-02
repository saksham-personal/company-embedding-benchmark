import sys
from embedding_bench.cli import main

raise SystemExit(main(["report", *sys.argv[1:]]))

