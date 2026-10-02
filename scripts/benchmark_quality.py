import sys
from embedding_bench.cli import main

raise SystemExit(main(["quality", *sys.argv[1:]]))

