import sys
from embedding_bench.cli import main

raise SystemExit(main(["build-controlled", *sys.argv[1:]]))

