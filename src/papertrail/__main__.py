"""`python -m papertrail` startup: resource guard before CLI imports."""
from .cli_bootstrap import main

raise SystemExit(main())
