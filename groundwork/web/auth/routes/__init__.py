"""Auth HTTP surface — a same-name package, one router.

guards owns the router + dependencies; account and keys_sessions add
routes on import. Public surface matches the old module.
"""
from __future__ import annotations

from .guards import (router, current_user, require_admin,    # noqa: F401
                     refuse_key_auth, _refuse_key_auth)
from . import account, keys_sessions                          # noqa: F401
