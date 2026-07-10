#!/usr/bin/env python3
"""Print exactly which EMOTIV account Cortex sees as logged in.

-32142 ("Unpublished application can only be accessed by the Owner") means the
account logged into EMOTIV Launcher is NOT the account that owns your Cortex
app - even when both display the same email. A native EmotivID (email+password)
and a Google/Apple SSO login create *distinct* accounts under one email, so the
website Owner and the Launcher login can look identical yet differ underneath.

This asks Cortex directly (getUserLogin needs no credentials), so you can
compare the 'username' it reports against the Owner shown on
emotiv.com/my-account/cortex-apps. If they differ, that's the -32142 cause -
recreate the app while logged into emotiv.com with the *same* method Launcher
uses, then swap the new id/secret into .env.
"""

from __future__ import annotations

import json
import ssl
import sys

CORTEX_URL = "wss://localhost:6868"


def main() -> None:
    try:
        import websocket
    except ImportError:
        sys.exit("websocket-client not installed - run: pip install -r requirements.txt")

    try:
        ws = websocket.create_connection(CORTEX_URL, sslopt={"cert_reqs": ssl.CERT_NONE})
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not reach Cortex at {CORTEX_URL} - is EMOTIV Launcher running? ({exc})")

    ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getUserLogin", "params": {}}))
    resp = json.loads(ws.recv())
    ws.close()

    if "error" in resp:
        sys.exit(f"Cortex error: {resp['error']}")

    users = resp.get("result", [])
    if not users:
        print("Cortex reports NO user logged in - open EMOTIV Launcher and sign in.")
        return

    for u in users:
        print("Cortex sees this account logged in:")
        print(f"  username           : {u.get('username')}")
        print(f"  currentOSUId       : {u.get('currentOSUId')}")
        print(f"  loggedInOSUsername : {u.get('loggedInOSUsername')}")
        print(f"  lastLoginTime      : {u.get('lastLoginTime')}")
    print()
    print("Now open emotiv.com/my-account/cortex-apps -> your app and check the Owner.")
    print("If the 'username' above does not match that Owner, that mismatch IS the -32142 cause.")


if __name__ == "__main__":
    main()
