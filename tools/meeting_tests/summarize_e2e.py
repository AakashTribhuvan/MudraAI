import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for r in d["rows"]:
    print(r)
print("PHRASES", d["phrases"]); print("UNSIGNED", d["unsigned"]); print("SKIPPED", d["skipped"], "ECHO", d["echoes"])
for s in d["snaps"]:
    print("  ", s.encode("ascii", "replace").decode())
print("AFTER STOP", d["afterStop"]); print("LOGS", d["logs"])
