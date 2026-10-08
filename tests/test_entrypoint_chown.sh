#!/bin/sh
# Регрессионный тест логики chown в entrypoint.sh (issue #104).
#
# Проверяет без Docker/root, что:
#   A. rootless/userns (uid_map "0 <non-zero> ...") — chown НЕ вызывается,
#      команда всё равно запускается;
#   B. rootful + /data не записывается пользователем `app` — chown вызывается;
#   C. rootful + /data записывается пользователем `app` — chown НЕ вызывается.
#
# Сценарий A эмулируется через `unshare -Ur` (контейнерный root → хостовый
# пользователь, как в rootless Docker); если unshare недоступен — пропускается.
# `id`, `setpriv`, `chown`, `mkdir` подменяются заглушками через PATH, поэтому
# тест ничего не меняет в системе.
#
# Запуск: sh tests/test_entrypoint_chown.sh
set -eu

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
SCRIPT="$ROOT/entrypoint.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STUB="$WORK/stub"
CHOWN_MARK="$WORK/chown.called"
RAN_MARK="$WORK/ran"
export CHOWN_MARK RAN_MARK
mkdir -p "$STUB"

cat > "$STUB/id" <<'EOF'
#!/bin/sh
case "${1:-}" in
  -u) echo 0 ;;
  -G) echo "${FAKE_APP_GROUPS:-1000}" ;;
  *) echo 0 ;;
esac
EOF

cat > "$STUB/setpriv" <<'EOF'
#!/bin/sh
# Интересует только проба записи `... test -w /data`; остальное — no-op.
for a in "$@"; do
  [ "$a" = "test" ] && exit "${FAKE_TEST_EXIT:-1}"
done
exit 0
EOF

cat > "$STUB/chown" <<'EOF'
#!/bin/sh
: > "${CHOWN_MARK:?}"
exit 0
EOF

cat > "$STUB/mkdir" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$STUB"/*

fail=0
check() { # <описание> <yes|no — ожидается ли вызов chown>
  if [ "$2" = "yes" ]; then
    if [ -f "$CHOWN_MARK" ]; then echo "PASS: $1 (chown вызван)"; else echo "FAIL: $1 (chown НЕ вызван)"; fail=1; fi
  else
    if [ -f "$CHOWN_MARK" ]; then echo "FAIL: $1 (chown вызван)"; fail=1; else echo "PASS: $1 (chown не вызван)"; fi
  fi
}

# A. rootless/userns.
rm -f "$CHOWN_MARK" "$RAN_MARK"
if command -v unshare >/dev/null 2>&1 && unshare -Ur true 2>/dev/null; then
  # shellcheck disable=SC2016  # $RAN_MARK раскрывается внутри кавычек-одиночек sh -c
  unshare -Ur env PATH="$STUB:$PATH" sh "$SCRIPT" sh -c 'echo RAN > "$RAN_MARK"'
  check "rootless/userns: chown запрещён" no
  if [ -f "$RAN_MARK" ]; then echo "PASS: rootless/userns: команда выполнена"; else echo "FAIL: rootless/userns: команда НЕ выполнена"; fail=1; fi
else
  echo "SKIP: сценарий rootless (unshare -Ur недоступен)"
fi

# B. rootful, каталог не записывается.
rm -f "$CHOWN_MARK"
env PATH="$STUB:$PATH" FAKE_TEST_EXIT=1 sh "$SCRIPT" true
check "rootful, запись невозможна: chown нужен" yes

# C. rootful, каталог записывается.
rm -f "$CHOWN_MARK"
env PATH="$STUB:$PATH" FAKE_TEST_EXIT=0 sh "$SCRIPT" true
check "rootful, запись возможна: chown не нужен" no

echo "----"
if [ "$fail" = "0" ]; then echo "ALL PASS"; else echo "SOME FAILED"; fi
exit "$fail"
