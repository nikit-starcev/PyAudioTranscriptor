#!/bin/sh
# Entrypoint контейнера PyAudioTranscriptor.
#
# Приложение работает от непривилегированного пользователя `app` (uid 1000).
# Если контейнер стартовал как root (так его запускают `docker run`/compose),
# скрипт сначала выравнивает владельца тома /data, затем сбрасывает привилегии
# через setpriv. Это необходимо, потому что на Docker Desktop bind-mount
# `./web-data:/data` виден внутри контейнера как root:root (root-squash), и от
# пользователя `app` в него нельзя писать. На обычном Linux-Docker, где
# владелец каталога совпадает с `app`, chown не выполняется.
#
# PUID/PGID (необязательно) позволяют подогнать uid/gid пользователя `app` под
# владельца каталогов на хосте: `PUID=$(id -u) PGID=$(id -g) docker compose up`.
set -eu

if [ "$(id -u)" = "0" ]; then
    if command -v usermod >/dev/null 2>&1 && [ -n "${PUID:-}" ] && [ "${PUID}" != "0" ]; then
        usermod -o -u "${PUID}" app
    fi
    if command -v groupmod >/dev/null 2>&1 && [ -n "${PGID:-}" ] && [ "${PGID}" != "0" ]; then
        groupmod -o -g "${PGID}" app
    fi

    mkdir -p /data
    # Чиним владельца тома только если `app` действительно не может писать,
    # чтобы не делать дорогой chown -R при каждом старте.
    if ! setpriv --reuid=app --regid=app --clear-groups -- test -w /data; then
        chown -R app:app /data 2>/dev/null || true
    fi

    # Создаём каталоги конвейера от имени `app`, чтобы они не остались root-owned.
    setpriv --reuid=app --regid=app --clear-groups -- \
        mkdir -p /data/voices /data/models

    # Итоговый набор дополнительных групп: собственные группы `app` (из
    # /etc/group, включая video/render) плюс группы, добавленные через
    # `group_add` (кроме root-группы 0 — её процесс-родитель тянет по
    # умолчанию и она не должна сохраняться у непривилегированного процесса).
    app_groups="$(id -G app 2>/dev/null || echo '')"
    add_groups="$(grep -E '^Groups:' /proc/self/status | cut -f2)"
    groups_list="$(printf '%s %s\n' "${app_groups}" "${add_groups}" \
        | tr ' ' '\n' | sed '/^$/d; /^0$/d' | sort -n -u | paste -sd, -)"

    if [ -n "${groups_list}" ]; then
        exec setpriv --reuid=app --regid=app --groups "${groups_list}" "$@"
    fi
    exec setpriv --reuid=app --regid=app --clear-groups "$@"
fi

exec "$@"
