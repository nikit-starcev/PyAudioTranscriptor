#!/bin/sh
# Entrypoint контейнера PyAudioTranscriptor.
#
# Приложение работает от непривилегированного пользователя `app` (uid 1000).
# Если контейнер стартовал как root, скрипт при необходимости выравнивает
# владельца тома /data и сбрасывает привилегии через setpriv.
#
# Различаем два принципиально разных случая по тому, на какой хостовый uid
# отображён КОНТЕЙНЕРНЫЙ root (первая строка /proc/self/uid_map):
#
# 1. rootful (обычный Linux-Docker, Docker Desktop). Контейнерный root — это
#    настоящий root хоста (uid 0 → host uid 0). На Docker Desktop bind-mount
#    `./web-data:/data` виден внутри как root:root, и записать в него от `app`
#    нельзя, поэтому владелец тома выравнивается (`chown -R app:app`); на
#    обычном Linux владелец уже совпадает с `app`, и chown не выполняется.
#    Приложение запускается от `app`.
#
# 2. rootless Docker и userns-remap. Контейнерный uid 0 отображён на НЕнулевой
#    хостовый uid (в rootless — на uid пользователя, запустившего демон;
#    остальные uid — на subuid). При этом файлы хоста, принадлежащие этому
#    пользователю, видны внутри как root:root, а in-container uid `app` (1000)
#    отображается на subuid (например, 525287). Рекурсивный
#    `chown -R app:app /data` переписал бы владельца ВСЕХ хостовых файлов
#    `web-data` на subuid и сломал бы доступ хостовому серверу
#    (SQLite: "attempt to write a readonly database"). Поэтому здесь chown не
#    выполняется вовсе, а приложение запускается от контейнерного root: он и
#    есть хостовый пользователь, так что файлы создаются под правильным
#    владельцем и хостовый сервер продолжает работать. `PUID`/`PGID` тут не
#    помогают (in-container uid 1000 → host subuid), поэтому не применяются.
#
# PUID/PGID (необязательно, только rootful) позволяют подогнать uid/gid
# пользователя `app` под владельца каталогов на хосте:
#   `PUID=$(id -u) PGID=$(id -g) docker compose up`.
set -eu

# Подсказка, если каталог моделей ещё пуст: модели скачиваются из веб-UI в
# /data/models и переживают перезапуски, поэтому это не ошибка — просто
# подсказываем, где их взять.
warn_missing_models() {
    if [ -d /data/models ] && [ -z "$(ls -A /data/models 2>/dev/null)" ]; then
        echo "PyAudioTranscriptor: каталог /data/models пуст — скачайте модели" \
             "в веб-UI (http://<host>:${PORT:-8790}/) или смонтируйте готовый" \
             "каталог моделей." >&2
    fi
}

if [ "$(id -u)" = "0" ]; then
    # uid контейнерного root на хосте: 0 — «настоящий» root (rootful); иначе
    # имеем user namespace (rootless/userns-remap), где трогать владельца
    # хостовых файлов нельзя.
    root_host_uid=""
    read -r _uid0 root_host_uid _rest < /proc/self/uid_map 2>/dev/null || true

    if [ "${root_host_uid:-}" = "0" ]; then
        # --- rootful: обычный Linux-Docker и Docker Desktop ---
        if command -v usermod >/dev/null 2>&1 && [ -n "${PUID:-}" ] && [ "${PUID}" != "0" ]; then
            usermod -o -u "${PUID}" app
        fi
        if command -v groupmod >/dev/null 2>&1 && [ -n "${PGID:-}" ] && [ "${PGID}" != "0" ]; then
            groupmod -o -g "${PGID}" app
        fi

        mkdir -p /data
        # Чиним владельца тома только если `app` действительно не может писать
        # (Docker Desktop отдаёт bind-mount как root:root; на обычном Linux
        # владелец уже совпадает с `app`), чтобы не делать дорогой chown -R
        # при каждом старте. Root здесь настоящий, поэтому хостовый владелец
        # файлов не переопределяется.
        if ! setpriv --reuid=app --regid=app --clear-groups -- test -w /data; then
            chown -R app:app /data 2>/dev/null || true
        fi

        # Создаём каталоги конвейера от имени `app`, чтобы они не остались root-owned.
        setpriv --reuid=app --regid=app --clear-groups -- \
            mkdir -p /data/voices /data/models
        warn_missing_models

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

    # --- rootless / userns-remap: владельца хостовых файлов не трогаем ---
    # Контейнерный root отображён на хостового пользователя (в rootless uid 0 →
    # host uid пользователя) и уже владеет bind-mount'ом, поэтому приложение
    # запускается от него: созданные файлы попадут к хостовому пользователю, и
    # хостовый сервер сможет их читать и писать.
    mkdir -p /data/voices /data/models
    warn_missing_models
    exec "$@"
fi

exec "$@"
