# Выпуск релиза

Версия пакета не хранится в `pyproject.toml` — её выводит
[`hatch-vcs`](https://github.com/ofek/hatch-vcs) из git-тега через
`setuptools-scm`. Единственный источник истины — тег вида `vX.Y.Z`.

## Как это работает

| Состояние репозитория | Версия пакета |
| --- | --- |
| Ровно на теге `v0.3.1` | `0.3.1` |
| 17 коммитов после `v0.3.1` | `0.3.2.dev17+g3f5f115` |
| Грязное дерево (есть правки) | `…dev17+g3f5f115.d20261002` |

- `X.Y.Z` — версия последнего тега; `devN` — число коммитов после него;
- `+gHASH` — короткий хеш коммита, `.dYYYYMMDD` — дата сборки грязного дерева.

Проверить текущую версию без установки:

```bash
.venv/bin/python -c "import audio_transcriber; print(audio_transcriber.__version__)"
```

Версия в колесе/метаданных:

```bash
uv build
# dist/audio_transcriber-X.Y.Z...whl
```

## Как выпустить релиз

1. Убедиться, что все изменения влиты в `main` и CI зелёный.

2. Обновить `CHANGELOG.md` (ведётся вручную): перенести нужные пункты из
   `[Unreleased]` в новый раздел с версией и датой, например:

   ```markdown
   ## [0.4.0] - 2026-11-01
   ```

   Формат заголовка важен: workflow ищет ровно `## [<версия>]` (часть с датой
   необязательна), чтобы собрать заметки релиза.

3. Закоммитить changelog (и любые правки кода) в `main`:

   ```bash
   git add CHANGELOG.md
   git commit -m "chore(release): 0.4.0"
   ```

4. Создать и запушить аннотированный тег (префикс `v` обязателен):

   ```bash
   git tag -a v0.4.0 -m "v0.4.0"
   git push origin v0.4.0
   ```

   Тег должен указывать ровно на тот коммит, где версия в `CHANGELOG.md`
   совпадает с тегом, иначе шаг проверки версии в release-workflow упадёт.

5. Дождаться workflow **Release** (`.github/workflows/release.yml`). Он:
   - прогоняет `ruff`, `mypy`, `pytest`;
   - собирает sdist и wheel через `uv build`;
   - проверяет, что версия артефакта совпадает с тегом;
   - берёт раздел `<версия>` из `CHANGELOG.md`;
   - создаёт GitHub Release и прикладывает артефакты.

## Публикация в PyPI

Намеренно не настроена. Чтобы включить, раскомментируйте шаг `uv publish` в
`.github/workflows/release.yml` и добавьте секрет `PYPI_API_TOKEN`.

## CI

`.github/workflows/ci.yml` запускается на push и pull request: `pytest`,
`ruff check src tests`, `mypy` (Linux, `uv`, кэш). Оба workflow делают
checkout с `fetch-depth: 0`, иначе `hatch-vcs` не увидит теги и подставит
dev-версию.
