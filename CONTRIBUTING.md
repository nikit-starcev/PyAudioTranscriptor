# Участие в разработке

Спасибо за интерес к PyAudioTranscriptor. Ниже — как сообщать о проблемах,
готовить окружение и присылать изменения.

## Обратная связь

- **Ошибка:** [форма сообщения об ошибке](https://github.com/nikit-starcev/PyAudioTranscriptor/issues/new?template=bug_report.yml).
  Приложите версию, ОС, способ установки, шаги воспроизведения и очищенный лог
  `audio-transcriber doctor`.
- **Идея или улучшение:** [форма предложения функции](https://github.com/nikit-starcev/PyAudioTranscriptor/issues/new?template=feature_request.yml).
  Опишите сценарий и пользу, а не только желаемое решение.
- **Уязвимость:** не создавайте публичный issue. Используйте приватный канал
  из [SECURITY.md](SECURITY.md).

Перед созданием issue проверьте существующие issues и
[документацию по архитектуре](docs/architecture/README.md). Один issue — одна
проблема или одно предложение.

## Локальная разработка

### Требования

- Python 3.12–3.14;
- [uv](https://docs.astral.sh/uv/);
- Git. Для сборки веб-интерфейса дополнительно нужен Node.js и npm.

### Установка

```bash
git clone https://github.com/nikit-starcev/PyAudioTranscriptor.git
cd PyAudioTranscriptor
uv sync
```

`uv sync` создаёт `.venv`, устанавливает проект в редактируемом режиме,
зависимости и dev-группу (`pytest`, `ruff`, `mypy`).

Для работы с веб-интерфейсом и опциональными движками установите extras:

```bash
uv sync --extra web --extra gigaam --extra sherpa --extra slides
```

Проверка окружения и запуск:

```bash
uv run audio-transcriber doctor
uv run audio-transcriber --version
uv run audio-transcriber web
```

### Конфигурация

```bash
cp config.example.env config.env
chmod 600 config.env
```

Заполните только нужные значения. Не коммитьте `config.env`, `.env`,
`web-data/`, `web-data/secrets.json`, модели, результаты и загруженные файлы:
они содержат локальные настройки, персональные данные и секреты.

### Проверки

Перед отправкой изменений выполните те же проверки, что и CI:

```bash
uv lock --check
uv run --no-sync ruff check src tests
uv run --no-sync mypy
uv run --no-sync pytest
```

- После правок `pyproject.toml` обновите lock-файл командой `uv lock` и
  закоммитьте `uv.lock`.
- Интеграционные тесты медленные, требуют моделей и реальных аудиофайлов:
  `uv run --no-sync pytest -m integration`.
- CI проверяет Python 3.12, 3.13 и 3.14.

### Веб-интерфейс

Исходники SPA — в `webui/`, собранная статика коммитится в
`src/audio_transcriber/web/static/`. После правок в `webui/` пересоберите её:

```bash
cd webui
npm install
npm run lint
npm run build
```

## Стиль кода

- Соблюдайте существующие соглашения и структуру пакетов; не делайте
  несвязанных рефакторингов в одном pull request.
- Python: `ruff` (включая сортировку импортов) и `mypy` должны проходить без
  ошибок. Публичные функции и методы аннотируйте типами.
- Комментарии и docstring — по существу, для неочевидной логики и ограничений.
- Новое поведение и исправления сопровождайте тестами; тесты не должны
  зависеть от сети и реальных моделей, если это не `integration`-тест.
- Не добавляйте зависимости без необходимости. Новая зависимость требует
  обоснования в PR и обновления `pyproject.toml`/`uv.lock`.
- Пользовательские тексты, сообщения об ошибках и документация — на русском
  языке, в деловом тоне.

## Коммиты

Используется формат [Conventional Commits](https://www.conventionalcommits.org/ru/):

```text
<type>(<scope>): <краткое описание> (#N)
```

Типы: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`,
`chore`. Scope — затронутая подсистема (`web`, `webui`, `pipeline`,
`diarization`, `config` и т. п.). Номер issue указывайте в заголовке или теле.

Примеры:

```text
feat(web): #148 установка OCR-зависимостей слайдов из UI
fix(pipeline): сужение типа recognizer для mypy (#143)
docs: #149 инфраструктура обратной связи
```

Один коммит — одно логическое изменение. В теле объясните, что и почему
сделано, если это не очевидно из заголовка.

## Pull request

1. Ответвитесь от актуальной `main`:

   ```bash
   git switch main
   git pull
   git switch -c feat/N-краткое-описание
   ```

2. Внесите изменения, добавьте тесты и выполните проверки из раздела
   «Проверки».
3. Откройте pull request в `main` и заполните шаблон. Свяжите PR с issue
   (`Closes #N`), опишите изменение и способ проверки.
4. Держите PR сфокусированным. Черновик (`Draft`) используйте для раннего
   обсуждения. Учтите замечания ревью и дождитесь зелёного CI.

Прямые коммиты в `main` не используются: изменения вливаются через pull
request.

## CHANGELOG и TODO

- **`CHANGELOG.md`** ведётся вручную по [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).
  Пользовательские изменения добавляйте в `[Unreleased]` в раздел `Added`,
  `Changed`, `Fixed` или `Removed`. Внутренние изменения без влияния на
  пользователя можно не указывать.
- **`TODO.md`** — список открытых задач со ссылками на issues. Если работа
  закрывает пункт, отметьте его выполненным и синхронизируйте с issue.
- При изменении поведения, CLI, конфигурации или API обновите `README.md` и
  соответствующие файлы в `docs/architecture/`.
