# Подпись релизных инсталляторов

Инсталляторы (`.exe`/Inno Setup, `.dmg`, AppImage) собирает
`scripts/build_installer.py`. Подпись — **опциональна** и управляется
переменными окружения:

- переменная **задана** → соответствующий инсталлятор подписывается (macOS —
  ещё и нотаризуется);
- переменная **не задана (пустая)** → подпись пропускается, сборка **не
  падает**. Это «спящий» (dormant) режим: локально и в форках всё собирается
  без сертификатов, инсталлятор просто получается неподписанным.

Сертификаты и пароли **никогда не коммитятся** в репозиторий — только
GitHub-secrets (или переменные окружения локально).

## Куда попадают артефакты

При запуске `build-installers.yml` по тегу `v*` собранные инсталляторы и их
`.sha256` **прикладываются к GitHub Release** этого тега (release assets) — рядом
с wheel/sdist от `release.yml` и архивом portable-бандла от
`build-portable.yml`. При ручном запуске (`workflow_dispatch`) релиз не
трогается: файлы доступны только как workflow-артефакты
`installer-<os>-<arch>`.

## Переменные окружения

| Переменная | Платформа | Назначение |
|------------|-----------|------------|
| `WINDOWS_CERT_FILE` | Windows | файл сертификата подписи кода (PKCS#12: `.pfx`/`.p12`) |
| `WINDOWS_CERT_PASSWORD` | Windows | пароль к этому сертификату |
| `APPLE_CERT_P12` | macOS | сертификат **Developer ID Application** (PKCS#12: `.p12`) |
| `APPLE_CERT_PASSWORD` | macOS | пароль к `.p12` |
| `APPLE_ID` | macOS | Apple ID для нотаризации |
| `APPLE_TEAM_ID` | macOS | идентификатор команды (Team ID) |
| `APPLE_APP_PASSWORD` | macOS | app-specific password для нотаризации |

В CI (`build-installers.yml`) значения берутся из `${{ secrets.* }}` с теми же
именами. Если секрет не заведён, переменная пустая → соответствующая подпись
пропускается.

## Windows — Authenticode

Нужен сертификат **подписи кода** (Code Signing), выпущенный
удостоверяющим центром, — **OV** или **EV**:

- OV (Organization Validation) — обычный выпуск, файл `.pfx` с приватным
  ключом можно экспортировать и хранить в секрете;
- EV (Extended Validation) — «мгновенная» репутация для SmartScreen, но ключ
  обычно на аппаратном токене, который в CI не подключить. Для EV в облаке
  нужен сервис вроде **Azure Trusted Signing** — тогда схему подписи в
  `scripts/build_installer.py`/workflow адаптируют отдельно.

Что делает подпись: `signtool.exe sign` с доверенной меткой времени
(`/tr <tsa-url>`), алгоритм `sha256`. Неподписанный `.exe` запускается, но
SmartScreen показывает предупреждение.

Где взять: у любого крупного CA (DigiCert, Sectigo, GlobalSign, SSL.com) —
раздел «Code Signing».

## macOS — Developer ID + нотаризация

1. **Сертификат.** Нужно членство в **Apple Developer Program** (платное).
   В *Certificates, Identifiers & Profiles* создаётся сертификат
   **Developer ID Application**, экспортируется вместе с приватным ключом в
   `.p12` (это и есть `APPLE_CERT_P12`).
2. **Подпись.** `codesign --deep --force --options runtime` (hardened runtime —
   обязателен для нотаризации), затем проверка `codesign --verify`.
3. **Нотаризация.** `xcrun notarytool submit --apple-id "$APPLE_ID"
   --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD" --wait`.
4. **Stapling.** `xcrun stapler staple` — «пришивает» результат нотаризации к
   `.dmg`/приложению, чтобы macOS проверяла его офлайн.

`APPLE_APP_PASSWORD` — это **app-specific password** (создаётся на
[appleid.apple.com](https://appleid.apple.com) → *Sign-In and Security* →
*App-Specific Passwords*), а не пароль от Apple ID. `APPLE_TEAM_ID` виден в
*Membership Details*.

Без нотаризации `.dmg` открывается, но Gatekeeper предупреждает о
«неизвестном разработчике».

## Как завести секреты в GitHub

*Settings → Secrets and variables → Actions → New repository secret* — по одному
на каждое имя из таблицы выше. Секреты недоступны в прогонах из форков, поэтому
в PR из форков подпись предсказуемо отключается.

## Локальная сборка с подписью

```bash
export WINDOWS_CERT_FILE=/secure/code-signing.pfx
export WINDOWS_CERT_PASSWORD=...          # Windows

export APPLE_CERT_P12=/secure/developer-id.p12
export APPLE_CERT_PASSWORD=...
export APPLE_ID=dev@example.com
export APPLE_TEAM_ID=ABCDE12345
export APPLE_APP_PASSWORD=....            # macOS

python scripts/build_installer.py --target windows \
  --bundle-dir dist/audio-transcriber --out dist-installers
```

Файлы сертификатов храните вне репозитория и с правами `600`. Проверить
подпись: `signtool verify /pa <installer>.exe` (Windows),
`codesign -dv --verbose=4` и `spctl -a -vv` (macOS).

## Безопасность

- Держите отдельные сертификаты только для релизов; при утечке — отзывайте у CA.
- Не печатайте содержимое сертификатов и паролей в логах CI (GitHub маскирует
  известные секреты, но сгенерированные файлы маскировать нельзя).
- Пароли app-specific можно и нужно периодически перевыпускать.
