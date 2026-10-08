# Дизайн-система AudioTranscriber

Единый набор токенов и компонентов для всего веб-интерфейса. Всё лежит в
`webui/src/components/ui/` и реэкспортируется из `webui/src/components/ui/index.ts`.

```ts
import { Button, Card, CardHeader, CardContent, Badge, Input } from '../ui'
```

Все компоненты — TypeScript, пропсы экспортируются, стили — только Tailwind
v4-утилиты от семантических токенов (см. `webui/src/index.css`). Никаких
хардкод-цветов (`slate-*`, `blue-*`) в новом коде быть не должно.

## Токены

Определены в `webui/src/index.css`:

- **Семантические цвета** (light + dark, переключаются классом `.dark` на `<html>`):
  `bg`, `surface`, `surface-2`, `surface-3`, `border`, `border-strong`, `text`,
  `muted`, `primary`, `primary-hover`, `primary-fg`, `primary-soft`,
  `primary-soft-fg`, `ring`, `success`, `warn`, `danger`, `info`, `overlay`.
  Для мягких плашек есть пары `*-soft` (фон) и `*-soft-fg` (текст):
  `success-soft`/`success-soft-fg`, `warn-soft`, `danger-soft`, `info-soft`,
  `neutral-soft`, `primary-soft`.
  Использование: `bg-surface`, `text-muted`, `border-border`, `bg-primary`,
  `text-primary-fg`, `bg-success-soft text-success-soft-fg`.
- **Отступы** — базовая единица 4px (`--spacing: 0.25rem`), шкала
  4/8/12/16/20/24/32 через числовые утилиты: 4px=`*-1`, 8px=`*-2`, 12px=`*-3`,
  16px=`*-4`, 20px=`*-5`, 24px=`*-6`, 32px=`*-8` (например `p-4`, `gap-2`,
  `space-y-6`, `mt-8`). Именованных `--spacing-sm/md/lg/xl/2xl` намеренно нет:
  они конфликтуют с размерной шкалой `max-w-sm/md/lg/xl/2xl`.
- **Радиусы**: `rounded-sm` (6px), `rounded-md` (8px), `rounded-lg` (12px),
  `rounded-xl` (16px).
- **Тени**: `shadow-sm`, `shadow-md`, `shadow-lg`.
- **Шрифт**: `text-xs` (12), `text-sm` (14), `text-base` (16), `text-lg` (18),
  `text-xl` (20), `text-2xl` (24). `font-sans`, `font-mono`.

### Тёмная тема

Класс `.dark` на `<html>` ставит `theme.ts` (`useThemeMode`). Не использовать
`dark:`-варианты с палитрой — токены уже меняются автоматически. Компоненты,
использующие хардкод-цвета со `dark:` (старые панели), продолжают работать.

## Компоненты

### Button

```tsx
<Button variant="primary" onClick={run}>Запустить</Button>
<Button variant="secondary" size="sm" icon={<Play className="h-4 w-4" />}>Играть</Button>
<Button variant="danger" loading>Удаляю…</Button>
<Button variant="ghost" fullWidth>Отмена</Button>
```

Пропсы (наследует `ButtonHTMLAttributes`): `variant?: 'primary' | 'secondary' | 'ghost' | 'danger'` (default `secondary`),
`size?: 'sm' | 'md'` (default `md`), `loading?: boolean`, `icon?: ReactNode`,
`iconRight?: ReactNode`, `fullWidth?: boolean`, `iconOnly?: boolean`,
`aria-label?`. При `loading` кнопка автоматически `disabled` и показывает спиннер.

### IconButton

Обязателен `aria-label` (тип требует строку). Рендерит `Button` с `iconOnly`.

```tsx
<IconButton aria-label="Обновить" onClick={refresh}><RefreshCw className="h-4 w-4" /></IconButton>
```

Пропсы: `aria-label: string` (обязателен), `children: ReactNode` (иконка),
`variant?`, `size?`, `loading?`.

### Input / Textarea / Select

Наследуют нативные пропсы; общий пропс `invalid?: boolean` (подсветка + `aria-invalid`).

```tsx
<Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Поиск" invalid={!!err} />
<Textarea rows={6} value={text} onChange={(e) => setText(e.target.value)} />
<Select value={fmt} onChange={(e) => setFmt(e.target.value)}>
  <option value="txt">TXT</option>
</Select>
```

### Field

Обёртка «подпись + контрол + подсказка/ошибка». Использовать для доступных форм.

```tsx
<Field label="Название" htmlFor="name" hint="Видно всем" error={fieldError} required>
  <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
</Field>
```

Пропсы: `label: ReactNode`, `htmlFor?: string`, `hint?: ReactNode`,
`error?: ReactNode`, `required?: boolean`, `children: ReactNode`, `className?`.

### Checkbox / Switch

```tsx
<Checkbox label="Показать удалённые" checked={show} onChange={(e) => setShow(e.target.checked)} />
<Switch checked={on} onChange={setOn} label="LLM-постобработка" aria-label="LLM-постобработка" />
```

`Checkbox` — нативный input с опциональной подписью (`label`); если `label`
задан без `id`, клик по подписи всё равно переключает (label-обёртка).
`Switch` — `role="switch"`, пропсы `checked`, `onChange(checked)`, `label?`,
`disabled?`, `id?`, `aria-label?`.

### Card (+ Header/Content/Footer/Title)

```tsx
<Card>
  <CardHeader
    title="Задачи"
    description="Все запуски"
    actions={<Button size="sm" onClick={refresh}>Обновить</Button>}
  />
  <CardContent>…</CardContent>
  <CardFooter><Button variant="ghost">Отмена</Button></CardFooter>
</Card>
```

`Card`: `padded?: boolean`, остальное — `HTMLAttributes<HTMLDivElement>`.
`CardHeader`: `title`, `description?`, `actions?`, `className?`, `id?`
(заголовок — `<h2>`; `id` для `aria-labelledby`/якорей).
`CardTitle` — отдельный `<h2>` для произвольной разметки.

### Badge

```tsx
<Badge tone="success">Готово</Badge>
<Badge tone="running" />{/* нет */}
<Badge tone="warn" icon={<Clock className="h-3 w-3" />}>В очереди</Badge>
```

Пропсы: `tone?: 'neutral' | 'success' | 'warn' | 'danger' | 'info' | 'primary'`
(default `neutral`), `icon?`, `children`, `className?`.

### Tooltip

Только визуальная подсказка (a11y-имя всё равно задаётся через `aria-label`).

```tsx
<Tooltip label="Скачать результат">
  <IconButton aria-label="Скачать"><Download className="h-4 w-4" /></IconButton>
</Tooltip>
```

Пропсы: `label: string`, `side?: 'top' | 'bottom'`, `children`, `className?`.

### Modal

Управляемый (`open` + `onClose`): Escape, focus-trap, возврат фокуса, клик по
оверлею, `overscroll-behavior: contain`, кнопка закрытия, блокировка скролла body.

```tsx
<Modal open={open} onClose={close} title="Настройки" size="lg"
  footer={<><Button variant="ghost" onClick={close}>Отмена</Button><Button variant="primary" onClick={save}>Сохранить</Button></>}>
  …тело…
</Modal>
```

Пропсы: `open`, `onClose`, `title?`, `description?`, `children`, `footer?`,
`size?: 'sm' | 'md' | 'lg' | 'xl'` (default `md`), `aria-label?`
(если нет `title`, задайте `aria-label`).

### Tabs / TabPanel

Управляемые вкладки с roving tabindex и стрелками ←/→/Home/End.

```tsx
const [tab, setTab] = useState('transcript')
<Tabs aria-label="Секции задачи" value={tab} onChange={setTab}
  items={[{ id: 'transcript', label: 'Стенограмма' }, { id: 'chat', label: 'Чат' }]} />
<TabPanel value="transcript" active={tab}>…</TabPanel>
```

`Tabs`: `items: TabItem[]` (`{ id, label, icon?, disabled? }`), `value`,
`onChange(id)`, `aria-label` (обязателен), `className?`.
`TabPanel`: `value`, `active`, `children`, `className?`.

### EmptyState

```tsx
<EmptyState icon={<Inbox className="h-8 w-8" />} title="Задач пока нет"
  description="Загрузите файл и поставьте его в очередь"
  action={<Button variant="primary">Загрузить файл</Button>} />
```

Пропсы: `icon?`, `title`, `description?`, `action?`, `className?`.

### ProgressBar

```tsx
<ProgressBar value={42} max={100} label="Прогресс задачи" tone="primary" />
<ProgressBar indeterminate label="Загрузка" />
```

Пропсы: `value?`, `max?` (default 100), `tone?: 'primary' | 'success' | 'warn' | 'danger'`,
`indeterminate?`, `size?: 'sm' | 'md'`, `label?`, `className?`.

### Spinner

```tsx
<Spinner label="Загрузка" />
<Spinner size={20} />
```

Пропсы: `size?: number` (default 16), `label?` (если задан — `role="status"` + `sr-only`).

### Alert

Инлайн-сообщение. Для асинхронных/меняющихся сообщений ставьте `live` —
добавит `aria-live="polite"` и `role="status"`.

```tsx
<Alert tone="danger" title="Ошибка" live onDismiss={() => setError(null)}>{error}</Alert>
<Alert tone="success">Задача запущена</Alert>
```

Пропсы: `tone?: 'info' | 'success' | 'warn' | 'danger'` (default `info`),
`title?`, `children?`, `onDismiss?`, `dismissLabel?`, `live?: boolean`,
`className?`.

## Правила a11y

- Иконочные кнопки — всегда `IconButton` с `aria-label`.
- Поля — через `Field` (`label` + `htmlFor` + `id`).
- Интерактив — `<button>`/`<a>`/`<label>`, не `div` с `onClick`.
- Один `<h1>` на страницу (в topbar — бренд), заголовки секций — `<h2>`.
- Длинные строки: `truncate` + `min-w-0` (и `title`).
- Таймкоды/числа: класс `tabular-nums`.
- Деструктивные действия — подтверждение (см. `window.confirm` в контроллере).
- `focus-visible` ring уже задан глобально и в компонентах.
- При `prefers-reduced-motion` анимации отключаются глобально.
