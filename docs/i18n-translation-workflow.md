# I18n Translation Workflow

This project uses a lightweight client-side i18n module loaded from [src/static/i18n/i18n.js](../src/static/i18n/i18n.js).

## Core Rule

Chinese-first UI development is supported.

You can:

1. Write normal Mandarin text in HTML first.
2. Later add i18n keys and translation files.

However, a string only becomes switchable if it has i18n wiring (`data-i18n*` or `i18n.t(...)`).

## Where Translations Live

- Chinese dictionary: [src/static/i18n/zh.json](../src/static/i18n/zh.json)
- English dictionary: [src/static/i18n/en.json](../src/static/i18n/en.json)
- Runtime engine: [src/static/i18n/i18n.js](../src/static/i18n/i18n.js)

## How Rendering Works

At page load, the app fetches both JSON files and applies keys to the DOM.

Supported bindings:

- `data-i18n="key"` -> sets `textContent`
- `data-i18n-html="key"` -> sets `innerHTML` (trusted content only)
- `data-i18n-placeholder="key"` -> sets `placeholder`
- `data-i18n-title="key"` -> sets `title`
- `data-i18n-aria-label="key"` -> sets `aria-label`
- `<html data-i18n-title="title">` -> sets document title
- `i18n.t("key")` -> use in JavaScript logic/messages

Fallback behavior in `i18n.t(...)` is:

1. current language key
2. Chinese key
3. key literal (if missing in both)

## Standard Workflow For New UI Text

1. Add Mandarin UI text in HTML (this is allowed as the first step).
2. Add a stable key namespace, for example:
   - `topbar.*`, `digest.*`, `settings.*`, `tutorial.*`, `js.*`
3. Add `data-i18n*` attributes to HTML or `i18n.t(...)` in JS.
4. Add the same key to both JSON files:
   - [src/static/i18n/zh.json](../src/static/i18n/zh.json)
   - [src/static/i18n/en.json](../src/static/i18n/en.json)
5. Toggle language in UI and verify both Chinese and English rendering.

## Examples

### Text node

```html
<button data-i18n="topbar.digest">定时摘要</button>
```

### Placeholder

```html
<input
  data-i18n-placeholder="command.keyword_placeholder"
  placeholder="留空则取最近邮件"
/>
```

### JS dynamic message

```javascript
showToast(i18n.t("js.digest_saved"));
```

### JS interpolation

```javascript
i18n.t("js.deep_selected_count", { n: 5, max: 100 });
```

## Rules For `data-i18n-html`

Use `data-i18n-html` only for trusted static content that intentionally contains markup (for example tutorial paragraphs with `<strong>`).

Do not use `data-i18n-html` with untrusted user input.

## Key Naming Conventions

- Use lowercase dot-separated keys.
- Prefix by feature area.
- Keep keys stable; avoid rename churn.
- Do not reuse one key for different meanings.

Good:

- `digest.btn_save`
- `settings.api_key_note`
- `js.digest_load_fail`

Avoid:

- `save1`, `msgA`, `temp_text`

## Definition Of Done For I18n Changes

A translation change is complete only when all are true:

1. UI nodes are wired (`data-i18n*` or `i18n.t(...)`).
2. Key exists in both JSON files.
3. Chinese and English both look correct in UI.
4. No broken placeholders (`{name}` style variables match code params).
5. No accidental use of `data-i18n-html` for unsafe content.

## Quick Checks (PowerShell)

Find i18n tags in templates:

```powershell
Select-String -Path src\templates\*.html -Pattern 'data-i18n'
```

Check if a key exists in both dictionaries:

```powershell
Select-String -Path src\static\i18n\zh.json -Pattern '"digest.btn_save"'
Select-String -Path src\static\i18n\en.json -Pattern '"digest.btn_save"'
```

## Notes For Login/Register Pages

Login and register pages also use the same i18n runtime via:

```html
<script src="/static/i18n/i18n.js"></script>
```

So the same rules apply to:

- [src/templates/login.html](../src/templates/login.html)
- [src/templates/register.html](../src/templates/register.html)
