# AGENTS: I18n Translation Rules

This file defines mandatory behavior for coding agents (Cursor, Claude Code, Codex, Copilot Chat, and similar tools) when editing UI text.

## Scope

Apply these rules when changing:

- HTML templates in [src/templates/index.html](src/templates/index.html), [src/templates/login.html](src/templates/login.html), [src/templates/register.html](src/templates/register.html)
- Client JS strings in inline scripts
- Translation dictionaries in [src/static/i18n/zh.json](src/static/i18n/zh.json) and [src/static/i18n/en.json](src/static/i18n/en.json)

## Required Workflow

When adding or changing user-facing UI text, agents MUST do all steps in the same change:

1. Add or update the UI node with `data-i18n*` or `i18n.t(...)`.
2. Add/update the key in both translation files.
3. Keep Chinese fallback text in HTML where existing style expects it.
4. Preserve key namespace style (`topbar.*`, `digest.*`, `settings.*`, `tutorial.*`, `js.*`).
5. Avoid renaming existing keys unless explicitly requested.

## Binding Rules

Use the correct binding type:

- Visible text: `data-i18n`
- Placeholder: `data-i18n-placeholder`
- Title tooltip: `data-i18n-title`
- Accessibility label: `data-i18n-aria-label`
- Markup-rich trusted text only: `data-i18n-html`
- JS runtime text: `i18n.t("...")`

Never use `data-i18n-html` for untrusted or user-generated content.

## Missing Key Policy

If a referenced key is missing:

1. Add it to [src/static/i18n/zh.json](src/static/i18n/zh.json)
2. Add it to [src/static/i18n/en.json](src/static/i18n/en.json)
3. Ensure placeholders match exactly across languages (`{n}`, `{max}`, etc.)

Do not leave key literals visible in UI.

## New Page Policy

If a new HTML page is added and should support language switch:

1. Include `<script src="/static/i18n/i18n.js"></script>`
2. Add `data-i18n-title` on `<html>` when page title is translatable
3. Use `data-i18n*` attributes for all user-facing static text

## Completion Checklist (Agent Must Verify)

Before finishing, confirm:

1. Every changed UI string is wired for i18n.
2. New keys exist in both JSON files.
3. No obvious layout break from longer English text.
4. No unsafe `data-i18n-html` usage.

For human-readable implementation details, see:

- [docs/i18n-translation-workflow.md](docs/i18n-translation-workflow.md)
