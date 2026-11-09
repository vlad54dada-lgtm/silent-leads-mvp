# CLAUDE.md — правила розробки LeadFlow

> Що будуємо — у [PRODUCT.md](PRODUCT.md). Тут — **як** будувати. Читай обидва файли на старті сесії.

## Мова

- Спілкування з користувачем і тексти інтерфейсу — **українською**.
- Код, назви змінних, таблиць, комітів — **англійською**.

## Стек

| Шар | Технологія |
|---|---|
| Фреймворк | Next.js (App Router) + TypeScript (`strict`) |
| UI | Tailwind CSS + shadcn/ui |
| БД / Auth | Supabase (Postgres, Row Level Security, Supabase Auth) |
| Валідація | Zod — на кожній межі (webhook, форми, відповіді AI) |
| AI | Anthropic SDK (`@anthropic-ai/sdk`), модель з env `ANTHROPIC_MODEL` (за замовч. `claude-sonnet-5-5`) |
| Сповіщення | Telegram Bot API (просто `fetch`, без зайвих бібліотек) |
| Тести | Vitest (unit), Playwright (e2e ключових сценаріїв) |
| Деплой | Vercel |
| Пакетний менеджер | npm, Node 24 |

Нові залежності — лише якщо без них не обійтися; пояснюй, навіщо.

## Структура проєкту (цільова)

```
src/
  app/                    # маршрути Next.js
    (dashboard)/leads/    # список і картка ліда
    api/leads/intake/     # webhook прийому лідів
  lib/
    leads/                # доменна логіка: normalize, dedupe, statuses
    ai/                   # скоринг через Claude + Zod-схема відповіді
    notify/               # Telegram
    supabase/             # клієнти: server, browser, admin (service role)
  components/             # UI-компоненти
supabase/migrations/      # SQL-міграції (єдине джерело правди для схеми)
tests/                    # e2e (Playwright)
```

Бізнес-логіка живе в `src/lib/`, а не в компонентах чи route handlers — так її легко тестувати.

## Команди

> Проєкт ще не ініціалізовано — команди стануть доступні після scaffold.

```bash
npm run dev          # локальний сервер
npm run lint         # ESLint
npm run typecheck    # tsc --noEmit
npm test             # Vitest
npm run test:e2e     # Playwright
npx supabase migration new <name>   # нова міграція
```

## Правила — ЗАВЖДИ

- **Спочатку зберегти ліда, потім усе інше.** AI-скоринг і Telegram виконуються після запису
  в БД; їх збій позначається (`ai_status = failed`) і не ламає прийом заявки.
- **RLS увімкнено на кожній таблиці**, з явними політиками. Нова таблиця без RLS = баг.
- **Валідуй вхідні дані Zod-схемою** — webhook, форми, CSV, і відповідь Claude (AI може повернути невалідний JSON).
- **Webhook захищений** секретом у заголовку (`x-intake-secret`) і ідемпотентний
  (повторна та сама заявка не створює дубль).
- **Нормалізуй** email (lowercase, trim) і телефон (E.164) перед дедуплікацією.
- **Кожна зміна ліда пишеться в `lead_events`** (статус, нотатка, AI, сповіщення).
- Схема БД змінюється **лише через міграції** в `supabase/migrations/`.
- Перед комітом: `lint` + `typecheck` + `test` мають бути зеленими.
- Для нової логіки в `src/lib/` — unit-тест (нормалізація, дедуплікація, парсинг відповіді AI — обов'язково).
- Маленькі коміти у форматі Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`.
- Робота — у feature-гілках (`feat/lead-intake`), не напряму в `main`.

## Правила — НІКОЛИ

- **Ніколи не коміть секрети.** Ключі — тільки в `.env.local` (в `.gitignore`); у репо — `.env.example` без значень.
- **Ніколи не використовуй `SUPABASE_SERVICE_ROLE_KEY` на клієнті** — лише в серверному коді.
- Ніколи не логуй персональні дані лідів (email, телефон, повідомлення) у відкритому вигляді.
- Ніколи не викликай Claude синхронно в запиті webhook так, щоб це затримувало відповідь сайту.
- Ніколи не використовуй `any` і не вимикай правила лінтера, щоб «просто запрацювало».
- Не розширюй скоуп за межі PRODUCT.md (розділ «Won't have») без погодження з користувачем.

## Змінні середовища

```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5-5
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
INTAKE_WEBHOOK_SECRET=
NEXT_PUBLIC_APP_URL=
```

## Definition of Done (для кожної задачі)

1. Функція працює за сценарієм з PRODUCT.md.
2. Lint, typecheck, тести — зелені.
3. Нові таблиці мають RLS, нові вхідні дані — Zod-валідацію.
4. Немає секретів і персональних даних у коді/логах.
5. Якщо змінилась поведінка або команди — оновлено PRODUCT.md / CLAUDE.md.
