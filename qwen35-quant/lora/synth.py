"""Synthetic ipsupport-code conversations for the actions real traces barely
have (git, calc, help, done, agent, web.stackexchange, file.find / search /
append / read windows, run with cwd / timeout). Same format, system prompt
and tool schemas as build_dataset.py; Russian and English goals; every
final answer is short and ends with the prompt's "NEXT:" line.

    python synth.py --request captured_request.json --out synth.jsonl [--per-scenario 12]
"""

from __future__ import annotations

import argparse
import json
import random

from build_dataset import system_prompt

R = random.Random(0)
PROJECTS = ["api-server", "notecli", "quotes", "billing", "tg-bot", "scraper", "dashboard", "infra"]
LANGS = [("go", "main.go", "go test ./..."), ("python", "app.py", "pytest -q"), ("node", "index.js", "npm test"),
         ("rust", "src/main.rs", "cargo test")]
FILES = ["README.md", "config.yaml", "main.go", "app.py", "src/utils.py", "internal/db/db.go", "index.js", "Makefile"]
AUTHORS = ["Roman", "Alex", "Dana"]
DATES = ["2026-10-0%d" % d for d in range(1, 8)]


def pick(xs):
    return R.choice(xs)


def call(tool, action, **params):
    return (tool, action, params)


class Conv:
    def __init__(self, goal: str):
        self.goal = goal
        self.steps = []      # [(calls, observations, text)]
        self.final = None
        self.done = False

    def step(self, calls, observations, text=""):
        self.steps.append((calls if isinstance(calls, list) else [calls],
                           observations if isinstance(observations, list) else [observations], text))
        return self

    def end(self, text, next_step=None, done=False):
        self.final = text + (f"\nNEXT: {next_step}" if next_step else "")
        self.done = done
        return self

    def render(self, template, tools, date):
        messages = [{"role": "system", "content": system_prompt(template, date)},
                    {"role": "user", "content": self.goal}]
        n = 0
        for calls, observations, text in self.steps:
            tcs = []
            for tool, action, params in calls:
                n += 1
                tcs.append({"id": f"call_{n}", "type": "function", "function": {
                    "name": tool, "arguments": json.dumps({"action": action, "params": params}, ensure_ascii=False)}})
            messages.append({"role": "assistant", "content": text, "tool_calls": tcs})
            for tc, obs in zip(tcs, observations):
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
        if self.done:
            n += 1
            messages.append({"role": "assistant", "content": self.final, "tool_calls": [{
                "id": f"call_{n}", "type": "function",
                "function": {"name": "done", "arguments": json.dumps({"action": "done", "params": {}})}}]})
        else:
            messages.append({"role": "assistant", "content": self.final})
        return {"messages": messages, "tools": tools}


LANG = {"ru": True}


def ru():
    """The conversation's language, drawn once per conversation (main)."""
    return LANG["ru"]


# ---------------------------------------------------------------- git

def git_status():
    f1, f2 = R.sample(FILES, 2)
    goal = pick(["что у нас в гите?", "покажи статус репо", "есть незакоммиченные изменения?"]) if ru() \
        else pick(["what's the git status?", "any uncommitted changes?", "show me the repo status"])
    obs = f"On branch main\nChanges not staged for commit:\n\tmodified:   {f1}\n\nUntracked files:\n\t{f2}"
    c = Conv(goal).step(call("git", "status"), obs)
    return c.end(f"На main: изменён `{f1}`, не отслеживается `{f2}`." if goal[0] in "чпе" else
                 f"On main: `{f1}` modified, `{f2}` untracked.", "посмотреть diff" if goal[0] in "чпе" else "review the diff")


def git_diff():
    f = pick(FILES)
    staged = R.random() < 0.3
    goal = (f"что поменялось в {f}?" if ru() else f"what changed in {f}?") + (" (в индексе)" if staged else "")
    obs = (f"diff --git a/{f} b/{f}\n--- a/{f}\n+++ b/{f}\n@@ -10,7 +10,7 @@\n-timeout = 30\n+timeout = 60\n"
           " retries = 3")
    params = {"path": f}
    if staged:
        params["staged"] = True
    c = Conv(goal).step(call("git", "diff", **params), obs)
    return c.end(f"В `{f}` таймаут поднят с 30 до 60." if ru() else f"`{f}`: the timeout went from 30 to 60.",
                 None)


def git_log():
    n = pick([3, 5, 10])
    goal = pick([f"последние {n} коммитов?", "что коммитили в последнее время?"]) if ru() else \
        pick([f"show the last {n} commits", "what's been committed lately?"])
    lines = [f"{R.getrandbits(28):07x} {pick(['fix', 'add', 'refactor', 'docs'])}: {pick(['retry on 429', 'config loader', 'db pool', 'README'])} ({pick(AUTHORS)}, {pick(DATES)})" for _ in range(n)]
    c = Conv(goal).step(call("git", "log", n=n), "\n".join(lines))
    return c.end(("Последние коммиты:\n" if ru() else "Latest commits:\n") + "\n".join("- " + l.split(" ", 1)[1] for l in lines[:3]), None)


def git_show():
    ref = pick(["HEAD", "HEAD~1", "v1.2.0"])
    goal = f"что в коммите {ref}?" if ru() else f"what's in {ref}?"
    obs = f"commit {R.getrandbits(40):010x}\nAuthor: {pick(AUTHORS)}\nDate:   {pick(DATES)}\n\n    fix: handle empty config\n\n config.yaml | 2 +-\n 1 file changed"
    c = Conv(goal).step(call("git", "show", ref=ref), obs)
    return c.end("Коммит чинит пустой конфиг: одна строка в `config.yaml`." if ru() else
                 "It fixes an empty config: one line in `config.yaml`.", None)


def git_commit():
    f = pick(FILES)
    msg = pick(["fix: retry on timeout", "docs: update README", "feat: add config flag"])
    goal = f"закоммить {f} с сообщением '{msg}'" if ru() else f"commit {f} as '{msg}'"
    c = Conv(goal).step(call("git", "add", paths=f), "ok").step(call("git", "commit", message=msg),
                                                                 f"[main {R.getrandbits(28):07x}] {msg}\n 1 file changed, 4 insertions(+), 1 deletion(-)")
    return c.end(f"Закоммитил `{f}`: «{msg}»." if ru() else f"Committed `{f}`: \"{msg}\".", "запушить" if ru() else "push it")


def git_branch():
    name = pick(["feature/retry", "fix/config", "exp/cache"])
    if R.random() < 0.5:
        goal = "какие ветки есть?" if ru() else "list the branches"
        c = Conv(goal).step(call("git", "branch"), f"* main\n  {name}\n  release/1.2")
        return c.end(f"Ветки: main (текущая), {name}, release/1.2." if ru() else f"Branches: main (current), {name}, release/1.2.", None)
    goal = f"создай ветку {name} и переключись на неё" if ru() else f"create a branch {name} and switch to it"
    c = Conv(goal).step(call("git", "branch", name=name), f"created branch {name}").step(
        call("git", "checkout", ref=name), f"Switched to branch '{name}'")
    return c.end(f"Ветка `{name}` создана, я на ней." if ru() else f"Created `{name}` and switched to it.", None)


def git_clone():
    proj = pick(PROJECTS)
    url = f"https://github.com/example/{proj}.git"
    goal = f"склонируй {url}" if ru() else f"clone {url}"
    c = Conv(goal).step(call("git", "clone", url=url, dir=proj), f"Cloning into '{proj}'...\ndone.").step(
        call("file", "list", path=proj), "README.md\ngo.mod\nmain.go\ninternal/")
    return c.end(f"Склонировал в `{proj}/`: Go-проект (go.mod, main.go, internal/)." if ru() else
                 f"Cloned into `{proj}/`: a Go project (go.mod, main.go, internal/).", "собрать проект" if ru() else "build it")


def git_sync():
    action = pick(["pull", "fetch", "push"])
    goal = {"pull": ("подтяни изменения с origin", "pull from origin"), "fetch": ("сделай fetch", "fetch origin"),
            "push": ("запушь текущую ветку", "push the current branch")}[action][0 if ru() else 1]
    obs = {"pull": "Updating 1a2b3c4..5d6e7f8\nFast-forward\n main.go | 12 ++++++++----", "fetch": "From github.com:example/app\n   1a2b3c4..5d6e7f8  main -> origin/main",
           "push": "To github.com:example/app.git\n   1a2b3c4..5d6e7f8  main -> main"}[action]
    params = {"remote": "origin"} if action != "push" else {"remote": "origin", "branch": "main"}
    c = Conv(goal).step(call("git", action, **params), obs)
    return c.end({"pull": "Подтянул: main.go обновлён (fast-forward).", "fetch": "origin/main обновлена до 5d6e7f8.",
                  "push": "main запушена в origin."}[action] if ru() else
                 {"pull": "Pulled: main.go updated (fast-forward).", "fetch": "origin/main is now at 5d6e7f8.",
                  "push": "Pushed main to origin."}[action], None)


def git_init_remote():
    url = f"git@github.com:example/{pick(PROJECTS)}.git"
    goal = f"сделай тут репозиторий и добавь remote {url}" if ru() else f"init a repo here and add the remote {url}"
    c = Conv(goal).step(call("git", "init"), "Initialized empty Git repository in .git/").step(
        call("git", "remote", name="origin", url=url), "ok").step(call("git", "remote"), f"origin\t{url} (fetch)\norigin\t{url} (push)")
    return c.end(f"Репозиторий создан, origin → {url}." if ru() else f"Repo initialized, origin → {url}.", "первый коммит" if ru() else "first commit")


# ---------------------------------------------------------------- calc / help / done / agent / web

def calc():
    a, b = R.randint(12, 999), R.randint(2, 97)
    items = [
        (f"сколько будет {a} * {b} / 7?", f"what's {a} * {b} / 7?", f"{a} * {b} / 7", round(a * b / 7, 4)),
        (f"квадратный корень из {a}?", f"square root of {a}?", f"sqrt({a})", round(a ** 0.5, 6)),
        (f"{b}% от {a * 10}?", f"what's {b}% of {a * 10}?", f"{a * 10} * {b} / 100", round(a * 10 * b / 100, 4)),
        (f"сколько секунд в {b} сутках?", f"how many seconds in {b} days?", f"{b} * 24 * 3600", b * 86400),
    ]
    g_ru, g_en, expr, val = pick(items)
    goal = g_ru if ru() else g_en
    c = Conv(goal).step(call("calc", "calculate", expression=expr), str(val))
    return c.end(f"{expr} = {val}", None)


def help_lessons():
    domain = pick(["git", "run", "file", "web"])
    bad = {"git": ("git", "commit", {"msg": "fix"}, "missing required param(s): message — git.commit needs {\"message\": str}"),
           "run": ("run", "shell", {"cmd": "make"}, "missing required param(s): command — run.shell needs {\"command\": str}"),
           "file": ("file", "edit", {"file": "a.py", "find": "x", "replace": "y"}, "missing required param(s): path — file.edit needs {\"path\": str, ...}"),
           "web": ("web", "fetch", {"link": "https://example.com"}, "missing required param(s): url — web.fetch needs {\"url\": str}")}[domain]
    good = {"git": {"message": "fix"}, "run": {"command": "make"}, "file": {"path": "a.py", "find": "x", "replace": "y"},
            "web": {"url": "https://example.com"}}[domain]
    goal = {"git": "закоммить с сообщением fix", "run": "запусти make", "file": "в a.py замени x на y",
            "web": "открой https://example.com"}[domain] if ru() else \
        {"git": "commit with message fix", "run": "run make", "file": "in a.py replace x with y", "web": "open https://example.com"}[domain]
    lessons = f"{domain} usage:\n" + {"git": "commit: {\"message\": str}", "run": "shell: {\"command\": str, \"cwd\"?: str}",
                                      "file": "edit: {\"path\": str, \"find\"?: str, \"replace\"?: str}", "web": "fetch: {\"url\": str}"}[domain] + \
        "\nlessons:\n- use the exact param names above"
    c = Conv(goal).step(call(*bad[:2], **bad[2]), bad[3]).step(call("help", "lessons", domain=domain), lessons).step(
        call(bad[0], bad[1], **good), "ok")
    return c.end("Готово." if ru() else "Done.", None)


def done_tool():
    f = pick(["hello.sh", "notes.md", "todo.txt"])
    goal = f"создай пустой {f}" if ru() else f"create an empty {f}"
    c = Conv(goal).step(call("file", "write", path=f), f"wrote {f} (0 bytes)")
    return c.end(f"Создал пустой `{f}`." if ru() else f"Created an empty `{f}`.", None, done=True)


def agent_run():
    profile = pick(["claude", "codex"])
    proj = pick(PROJECTS)
    goal = f"попроси {profile} сделать ревью ~/gh/{proj}" if ru() else f"have {profile} review ~/gh/{proj}"
    task = (f"Review the code in ~/gh/{proj} for bugs: concurrency, error handling, resource leaks. "
            "Report each finding with file:line and why. Don't change any files.")
    obs = f"{profile} finished.\nFindings:\n- internal/db/db.go:42 rows not closed on error\n- main.go:88 context ignored in retry loop"
    c = Conv(goal).step(call("agent", "run", profile=profile, task=task, dir=f"~/gh/{proj}"), obs)
    return c.end(("Ревью от " if ru() else "Review from ") + profile + ":\n- db.go:42 rows not closed on error\n- main.go:88 context ignored in retry",
                 "починить db.go:42" if ru() else "fix db.go:42")


def web_stackexchange():
    err = pick([("go: cannot find main module", "go"), ("ModuleNotFoundError: No module named 'yaml'", "python"),
                ("error[E0502]: cannot borrow as mutable", "rust")])
    goal = f"что значит ошибка «{err[0]}»?" if ru() else f"what does \"{err[0]}\" mean?"
    obs = f"1. [answered, 124 votes] {err[0]} — accepted answer: " + {"go": "run `go mod init <name>` in the project root",
                                                                   "python": "install it: `pip install pyyaml`",
                                                                   "rust": "end the first borrow before taking a mutable one"}[err[1]]
    c = Conv(goal).step(call("web", "stackexchange", query=err[0], tag=err[1]), obs)
    return c.end(obs.split("accepted answer: ")[1].capitalize() + ".", None)


# ---------------------------------------------------------------- file / run variants

def file_find():
    pat = pick(["**/*.go", "**/*_test.py", "**/*.md", "**/Dockerfile"])
    goal = f"найди все файлы {pat}" if ru() else f"find all {pat} files"
    hits = {"**/*.go": "main.go\ninternal/db/db.go\ninternal/api/handler.go", "**/*_test.py": "tests/test_api.py\ntests/test_db.py",
            "**/*.md": "README.md\ndocs/setup.md", "**/Dockerfile": "Dockerfile\ndeploy/worker/Dockerfile"}[pat]
    c = Conv(goal).step(call("file", "find", pattern=pat), hits)
    return c.end(("Нашёл:\n" if ru() else "Found:\n") + "\n".join("- " + h for h in hits.split("\n")), None)


def file_search():
    q = pick(["TODO", "func main", "password", "timeout"])
    goal = f"где в коде встречается {q}?" if ru() else f"where does the code mention {q}?"
    hits = f"main.go:12: // {q} check\ninternal/db/db.go:40: {q}"
    c = Conv(goal).step(call("file", "search", query=q), hits)
    return c.end(f"`{q}` есть в main.go:12 и internal/db/db.go:40." if ru() else f"`{q}` is in main.go:12 and internal/db/db.go:40.", None)


def file_append():
    f = pick(["CHANGELOG.md", "notes.md", ".gitignore"])
    line = {"CHANGELOG.md": "- fix: retry on 429", "notes.md": "- проверить таймауты", ".gitignore": "*.log"}[f]
    goal = f"допиши в {f} строку «{line}»" if ru() else f"append \"{line}\" to {f}"
    c = Conv(goal).step(call("file", "append", path=f, content=line + "\n"), f"appended {len(line) + 1} bytes to {f}")
    return c.end(f"Добавил в `{f}`." if ru() else f"Appended to `{f}`.", None)


def file_read_window():
    f = pick(["server.log", "internal/api/handler.go", "data.csv"])
    off, lim = pick([(0, 40), (200, 50), (1000, 30)])
    goal = f"покажи строки {off + 1}-{off + lim} из {f}" if ru() else f"show lines {off + 1}-{off + lim} of {f}"
    body = "\n".join(f"{off + i + 1}: ..." for i in range(5)) + "\n…"
    c = Conv(goal).step(call("file", "read", path=f, offset=off, limit=lim), body)
    return c.end((f"Строки {off + 1}–{off + lim} из `{f}` выше." if ru() else f"Lines {off + 1}–{off + lim} of `{f}` are above."), None)


def file_mkdir_write():
    lang, main, test = pick(LANGS)
    d = pick(["cmd/tool", "scripts", "tools/gen"])
    goal = f"сделай папку {d} и в ней hello на {lang}" if ru() else f"make a {d} folder with a {lang} hello"
    name = {"go": "main.go", "python": "hello.py", "node": "hello.js", "rust": "main.rs"}[lang]
    code = {"go": "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hello\") }\n", "python": "print(\"hello\")\n",
            "node": "console.log(\"hello\");\n", "rust": "fn main() { println!(\"hello\"); }\n"}[lang]
    runcmd = {"go": f"go run ./{d}", "python": f"python3 {d}/{name}", "node": f"node {d}/{name}", "rust": f"rustc {d}/{name} -o /tmp/hello && /tmp/hello"}[lang]
    c = Conv(goal).step(call("file", "mkdir", path=d), f"created {d}").step(
        call("file", "write", path=f"{d}/{name}", content=code), f"wrote {d}/{name}").step(
        call("run", "shell", command=runcmd), "hello\nexit 0")
    return c.end(f"Создал `{d}/{name}`, запустил: печатает hello." if ru() else f"Created `{d}/{name}` and ran it: prints hello.", None)


def run_slow_build():
    lang, main, test = pick(LANGS)
    goal = "прогони тесты, они долгие" if ru() else "run the tests, they're slow"
    c = Conv(goal).step(call("run", "shell", command=test, timeout=600), "ok  \tall tests passed (212.4s)\nexit 0")
    return c.end("Все тесты прошли (212 с)." if ru() else "All tests passed (212 s).", None)


def run_cwd():
    sub = pick(["frontend", "services/api", "tools"])
    cmd = pick(["npm install", "go build ./...", "make"])
    goal = f"в {sub} сделай {cmd}" if ru() else f"run {cmd} in {sub}"
    c = Conv(goal).step(call("run", "shell", command=cmd, cwd=sub), "exit 0")
    return c.end(f"`{cmd}` в `{sub}` прошёл без ошибок." if ru() else f"`{cmd}` in `{sub}` succeeded.", None)


def no_tool_question():
    q_ru = ["что такое замыкание в Go?", "чем отличается rebase от merge?", "спасибо!", "как дела?", "ты кто?",
            "объясни, что такое idempotency", "ок, понял", "что лучше для логов, zap или slog?"]
    q_en = ["what's a closure in Go?", "rebase vs merge?", "thanks!", "who are you?", "explain idempotency briefly",
            "ok, got it", "zap or slog for logging?"]
    answers = {
        "спасибо!": "Пожалуйста!", "как дела?": "Отлично, готов помочь. Что делаем?", "ок, понял": "Хорошо. Если что, я тут.",
        "ты кто?": "Я ipsupport-code, локальный агент для кода: читаю и правлю файлы, запускаю команды, работаю с git.",
        "thanks!": "You're welcome!", "ok, got it": "Good. I'm here if you need anything.",
        "who are you?": "I'm ipsupport-code, a local coding agent: I read and edit files, run commands and use git.",
    }
    goal = pick(q_ru) if ru() else pick(q_en)
    text = answers.get(goal, "Коротко: " if goal in q_ru else "In short: ")
    if text in ("Коротко: ", "In short: "):
        text += {"что такое замыкание в Go?": "функция, которая захватывает переменные из окружения и живёт дольше него.",
                 "чем отличается rebase от merge?": "rebase переписывает твои коммиты поверх ветки, история линейная; merge сохраняет историю и добавляет коммит слияния.",
                 "объясни, что такое idempotency": "повторный вызов даёт тот же результат, что и первый.",
                 "что лучше для логов, zap или slog?": "slog в стандартной библиотеке и почти не уступает; zap быстрее на горячем пути.",
                 "what's a closure in Go?": "a function that captures variables from its scope and outlives it.",
                 "rebase vs merge?": "rebase replays your commits on top (linear history); merge keeps history and adds a merge commit.",
                 "explain idempotency briefly": "calling it again gives the same result as the first time.",
                 "zap or slog for logging?": "slog is in the standard library and close in speed; zap is faster on hot paths."}[goal]
    return Conv(goal).end(text, None)


SCENARIOS = [git_status, git_diff, git_log, git_show, git_commit, git_branch, git_clone, git_sync, git_init_remote,
             calc, help_lessons, done_tool, agent_run, web_stackexchange, file_find, file_search, file_append,
             file_read_window, file_mkdir_write, run_slow_build, run_cwd, no_tool_question]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scenario", type=int, default=12)
    args = ap.parse_args()
    request = json.load(open(args.request))
    template, tools = request["messages"][0]["content"], request["tools"]
    seen, rows = set(), []
    for scenario in SCENARIOS:
        kept = 0
        for _ in range(args.per_scenario * 5):
            LANG["ru"] = R.random() < 0.6
            conv = scenario()
            key = json.dumps([conv.goal, conv.steps, conv.final], ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            rows.append(conv.render(template, tools, pick(DATES)))
            kept += 1
            if kept >= args.per_scenario:
                break
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"synthetic": len(rows)}))


if __name__ == "__main__":
    main()
