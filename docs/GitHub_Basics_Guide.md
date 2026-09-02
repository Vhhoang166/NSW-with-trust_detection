# GitHub Basics Guide

This guide explains the GitHub ideas that came up while pushing the Moradi
work. It is written for future-you, not for Git experts.

## 1. Git vs GitHub

`git` is the tool on your computer.

Git tracks files, branches, commits, and history locally.

`GitHub` is the website/server.

GitHub stores a copy of your repository online so other people can see it,
review it, and collaborate.

Simple way to remember:

- Git = local version control on your machine.
- GitHub = online hosting for Git repositories.

## 2. Repository

A repository, or repo, is a project tracked by Git.

Example:

```text
https://github.com/AIPcoder/NSW-Trustworthy
```

This means:

- owner or organization: `AIPcoder`
- repository name: `NSW-Trustworthy`

Your local folder can point to a GitHub repo through a remote.

## 3. Remote

A remote is the online repo URL your local Git folder pushes to or pulls from.

Check remotes:

```bash
git remote -v
```

Example output:

```text
origin  https://github.com/AIPcoder/NSW-Trustworthy.git (fetch)
origin  https://github.com/AIPcoder/NSW-Trustworthy.git (push)
```

`origin` is just the default nickname for the remote.

Change the remote:

```bash
git remote set-url origin https://github.com/AIPcoder/NSW-Trustworthy.git
```

Add a remote if none exists:

```bash
git remote add origin https://github.com/AIPcoder/NSW-Trustworthy.git
```

## 4. Account Access

If GitHub says:

```text
Repository not found
```

it does not always mean the repo does not exist.

It can also mean:

- you are logged in as the wrong GitHub account,
- your account has not been invited,
- the repo is private,
- your token does not have permission,
- the repo URL is typed wrong.

Always test in the browser first:

```text
https://github.com/AIPcoder/NSW-Trustworthy
```

If you cannot open it in the browser with your current GitHub account, Git
push will not work either.

## 5. GitHub Authentication

GitHub no longer accepts normal account passwords for Git push over HTTPS.

You usually need a Personal Access Token.

Create one here:

```text
https://github.com/settings/tokens
```

For private repos, the token needs repo access.

When Git asks:

```text
Username:
Password:
```

use:

- username = your GitHub username,
- password = your token.

## 6. Clearing Old GitHub Credentials

If your Mac keeps using an old GitHub account, clear saved credentials.

Run:

```bash
printf "protocol=https\nhost=github.com\n" | git credential-osxkeychain erase
```

You can also try:

```bash
security delete-internet-password -s github.com
```

If it says the item was not found, that is okay.

Then push again. Git should ask you to log in again.

## 7. Branches

A branch is a separate line of work.

You usually do not work directly on `main`.

Instead, create a branch:

```bash
git checkout -b moradi-current
```

Check your current branch:

```bash
git branch --show-current
```

List branches:

```bash
git branch
```

## 8. Status

Use status constantly.

```bash
git status -sb
```

This tells you:

- what branch you are on,
- which files changed,
- which files are staged,
- which files are untracked.

Common symbols:

- `??` means untracked file.
- `M` means modified file.
- `D` means deleted file.
- `A` means added file.

## 9. Staging Files

Before committing, you stage the files you want.

Stage one file:

```bash
git add NSW_Moradi.py
```

Stage the Moradi file and document:

```bash
git add NSW_Moradi.py docs/NSW_Moradi_Changes.md
```

Avoid this when the worktree has unrelated changes:

```bash
git add -A
```

`git add -A` stages everything. That can accidentally include unrelated files.

## 10. Commit

A commit is a saved checkpoint.

Commit staged files:

```bash
git commit -m "Add current Moradi implementation"
```

A good commit message is short and says what changed.

Examples:

```text
Add current Moradi implementation
Fix Moradi detector F1 metrics
Document Moradi changes
```

## 11. Push

Push sends your branch to GitHub.

```bash
git push -u origin moradi-current
```

Meaning:

- `origin` = the GitHub remote,
- `moradi-current` = the branch name,
- `-u` = remember this remote branch for future pushes.

After the first push, you can usually just run:

```bash
git push
```

## 12. Delete a Remote Branch

If a branch already exists on GitHub and you want to remove it:

```bash
git push origin --delete moradi-current
```

If Git says the remote branch does not exist, that is fine.

## 13. Pull Request

A Pull Request, or PR, asks to merge your branch into another branch, usually
`main`.

After pushing a branch, GitHub often gives you a link to open a PR.

You can also open:

```text
https://github.com/AIPcoder/NSW-Trustworthy/pull/new/moradi-current
```

A PR should explain:

- what changed,
- why it changed,
- how it was tested.

## 14. Clean Temporary Repo Strategy

When your main project folder is messy, a safe strategy is to create a clean
temporary repo containing only the files you want to push.

Example:

```bash
cd "/Users/admin/Prof Sarper Aydin/NSW-with-trust_detection"

rm -rf /tmp/nsw-moradi-push
mkdir -p /tmp/nsw-moradi-push/docs

cp NSW_Moradi.py /tmp/nsw-moradi-push/NSW_Moradi.py
cp docs/NSW_Moradi_Changes.md /tmp/nsw-moradi-push/docs/NSW_Moradi_Changes.md

cd /tmp/nsw-moradi-push
git init
git checkout -b moradi-current
git add NSW_Moradi.py docs/NSW_Moradi_Changes.md
git commit -m "Add current Moradi implementation"
git remote add origin https://github.com/AIPcoder/NSW-Trustworthy.git
git push -u origin moradi-current
```

Why this is useful:

- It avoids accidentally pushing unrelated deleted or modified files.
- It creates a clean branch with only the intended files.

## 15. Useful Commands

Check current repo status:

```bash
git status -sb
```

Check remote:

```bash
git remote -v
```

Check current branch:

```bash
git branch --show-current
```

Create new branch:

```bash
git checkout -b branch-name
```

Stage files:

```bash
git add file1 file2
```

Commit:

```bash
git commit -m "Message"
```

Push:

```bash
git push -u origin branch-name
```

See commit history:

```bash
git log --oneline --decorate --graph --all
```

See changes before staging:

```bash
git diff
```

See staged changes:

```bash
git diff --staged
```

## 16. Mental Model

Think of GitHub work like this:

1. Edit files.
2. Check status.
3. Stage only the files you want.
4. Commit the staged files.
5. Push the branch to GitHub.
6. Open a PR.

The most important safety habit:

```bash
git status -sb
```

Run it often. It tells you what Git thinks is happening.

## 17. What Happened In This Project

The old local repo pointed to:

```text
Vhhoang166/NSW-with-trust_detection
```

But the new target repo is:

```text
AIPcoder/NSW-Trustworthy
```

That means the remote had to be changed.

Also, the local workspace had many unrelated changes, so the safer choice was
to create a temporary clean repo with only:

```text
NSW_Moradi.py
docs/NSW_Moradi_Changes.md
```

That avoided pushing unrelated files.

## 18. Final Advice

Before pushing, always ask:

- Am I on the right branch?
- Is `origin` pointing to the right GitHub repo?
- Does `git status -sb` show only the files I want?
- Am I logged into the GitHub account that has access?
- Did I commit before pushing?

If all answers are yes, pushing is usually safe.

