# Publishing mnesio

Releases are cut by pushing a version tag (`v0.1.0`, `v0.1.1`, …). The
[`release.yml`](.github/workflows/release.yml) workflow then publishes to
crates.io, PyPI, npm, and GHCR. Most of it is tokenless (OIDC / trusted
publishing / provenance), but **each registry needs one-time setup**, and two of
them need a **manual first publish** to establish ownership. This document is
that setup.

Do the one-time steps once; after that, every release is just:

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

---

## 1. Container image → GHCR ✅ works immediately

No setup, no secrets — the workflow authenticates with the built-in
`GITHUB_TOKEN`. On the first tag push it publishes:

```
ghcr.io/mnesio/mnesio:X.Y.Z
ghcr.io/mnesio/mnesio:X.Y            (kept up to date)
ghcr.io/mnesio/mnesio:latest
```

After the first publish, make the package public: repo → **Packages** → the
`mnesio` package → **Package settings** → **Change visibility → Public**. Then
anyone can:

```bash
docker run -p 7777:7777 ghcr.io/mnesio/mnesio:latest
```

Built for `linux/amd64` only (the Rust workspace is too heavy to cross-build
arm64 under emulation on the shared runner). Add a native arm64 runner later if
there's demand.

---

## 2. crates.io — first publish is MANUAL, then OIDC

crates.io Trusted Publishing must be configured **per crate**, and you can only
configure it on a crate that already exists. So the first release of each
`mnesio-*` crate is manual; CI handles every release after that.

**One-time bootstrap (claims all 18 names, in dependency order):**

```bash
cargo login                       # paste a crates.io API token (Account Settings → API Tokens)
cargo install cargo-workspaces --locked
cargo ws publish --from-git --allow-dirty --yes --no-git-commit
```

Then, on each crate's page → **Settings → Trusted Publishing → Add**, set:

- Repository owner: `mnesio`
- Repository name: `mnesio`
- Workflow filename: `release.yml`

**The `crates-io` job is gated off until you finish this section.** It only runs
when the repository variable `PUBLISH_CRATES_IO` is set to `true`
(repo → Settings → Secrets and variables → Actions → Variables). Without the
gate the job fails on every tag with `Status: 400 — No Trusted Publishing
config found`, which marks the whole release run red for setup that was never
done. Flip the variable once the manual first publish *and* the per-crate
trusted publishers are in place; after that, a failure is a real failure.

Once every crate has a trusted publisher, the `crates-io` CI job takes over —
delete your local token if you like.

> **Ongoing:** bump the workspace version before tagging. `cargo ws version`
> bumps every crate together; the internal deps already carry
> `version = "0.1.0"` in the root `Cargo.toml`, so keep those in lockstep.

---

## 3. PyPI — fully CI from day one (pending publisher)

PyPI supports a **pending publisher**, so CI can do the *first* publish
tokenlessly. No manual publish needed.

**One-time:** at <https://pypi.org/manage/account/publishing/> add a pending
publisher:

- PyPI project name: `mnesio`
- Owner: `mnesio` · Repository: `mnesio` · Workflow: `release.yml`
- Environment: (leave blank)

That's it — the `pypi` job builds a wheel + sdist with maturin and publishes via
OIDC. (The wheel is `linux/x86_64` manylinux only for now; add a matrix of
runners/Python versions later for macOS/Windows/arm wheels.)

---

### Note on the wheel build (fixed 2026-08-23)

The PyPI job failed on v0.1.1 *before* it ever reached the publish step, so no
amount of credential setup would have helped:

```
💥 maturin failed
  Caused by: python-source is set to `.../crates/mnesio-py/python`, but the
  python module at `.../crates/mnesio-py/python/mnesio` does not exist.
```

The package was renamed mneme → mnesio and the Python source directory was not:
it was still `python/mneme/` while `module-name = "mnesio._mnesio"` expects
`python/mnesio/`. Renamed. Verified locally with the same maturin version CI
uses (1.14.1) by building the wheel, installing it into a clean virtualenv, and
running a write + search round-trip — not just by watching it compile.

## 4. npm — first publish is MANUAL, then provenance

The SDK is scoped (`@mnesio/sdk`), so you need the npm **org** `mnesio`, and the
first publish establishes it.

**One-time:**

1. Create the npm org: <https://www.npmjs.com/org/create> → `mnesio` (free tier
   is fine for public packages).
2. First publish, manually:
   ```bash
   cd sdk/node
   npm login
   npm publish --provenance --access public
   ```
3. Add an **automation** token (npmjs → Access Tokens → Generate → *Automation*)
   as the repo secret **`NPM_TOKEN`** (repo → Settings → Secrets and variables →
   Actions). The `npm` CI job then publishes with provenance on every tag.

> npm also has newer OIDC "trusted publishing"; if you'd rather avoid the token,
> configure it on the package and drop the `NODE_AUTH_TOKEN` line — but the
> automation token is the reliable path today.

---

## Summary

| Target | First release | Later releases | Setup |
|---|---|---|---|
| **GHCR image** | CI ✅ | CI ✅ | none (make package public once) |
| **PyPI** | CI ✅ | CI ✅ | add a pending publisher |
| **crates.io** | **manual** | CI ✅ | manual bootstrap → per-crate trusted publisher |
| **npm** | **manual** | CI ✅ | create org → manual publish → `NPM_TOKEN` secret |

For launch, the important thing is **claiming the names** (crates.io + npm
manual bootstrap, PyPI pending publisher). The container image is the easiest
win — it works the moment you push the tag.
