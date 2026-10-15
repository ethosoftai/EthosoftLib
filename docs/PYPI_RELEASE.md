# Publishing EthosoftLib to PyPI

EthosoftLib uses a GitHub Actions **Trusted Publisher (OIDC)**. Do not
paste personal PyPI tokens into commits, GitHub workflow inputs, issue
comments, chat messages or logs.

## First-time PyPI setup (one-time account-owner action)

1. Sign in to https://pypi.org/ with the PyPI account that will own the
   `ethosoftlib` project.
2. If the project does not yet exist, go to **Account settings > Publishing**
   and add a **new pending GitHub publisher**. If the project already exists
   under your account, go to its **Manage > Publishing** tab instead.
3. Register exactly:
   - PyPI project name: `ethosoftlib`
   - GitHub owner: `ethosoftai`
   - GitHub repository: `EthosoftLib`
   - Workflow filename: `publish-pypi.yml`
   - GitHub environment: `pypi`
4. On GitHub, under **Settings > Environments**, create `pypi`, configure
   deployment branch/tag restrictions if appropriate, and set required
   reviewers if available to your repository and plan.
5. Enable two-factor authentication for PyPI maintainers. If an API token
   was ever disclosed, immediately **revoke** it from PyPI account settings
   and create a new one only when strictly necessary. Our workflow
   requires **no static PyPI token**.

A pending publisher only registers trust. It does not reserve the project
name; the first successful upload creates the project.

## Dry run and release

To test without publishing, open **Actions > Publish EthosoftLib to PyPI
> Run workflow** on the default branch. Only the `build` job executes; it
validates package metadata, runs a dependency-free subset of the tests,
builds the source distribution and wheel, and uploads both as an Actions
artifact.

To actually publish version `0.4.0`:
1. Make sure the desired code is on the default branch, all required CI
   jobs pass, and `pyproject.toml` declares version `0.4.0`.
2. Create a GitHub release with a tag `v0.4.0` pointing to that commit,
   using **Releases > Draft a new release > Publish release**.
3. The release workflow verifies that the GitHub release tag matches
   the exact project version, builds verified artifacts and uploads
   to PyPI using a short-lived OIDC token.
4. Inspect the workflow and https://pypi.org/project/ethosoftlib/ .
5. Smoke-test from a clean environment:
   `python -m pip install --no-cache-dir "ethosoftlib==0.4.0"`
   `python -c "import ethosoftlib; print(ethosoftlib.__version__)"`

If `0.4.0` already exists on PyPI, do not attempt to overwrite it.
Increment the project version and use a matching release tag. PyPI does not
permit replacing an uploaded distribution file.

## Native libraries

This first automated publication deliberately contains only:
- `ethosoftlib-0.4.0-py3-none-any.whl` (pure Python SDK)
- `ethosoftlib-0.4.0.tar.gz` (source distribution)

Mercan inference and NedoTokenizer bindings remain available at
`ethosoftlib.mercan` and `ethosoftlib.nedo`, but users must install the
respective native runtime separately to execute native operations. The
distribution does **not** include model weights, `libmercan`, or the Rust
NDSRF004 dynamic library. This is intentional until audited Linux manylinux
wheels and macOS/Windows distribution prerequisites are finalized.

The existing combined Mercan + Nedo CI native wheels are valuable validated
build artifacts, but they must **not** be uploaded to PyPI alongside the
universal wheel until platform tagging, native library dependencies, and
package version/release strategy have been reviewed. Do not publish
platform-native artifacts from untrusted sources.
