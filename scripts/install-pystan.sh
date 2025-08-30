#!/usr/bin/env bash
#
# install_pystan_in_poetry_env.sh
# Apple-silicon native install into the *current Poetry environment only*.
# Fixes macOS deployment-target mismatch by aligning httpstan's build target
# with Poetry's default, and uses a priorized smoke test.
#
# Usage:
#   chmod +x install_pystan_in_poetry_env.sh
#   ./install_pystan_in_poetry_env.sh
#
# Env vars you may override:
#   HTTPSTAN_VER : default 4.13.0
#   PYSTAN_VER   : default 3.10.0
#   MODE         : "adhoc" (default) or "lock"
#                  - adhoc: poetry-run pip install (no pyproject changes)
#                  - lock : poetry add (records deps in pyproject/lock)

set -euo pipefail

HTTPSTAN_VER="${HTTPSTAN_VER:-4.13.0}"
PYSTAN_VER="${PYSTAN_VER:-3.10.0}"
MODE="${MODE:-adhoc}"

# --- sanity checks ---
command -v poetry >/dev/null 2>&1 || { echo "Poetry not found."; exit 1; }
[[ -f "pyproject.toml" ]] || { echo "Run from project root (pyproject.toml)."; exit 1; }
[[ "$(uname -s)" == "Darwin" ]] || { echo "This targets macOS/Apple silicon."; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Install Command Line Tools: xcode-select --install"; exit 1; }

echo "==> Poetry environment:"
poetry env info || true
echo

# Ensure env exists (don’t modify deps yet)
poetry install --only-root >/dev/null 2>&1 || true

# Determine the deployment target used by this Poetry env (often 11.0 on arm64).
DEFAULT_TARGET="$(poetry run python -c 'import sysconfig;print(sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET") or "11.0")')"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-$DEFAULT_TARGET}"

# Arm64 & SDK flags (kept consistent across builds/links)
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export ARCHFLAGS="-arch arm64"
export SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || true)"
export CFLAGS="${CFLAGS:-} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
export CXXFLAGS="${CXXFLAGS:-} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
export LDFLAGS="${LDFLAGS:-} -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
if [[ -n "$SDKROOT" ]]; then
  export CFLAGS="$CFLAGS -isysroot $SDKROOT"
  export CXXFLAGS="$CXXFLAGS -isysroot $SDKROOT"
  export LDFLAGS="$LDFLAGS -isysroot $SDKROOT"
fi

echo "==> Using MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
echo

# Optional: remove previous installs to avoid mixing objects built with other targets
poetry run python -m pip uninstall -y httpstan pystan >/dev/null 2>&1 || true

# --- build httpstan wheel from source in a temp dir with the same target ---
TMPDIR="$(mktemp -d)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "==> Building httpstan ${HTTPSTAN_VER} from source (aligned target)…"
(
  cd "$TMPDIR"
  curl -L --fail -o "httpstan-${HTTPSTAN_VER}.tar.gz" \
    "https://github.com/stan-dev/httpstan/archive/refs/tags/${HTTPSTAN_VER}.tar.gz"
  tar -xzf "httpstan-${HTTPSTAN_VER}.tar.gz"
  cd "httpstan-${HTTPSTAN_VER}"

  # parallel build when possible
  export MAKEFLAGS="-j$(sysctl -n hw.ncpu || echo 4)"

  echo "Compiling C++ libs (make)…"
  make

  echo "Packaging wheel (poetry build)…"
  poetry build
)
HTTPSTAN_WHL="$(echo "$TMPDIR"/httpstan-"${HTTPSTAN_VER}"/dist/httpstan-*.whl)"

# --- install into THIS Poetry env ---
case "$MODE" in
  adhoc)
    echo "==> Installing httpstan wheel (adhoc)…"
    poetry run python -m pip install --no-deps "$HTTPSTAN_WHL"

    echo "==> Installing PyStan ${PYSTAN_VER} (adhoc)…"
    poetry run python -m pip install "pystan==${PYSTAN_VER}"
    ;;
  lock)
    echo "==> Recording httpstan wheel in pyproject/lock…"
    poetry add "$HTTPSTAN_WHL"

    echo "==> Recording pystan ${PYSTAN_VER} in pyproject/lock…"
    poetry add "pystan==${PYSTAN_VER}"
    ;;
  *) echo "Invalid MODE='$MODE' (use 'adhoc' or 'lock')."; exit 1;;
esac

# Clear any models compiled previously with a different target
rm -rf "$HOME/Library/Caches/httpstan/${HTTPSTAN_VER}/models" 2>/dev/null || true

echo "==> Smoke test: compile & sample with priors (no stanc warnings)"
poetry run python - <<'PY'
import stan
code = r"""
data { int<lower=0> N; array[N] real y; }
parameters { real mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(mu, sigma);
}
"""
data = {"N": 5, "y": [1.3, 0.9, 1.7, 1.1, 1.5]}
posterior = stan.build(code, data=data)
fit = posterior.sample(num_chains=2, num_samples=300)
print(fit.to_frame().head())
PY

echo "==> Done. PyStan is ready in this Poetry environment (target=$MACOSX_DEPLOYMENT_TARGET)."
[[ "$MODE" == "adhoc" ]] && echo "Note: not recorded in pyproject.toml (MODE=adhoc). Use MODE=lock to pin."
