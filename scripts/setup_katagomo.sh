#!/usr/bin/env bash
# Cài KataGomo (engine Caro/Gomoku) để dùng làm đối thủ ngoài ở bàn 11×11.
#
#   scripts/setup_katagomo.sh              # cài nếu chưa có
#   scripts/setup_katagomo.sh --rebuild    # xoá bản build cũ rồi build lại
#
# Kết quả nằm ở ~/.local/share/caro/katagomo (đổi bằng KATAGOMO_HOME):
#   katago                  engine Caro2024 đã vá luật freestyle (symlink tới src/build/katago)
#   b10_freestyle15x.bin.gz model freestyle 15×15, chạy được bàn 11×11
#   default_gtp.cfg         SIXWINRULE_ALWAYS + wallBlockRule=false
#
# Vì sao phải vá: Caro2024 gốc coi thế OXXXXXO (đúng 5 quân bị chặn hai đầu) là
# chưa thắng, trái luật đồ án. Patch đổi nhánh myConNum==5 để SIXWINRULE_ALWAYS
# thắng mọi dãy 5, kể cả bị chặn hai đầu.
#
# Yêu cầu: git, cmake ≥ 3.16, trình biên dịch C++14, curl, tar.
# Build CPU/Eigen, không cần GPU; lần đầu khoảng 2–5 phút tuỳ máy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${KATAGOMO_HOME:-$HOME/.local/share/caro/katagomo}"
REPO="${KATAGOMO_REPO:-https://github.com/hzyhhzy/KataGomo.git}"
BRANCH="${KATAGOMO_BRANCH:-Caro2024}"
MODEL_URL="${KATAGOMO_MODEL_URL:-https://github.com/hzyhhzy/KataGomo/releases/download/gomocup2025/b10_freestyle15x.bin.gz}"
CORES="$(nproc 2>/dev/null || echo 2)"
JOBS="${KATAGOMO_JOBS:-$(( CORES / 2 > 4 ? 4 : (CORES / 2 > 0 ? CORES / 2 : 1) ))}"
REBUILD=0

for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "tham số lạ: $arg" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "thiếu $1 — cài trước rồi chạy lại" >&2; exit 1; }; }
need git; need cmake; need curl; need tar; need c++; need patch

mkdir -p "$DEST"
echo "thư mục cài: $DEST"

# ---- model + config -------------------------------------------------------
if [[ ! -s "$DEST/b10_freestyle15x.bin.gz" ]]; then
  echo "tải model freestyle 15×15…"
  curl -L --fail --retry 2 -o "$DEST/b10_freestyle15x.bin.gz.part" "$MODEL_URL"
  mv "$DEST/b10_freestyle15x.bin.gz.part" "$DEST/b10_freestyle15x.bin.gz"
fi
install -m 0644 "$ROOT/configs/katagomo/default_gtp.cfg" "$DEST/default_gtp.cfg"

# ---- build ----------------------------------------------------------------
ENGINE_ID="caro2024-freestyle5"
if [[ "$REBUILD" == 1 ]]; then
  rm -rf "$DEST/src" "$DEST/katago" "$DEST/.engine"
fi

if [[ ! -x "$DEST/katago" || "$(cat "$DEST/.engine" 2>/dev/null || true)" != "$ENGINE_ID" ]]; then
  CURRENT_BRANCH="$(git -C "$DEST/src" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "clone KataGomo ($BRANCH)…"
    rm -rf "$DEST/src"
    git clone --depth 1 --branch "$BRANCH" "$REPO" "$DEST/src"
  fi

  if ! grep -q "freestyle: blocked five still wins" "$DEST/src/cpp/game/gamelogic.cpp"; then
    echo "áp patch luật freestyle (5 quân bị chặn hai đầu vẫn thắng)…"
    patch -d "$DEST/src" -p1 < "$ROOT/configs/katagomo/freestyle-five.patch"
  fi
  grep -q "freestyle: blocked five still wins" "$DEST/src/cpp/game/gamelogic.cpp" || {
    echo "không áp được patch — kiểm tra configs/katagomo/freestyle-five.patch" >&2; exit 1;
  }

  EIGEN_SRC="$DEST/eigen-3.4.0"
  if [[ ! -d "$EIGEN_SRC/Eigen" ]]; then
    echo "tải Eigen 3.4.0…"
    curl -L --fail --retry 2 "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz" \
      | tar -xz -C "$DEST"
  fi
  EIGEN_PREFIX="$DEST/eigen-install"
  if [[ ! -d "$EIGEN_PREFIX/include/eigen3/Eigen" ]]; then
    echo "cài header Eigen…"
    cmake -S "$EIGEN_SRC" -B "$DEST/eigen-build" -DCMAKE_INSTALL_PREFIX="$EIGEN_PREFIX" \
      -DEIGEN_BUILD_DOC=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --install "$DEST/eigen-build" >/dev/null
  fi

  AVX2=0
  if grep -qm1 'avx2' /proc/cpuinfo 2>/dev/null; then AVX2=1; fi
  echo "cấu hình build (AVX2=$AVX2, $JOBS luồng)…"
  cmake -S "$DEST/src/cpp" -B "$DEST/src/build" \
    -DUSE_BACKEND=EIGEN -DCMAKE_PREFIX_PATH="$EIGEN_PREFIX" \
    -DNO_LIBZIP=1 -DNO_GIT_REVISION=1 -DUSE_AVX2="$AVX2" \
    -DCMAKE_BUILD_TYPE=Release
  echo "biên dịch…"
  cmake --build "$DEST/src/build" -j "$JOBS"
  ln -sfn "$DEST/src/build/katago" "$DEST/katago"
  printf '%s\n' "$ENGINE_ID" > "$DEST/.engine"
fi
echo
echo "xong. Kiểm tra:"
printf 'boardsize 11\nclear_board\ngenmove b\nquit\n' |
  "$DEST/katago" gtp -model "$DEST/b10_freestyle15x.bin.gz" -config "$DEST/default_gtp.cfg" \
    -override-config logFile=/dev/null -override-config logAllGTPCommunication=false 2>/dev/null |
  tail -4
echo
echo "Chạy UI:  uv run uvicorn caro.ui.server:app --port 8000"
echo "Chọn tác tử KataGomo-11 trong danh sách."
