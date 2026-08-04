dev:
    uv run uvicorn server.main:app --host "localhost" --port 8086 --reload --log-level debug --log-config server/utils/logging_config.yaml --timeout-graceful-shutdown 5 --ws-ping-interval 5 --ws-ping-timeout 10

dev-trace:
    uv run uvicorn server.main:app --reload --log-level trace --log-config server/utils/logging_config.yaml

test *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov scripts --cov-branch --cov-report=term-missing --no-cov-on-fail {{FLAGS}}

ntest *FLAGS:
    uv run pytest --showlocals --tb=auto -ra --cov server --cov-branch --cov-report=term-missing --no-cov-on-fail -n auto {{FLAGS}}

ruff *FLAGS:
    uv run ruff check server/ tests/ {{FLAGS}}

ruff-format *FLAGS:
    uv run ruff format server/ tests/ {{FLAGS}}

auth-static:
    uv run python ./server/scripts/check_auth.py static ./server/ -v

auth-runtime:
    uv run python ./server/scripts/check_auth.py runtime ./server/main.py -v

pyright:
    uv run pyright

mypy *FLAGS:
    uv run mypy server/ tests/ {{FLAGS}}

pre-commit:
    uv run pre-commit run --all-files

todo:
    git grep "# TODO" -- "*.py"
    git grep "# FIX" -- "*.py"
    git grep "# DONE" -- "*.py"

license-check *FLAGS:
    uv run scripts/check_license_header.py {{FLAGS}}

check: license-check ruff ruff-format pyright auth-static auth-runtime

ui-rebuild:
   (cd $(git rev-parse --show-toplevel)/webui && npm run buildx)

test-webui:
   (cd $(git rev-parse --show-toplevel)/webui && npm test)

ts-check *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome check src {{FLAGS}})

ts-lint *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome lint src {{FLAGS}})

ts-format *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome format src {{FLAGS}})

ts-fix *FLAGS:
   (cd $(git rev-parse --show-toplevel)/webui && npx biome check --write src {{FLAGS}})

env-copy:
    uv run python ./scripts/copy_envs.py

replace-docker *FLAGS:
    docker stop infra-infra-1; just dev {{FLAGS}}

inspector:
    npx @modelcontextprotocol/inspector node build/index.js

get-ollama-models *FLAGS:
    uv run python ./scripts/get_ollama_models.py {{FLAGS}}

clear-ollama-cache:
    uv run python ./scripts/get_ollama_models.py --clear-cache

get-vllm-models *FLAGS:
    uv run python ./scripts/get_huggingface_models.py --type llm --top-by-downloads 200 --top-by-likes 100 --top-by-trending 50 --output static/vllm-min.json {{FLAGS}}
    uv run python ./scripts/get_huggingface_models.py --type reranker --top-by-downloads 10 --top-by-trending 10 --output static/vllm-min.json {{FLAGS}}
    uv run python ./scripts/get_huggingface_models.py --type embedding --top-by-downloads 10 --top-by-trending 10 --output static/vllm-min.json {{FLAGS}}

