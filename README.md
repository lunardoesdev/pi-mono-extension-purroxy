# pi-mono Purroxy Extension

`pi-mono` extension for routing provider traffic through custom base URLs.

It supports:

- Anthropic
- OpenAI
- Gemini (`google`)
- Vertex (`google-vertex`)
- xAI
- GLM (`zai`)
- DeepSeek

`DeepSeek` is added as a custom provider. `GLM` maps to pi-mono's built-in `zai` provider because that is where the bundled GLM models live.

## Files

- `src/index.ts`: extension entrypoint
- `package.json`: package-style `pi` extension manifest

## Install

Install dependencies once:

```bash
npm install
```

Load directly:

```bash
pi -e /path/to/pi-mono-extension-purroxy
```

Or symlink it into your extensions directory:

```bash
mkdir -p ~/.pi/extensions
ln -s $(pwd) ~/.pi/extensions/purroxy
```

## Environment

Copy-paste template:

```bash
# Anthropic
export ANTHROPIC_BASE_URL="https://your-proxy.example/anthropic"
export ANTHROPIC_API_KEY="your-anthropic-key"

# OpenAI
export OPENAI_BASE_URL="https://your-proxy.example/openai/v1"
export OPENAI_API_KEY="your-openai-key"

# Gemini (google)
export GEMINI_BASE_URL="https://your-proxy.example/google/v1beta"
export GEMINI_API_KEY="your-gemini-key"

# Vertex
export VERTEX_BASE_URL="https://your-proxy.example/vertex"
export VERTEX_API_KEY="your-vertex-key"

# If you use ADC instead of a Vertex API key, set these instead:
# export GOOGLE_CLOUD_PROJECT="your-gcp-project"
# export GOOGLE_CLOUD_LOCATION="us-central1"

# xAI
export XAI_BASE_URL="https://your-proxy.example/xai/v1"
export XAI_API_KEY="your-xai-key"

# GLM
export GLM_BASE_URL="https://your-proxy.example/glm/v4"
export GLM_API_KEY="your-glm-key"

# DeepSeek
export DEEPSEEK_BASE_URL="https://your-proxy.example/deepseek/v1"
export DEEPSEEK_API_KEY="your-deepseek-key"

# Optional: custom DeepSeek model list
export DEEPSEEK_MODELS="deepseek-chat,deepseek-reasoner"
```

Accepted variable names by provider:

- Anthropic: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`
- OpenAI: `OPENAI_BASE_URL`, `OPENAI_API_KEY`
- Gemini: `GEMINI_BASE_URL` or `GOOGLE_BASE_URL`, plus `GEMINI_API_KEY`
- Vertex: `VERTEX_BASE_URL` or `GOOGLE_VERTEX_BASE_URL`, plus `VERTEX_API_KEY` or `GOOGLE_VERTEX_API_KEY` or `GOOGLE_CLOUD_API_KEY`
- Vertex with ADC: `VERTEX_BASE_URL` plus `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`
- xAI: `XAI_BASE_URL` or `GROK_BASE_URL`, plus `XAI_API_KEY` or `GROK_API_KEY`
- GLM: `GLM_BASE_URL` or `ZAI_BASE_URL` or `ZHIPU_BASE_URL`, plus `GLM_API_KEY` or `ZAI_API_KEY` or `ZHIPU_API_KEY`
- DeepSeek: `DEEPSEEK_BASE_URL`, `DEEPSEEK_API_KEY`

If `DEEPSEEK_API_KEY` is set but `DEEPSEEK_BASE_URL` is not, the extension falls back to the official OpenAI-compatible DeepSeek endpoint.

## Notes

- Anthropic expects a host-style base URL, so the extension strips accidental `/v1` or `/messages` suffixes.
- OpenAI-compatible providers keep versioned prefixes such as `/v1`, but strip full endpoint suffixes such as `/chat/completions` or `/responses`.
- The extension installs a custom Vertex streamer because pi-mono's built-in `google-vertex` provider does not currently consume `model.baseUrl`.
