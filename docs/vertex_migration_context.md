# Vertex Gemini 迁移上下文

## 1. 文档用途

这份文档用于记录当前仓库从 Google AI Studio / Gemini Developer API 迁移到 Vertex AI Gemini 的背景、结论、已实现改动、验证结果和后续建议。

目标不是做完整对外文档，而是作为后续新对话恢复上下文的统一入口。新对话开始时，可先阅读本文件，再结合 `vat/llm/client.py`、`vat/config.py` 和 `config/default.yaml` 理解当前状态。

---

## 2. 背景与目标

### 2.1 背景

项目此前主要通过 Google AI Studio 提供的 OpenAI-compatible 接口访问 Gemini。后续由于可用条款和免费额度策略变化，需要切换到 Vertex AI。

### 2.2 用户目标

本次迁移的核心目标如下：

- 尽量少改代码，最好只改配置
- 如果必须改代码，改动应尽量收敛到 LLM client 层
- 上层业务逻辑不应感知 provider 差异
- `call_llm(...)` 的调用方式尽量保持不变
- 上层依赖的返回结构仍然维持 `response.choices[0].message.content`
- 优先考虑长期可用的认证方式，避免频繁手动刷新 token

### 2.3 已达成的设计结论

最终采用双 client 思路：

- `openai_compatible`
  - 用于原有 OpenAI-compatible 服务
  - 包括 AI Studio、OpenAI-compatible 中转服务、火山等
- `vertex_native`
  - 用于 Vertex Gemini 原生 REST API
  - 由 client 层做请求格式和响应格式适配

上层逻辑继续统一调用 `call_llm(...)`，不直接处理 provider 差异。

---

## 3. Vertex Gemini 的几种使用方式

### 3.1 方式 A：Vertex Native + API key

这是最简单的一条 Vertex Native 配置方式，但**不是当前项目的正式可用方案**。

典型 endpoint：

```text
https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key=API_KEY
https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key=API_KEY
```

特点：

- 配置简单
- 不依赖 access token refresh
- 与 Vertex 原生 REST 语义一致
- 适合单人开发、快速迁移、尽量少改代码的场景

当前仓库里的 `vertex_native` client 仍保留对这一路径的支持；
但在当前项目里，针对 `gemini-3-flash-preview` 的真实实测结果是：

- `vertex_native + api_key`
- 返回 `401 Unauthorized`
- 错误信息：`API keys are not supported by this API`

### 3.2 方式 B：Vertex Native + ADC / Bearer token

这是 Google 官方示例中的标准 GCP 认证方式，也是**当前项目真实可用的正式方案**。

典型 endpoint：

```text
https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
```

Header：

```text
Authorization: Bearer $(gcloud auth print-access-token)
```

特点：

- 更贴近正式 GCP 资源访问方式
- 适合生产或长期规范化部署
- 需要 `project_id` 与 `location`
- 认证通常通过 ADC / gcloud / 服务账号体系完成

### 3.3 方式 C：OpenAI-compatible endpoint

Vertex 也支持通过 OpenAI-compatible 方式接入，但通常需要项目级 endpoint 和 access token。它更适合已有系统强依赖 OpenAI SDK 行为且不愿改 client 结构的场景。

本仓库已经有统一 client 抽象，因此不需要强行把 Vertex 继续包装成 OpenAI-compatible 入口；直接走 `vertex_native` 更清晰。

---

## 4. 本次会话中核实过的官方信息

### 4.1 Vertex 原生 REST 用法是官方支持的

已核实 Google 官方 inference 文档，`generateContent` / `streamGenerateContent` 是 Vertex Gemini 的标准 Model API 入口。

官方文档：

- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

### 4.2 API key 可以用于 Vertex Gemini，但官方更推荐生产环境使用 ADC

已核实 Google 官方 API key 文档，Vertex Gemini 可以用以下两类认证：

- Google Cloud API key
- Application Default Credentials

官方口径：

- API key 适合 testing / 快速开始
- production 更推荐 ADC

官方文档：

- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys

### 4.3 Express Mode 的定位

Express Mode 是 Vertex AI 的简化开通 / 快速 onboarding 入口，便于快速创建 API key 并开始试用。它不是另一套完全独立的模型接口，也不是永久免费的独立产品层。

相关链接：

- Overview:
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
- 入口：
  - https://console.cloud.google.com/expressmode
- Credentials：
  - https://console.cloud.google.com/apis/credentials

### 4.4 计费结论

已核实 Vertex AI pricing 页，Gemini 2.0 / 2.5 系列按 token 计费。页面可能提供 modality 或字符换算视图，但实际 billing 仍以 token 为准。

已确认的信息：

- `Gemini 2.0 / 2.5` 按 token 计费
- `streamGenerateContent` 不是一条额外独立收费模式，本质仍按模型 token 消耗计费
- 若请求失败并返回 400 / 500，官方文档说明不收费
- 长上下文会进入 long context 费率
- grounding 等附加能力可能产生单独计费项

官方链接：

- Pricing:
  - https://cloud.google.com/vertex-ai/generative-ai/pricing
- Standard PayGo:
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/standard-paygo
- Gemini 2.5 Flash-Lite:
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite

### 4.5 关于“免费”的准确理解

需要避免将 Vertex Gemini 理解为“永久免费”。更准确的说法是：

- 新用户可能有 Google Cloud trial / 抵扣金
- 用户在本次会话中明确说明，其所谓“免费”是指 Google Cloud 的 `90 天 / 300 美元抵扣金`
- 正式口径仍应理解为 PayGo 计费，只是在试用阶段可由试用额度抵扣

---

## 5. 本次会话中给出的关键示例

### 5.1 用户已验证通过的 API key + streamGenerateContent 示例

原始示例中的 API key 已脱敏。该调用在用户环境中已跑通。

```bash
curl "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent?key=<REDACTED_API_KEY>" \
-X POST \
-H "Content-Type: application/json" \
-d '{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Explain how AI works in a few words"
        }
      ]
    }
  ]
}'
```

该示例说明：

- API key 方式可直接调用 Vertex 原生接口
- `publishers/google/models/{model}:streamGenerateContent` 路径在用户环境中可用
- 对当前项目而言，`vertex_native + api_key` 是现实可行路线

### 5.2 用户给出的官方 Bearer token 示例

用户随后提供了官方文档风格的 Bearer token 调用示例，也已在用户环境中成功返回结果。

```bash
GOOGLE_CLOUD_PROJECT=vertex-4902033
GOOGLE_CLOUD_LOCATION="global"
API_ENDPOINT="https://aiplatform.googleapis.com"
MODEL_ID="gemini-2.5-flash"
GENERATE_CONTENT_API="generateContent"

curl \
-X POST \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
"${API_ENDPOINT}/v1/projects/${GOOGLE_CLOUD_PROJECT}/locations/${GOOGLE_CLOUD_LOCATION}/publishers/google/models/${MODEL_ID}:${GENERATE_CONTENT_API}" -d \
$'{
  "contents": {
    "role": "user",
    "parts": {
      "text": "Explain how AI works in a few words"
    }
  }
}'
```

用户给出的返回中包含：

- `usageMetadata.promptTokenCount`
- `usageMetadata.candidatesTokenCount`
- `usageMetadata.totalTokenCount`
- `usageMetadata.thoughtsTokenCount`

其中一个关键观察是：

- `thoughtsTokenCount` 很高
- 对当前项目的字幕翻译 / 文本处理场景而言，这意味着默认 thinking 可能显著增加成本与延迟

这一观察会影响后续优化方向：

- 对翻译、ASR 纠错、结构化 JSON 任务，应重点评估是否显式关闭或压低 thinking

### 5.3 官方链接（用户明确提供）

- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start#googlegenaisdk_textgen_with_txt-drest
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

---

## 6. 关于 API key 来源的澄清

用户说明其 API key 来自：

- `Vertex AI Studio`
- `settings/usage-dashboard`
- 并带有明确的 Google Cloud `project`

据此，当前更稳妥的理解是：

- 这把 key 更像是当前 GCP project 下可用于 Vertex 的 Google Cloud API key
- 不应先入为主地把它等同于 Express Mode 专用 key
- 是否来自 Express Mode 的 UI 入口，不影响其已经在 Vertex 原生 REST 中可用这一事实

因此，当前迁移决策不应建立在“它是不是 Express Mode key”上，而应建立在：

- 它能否稳定访问 Vertex Gemini 原生接口
- 当前项目是否更适合用 API key 还是 ADC

---

## 7. 当前仓库状态与使用方式

### 7.1 配置层现状

相关文件：

- `vat/config.py`
- `config/default.yaml`

当前仓库里与 Vertex 直接相关的全局字段包括：

- `llm.provider`
- `llm.auth_mode`
- `llm.location`
- `llm.project_id`
- `llm.credentials_path`

当前 `config/default.yaml` 中的默认片段仍然是：

```yaml
llm:
  provider: "openai_compatible"
  auth_mode: "api_key"
  api_key: "${VAT_GOOGLE_APIKEY}"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  model: "gemini-3-flash-preview"
  location: "global"
  project_id: ""
  credentials_path: ""
```

这表示：

- 默认全局入口仍是 Google AI Studio 的 OpenAI-compatible 路线
- 仓库已经具备切换到 `vertex_native` 的配置能力
- `auth_mode` 用来区分 `api_key` 和 `adc`
- `location`、`project_id`、`credentials_path` 主要服务于 Vertex Native 场景

若要显式切到 Vertex Native，当前更实用的配置示例是：

```yaml
llm:
  provider: "vertex_native"
  auth_mode: "api_key"
  api_key: "${VAT_VERTEX_APIKEY}"
  model: "gemini-2.5-flash"
  location: "global"
```

若要走 ADC，配置示例是：

```yaml
llm:
  provider: "vertex_native"
  auth_mode: "adc"
  model: "gemini-2.5-flash"
  location: "global"
  project_id: "vertex-490203"
  credentials_path: "/home/gzy/.ssh/vat_vertex.json"
```

### 7.2 LLM client 的实际行为

相关文件：

- `vat/llm/client.py`
- `vat/llm/readme.md`

当前已经落地的行为包括：

- `openai_compatible` 与 `vertex_native` 双 provider 路由
- `vertex_native + api_key`
  - 走 `https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key=...`
- `vertex_native + adc`
  - 走 `https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
  - Bearer token 由 `google-auth` 在 client 层获取
- Vertex 返回继续统一适配到 `response.choices[0].message.content`
- 若某个调用显式传入 `base_url`，则仍优先走原有 OpenAI-compatible client

这意味着当前仓库支持两种共存方式：

- 全局默认仍保留 OpenAI-compatible
- 某次迁移或某个阶段可切到 `vertex_native`

### 7.3 这台机器上的使用提示

当前与本机环境直接相关、且对后续恢复上下文有帮助的信息包括：

- 本机存在 service account JSON：
  - `/home/gzy/.ssh/vat_vertex.json`
- 这份 JSON 对应的 project_id：
  - `vertex-490203`
- `credentials_path` 建议始终使用绝对路径
- 此前出现过 `GOOGLE_APPLICATION_CREDENTIALS=.ssh/vat_vertex.json` 这种相对路径配置，容易在不同工作目录下失效

### 7.4 当前仓库中与 Vertex 直接相关的文档和测试

文档：

- `docs/vertex_migration_context.md`
- `docs/vertex_integration_guide.md`
- `docs/superpowers/plans/2026-03-15-vertex-benchmark-progress.md`

测试：

- `tests/test_llm_client_vertex.py`
- `tests/test_config.py`
- `tests/test_vertex_translation_flow.py`
- `tests/test_translator_error_handling.py`

---

## 8. 当前推荐的项目使用方式

### 8.1 现阶段推荐

对于当前项目，默认仍建议优先采用：

- `Vertex Native + API key + generateContent`

原因：

- 配置最简单
- 迁移成本最低
- 不需要额外处理 token refresh
- 与当前批处理翻译、断句、视频信息翻译这类“等待完整结果再后处理”的场景更契合

### 8.2 长期部署推荐

若后续进入更稳定、长期的部署环境，则建议优先考虑：

- `Vertex Native + ADC + generateContent`

原因：

- 更符合 Google 官方推荐的认证方式
- 权限模型更规范
- 与 GCP service account / IAM 体系更一致

### 8.3 为什么当前默认不把 stream 作为主路径

当前项目主要任务是：

- 字幕翻译
- 文本优化
- 结构化解析
- 视频信息翻译

这些任务通常都需要等待完整结果后再做后处理，因此默认更适合：

- `generateContent`

`streamGenerateContent` 已被确认是官方支持接口，但它返回的是事件流而非当前统一抽象下的单对象结构，因此现阶段不适合作为默认实现。

### 8.4 thinking 的后续关注点

用户之前给出的 `usageMetadata` 显示 `thoughtsTokenCount` 偏高，这对当前场景有两个直接影响：

- 成本可能明显上升
- 延迟可能明显上升

对翻译、ASR 纠错、结构化抽取这类任务，后续应优先评估是否需要显式关闭或压低 thinking。

---

## 9. 当前仓库中已有的验证记录

本节记录的是当前仓库内已经留下的验证证据与结论，用于恢复上下文；不表示本次会话重新执行了这些测试或 benchmark。

### 9.1 测试覆盖面

从当前测试文件可以确认，仓库至少已经覆盖了以下行为：

- `tests/test_llm_client_vertex.py`
  - `vertex_native + api_key` 请求构造与响应适配
  - `vertex_native + adc` 的 project-scoped endpoint
- `tests/test_config.py`
  - `vertex_native + auth_mode=adc` 在无 API key 时的配置可用性判断
- `tests/test_vertex_translation_flow.py`
  - 从 `Config.from_dict(...)` 到 `LLMTranslator.translate_subtitle(...)` 再到 `translated.srt` 产出的链路
  - API key 与 ADC 两条 Vertex 路线
- `tests/test_translator_error_handling.py`
  - 迁移过程中相关翻译路径的回归覆盖

### 9.2 已记录的测试结果

`docs/vertex_integration_guide.md` 中已经记录过一次更完整的验证命令：

```bash
HOME=/tmp pytest tests/test_llm_client_vertex.py tests/test_config.py tests/test_translator_error_handling.py tests/test_vertex_translation_flow.py -q
```

该文档记录的结果是：

- `39 passed`

这说明仓库中至少存在过一轮同时覆盖配置、client、翻译流程与回归路径的集中验证。

### 9.3 已记录的真实翻译链路验证

`docs/vertex_integration_guide.md` 中还记录了两条最小真实翻译流程：

- API key
  - 输入：`おはようございます`
  - 输出：`早上好`
  - 产物：`/tmp/vat-e2e-api-key/translated.srt`
- ADC
  - 输入：`こんばんは`
  - 输出：`晚上好`
  - 产物：`/tmp/vat-e2e-adc/translated.srt`

这类记录比单纯的 client 单测更有价值，因为它说明调用链至少曾经从配置层一路走到最终字幕文件产出。

### 9.4 已记录的 benchmark 结论

`docs/superpowers/plans/2026-03-15-vertex-benchmark-progress.md` 中已经记录过一次 `gemini-3-flash-vertex` 的 benchmark 结果，结论可概括为：

- Vertex Native 接口链路可用
- 与历史 `gemini-3-flash` 基线相比，整体质量接近
- 输出并非逐句完全一致，更像是同模型在不同官方入口下的自然改写

如果后续目标是“验证 Vertex 接口是否能替代旧入口”，这份 benchmark 记录值得保留；如果只是日常使用接口，它更适合作为辅助背景，而不是主使用文档。

---

## 10. 当前已知限制与未完成事项

### 10.1 当前实现仍然偏保守

当前 `vertex_native` 已经不再只是 API key 最小接入，但整体实现仍然偏保守，主要聚焦于：

- `messages`
- `temperature`
- `max_tokens -> maxOutputTokens`
- 返回结构适配到 `response.choices[0].message.content`

### 10.2 目前还没有纳入默认实现的内容

当前尚未系统纳入默认实现的内容包括：

- `streamGenerateContent` 作为默认路径
- thinking 参数显式控制
- 更细的 `usageMetadata` 向上层透传
- 更多 Vertex 原生 generationConfig 字段

### 10.3 `location` 在两条路径里的语义不同

当前 API key 路径采用的是：

- `v1/publishers/google/models/{model}:generateContent?key=...`

因此：

- `location` 在 API key 路径里没有进入 URL
- `location` 在 ADC 的 project-scoped URL 中才会直接参与路由

这点在阅读配置时需要区分清楚，避免误以为 `location=global` 会影响 API key 路径的 endpoint。

---

## 11. 推荐的新对话恢复方式

若后续开启新对话，建议按以下顺序恢复上下文：

1. 先阅读：
   - `docs/vertex_migration_context.md`
2. 再阅读：
   - `docs/vertex_integration_guide.md`
3. 若涉及效果对比，再阅读：
   - `docs/superpowers/plans/2026-03-15-vertex-benchmark-progress.md`
4. 然后检查代码与测试：
   - `vat/llm/client.py`
   - `vat/config.py`
   - `config/default.yaml`
   - `tests/test_llm_client_vertex.py`
   - `tests/test_config.py`
   - `tests/test_vertex_translation_flow.py`

可直接在新对话里给出类似指令：

```text
先阅读 docs/vertex_migration_context.md 和 docs/vertex_integration_guide.md；如果要看质量对比，再看 docs/superpowers/plans/2026-03-15-vertex-benchmark-progress.md。然后检查 vat/llm/client.py、vat/config.py、config/default.yaml、tests/test_llm_client_vertex.py、tests/test_config.py 和 tests/test_vertex_translation_flow.py。我们继续处理 Vertex Gemini 集成，但不改上层业务逻辑。
```

---

## 12. 本次结论的简短版

- 当前仓库已经支持 `vertex_native + api_key` 与 `vertex_native + adc`
- 对实际使用而言，`api_key + generateContent` 仍是最省心的现阶段方案
- 对长期部署而言，`adc + generateContent` 更值得保留为正式方案
- 对恢复上下文而言，应优先看迁移上下文文档、集成说明和 benchmark 记录，而不是早期过程性计划
- `streamGenerateContent`、thinking 控制、`usageMetadata` 暴露仍是后续可继续补强的方向

---

## 13. 相关文件索引

### 13.1 文档

- `docs/vertex_migration_context.md`
- `docs/vertex_integration_guide.md`
- `docs/superpowers/plans/2026-03-15-vertex-benchmark-progress.md`

### 13.2 代码与测试

- `vat/llm/client.py`
- `vat/config.py`
- `config/default.yaml`
- `tests/test_llm_client_vertex.py`
- `tests/test_config.py`
- `tests/test_vertex_translation_flow.py`
- `tests/test_translator_error_handling.py`

### 13.3 外部链接

- Vertex inference 文档：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- API key 文档：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
- Vertex start 文档：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start#googlegenaisdk_textgen_with_txt-drest
- Express Mode overview：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
- Express Mode 入口：
  - https://console.cloud.google.com/expressmode
- Credentials：
  - https://console.cloud.google.com/apis/credentials
- Pricing：
  - https://cloud.google.com/vertex-ai/generative-ai/pricing
- Standard PayGo：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/standard-paygo
- Gemini 2.5 Flash-Lite：
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite
