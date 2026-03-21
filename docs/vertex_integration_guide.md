# Vertex Gemini 集成说明

## 1. 这次分支改了什么

这次分支的目标是把 VAT 的 LLM 接入从“只能走 OpenAI-compatible / Vertex API key 最小支持”补到“正式支持 Vertex Native 的两种认证模式”，同时不改上层翻译、断句、视频信息翻译的调用方式。

本次改动包括：

- `vat/llm/client.py`
  - `vertex_native` 现在支持两种认证：
    - `auth_mode=api_key`
    - `auth_mode=adc`
  - `adc` 路线使用 `google-auth` 获取 Bearer token
  - 继续把 Vertex 返回适配为 `response.choices[0].message.content`
- `vat/config.py`
  - 新增全局配置：
    - `llm.auth_mode`
    - `llm.project_id`
    - `llm.credentials_path`
  - `LLMConfig.is_available()` 现在会按认证模式判断可用性
- `config/default.yaml`
  - 补充 Vertex 的新配置项
- `tests/test_llm_client_vertex.py`
  - 新增 ADC 路线测试
- `tests/test_config.py`
  - 新增 Vertex ADC 配置可用性测试
- `tests/test_vertex_translation_flow.py`
  - 新增从 `Config -> LLMTranslator -> Vertex -> translated.srt` 的集成测试
- `vat/llm/readme.md`
  - 补充 Vertex Native 的使用说明

## 2. 为什么这样改

项目当前已经有统一的 `call_llm(...)` 抽象。相比强行把 Vertex 包成 OpenAI-compatible，直接在 client 层对 Vertex Native 做适配更清楚，也更容易把 API key 和 ADC 两条路一起支持掉。

这样做的结果是：

- 上层业务不需要知道 provider 差异
- 现有翻译器、断句、场景识别调用方式不变
- API key 可以继续用于快速验证
- ADC 可以作为长期和正式部署方案

## 3. 现在支持的接入方式

### 3.1 OpenAI-compatible

配置示例：

```yaml
llm:
  provider: "openai_compatible"
  auth_mode: "api_key"
  api_key: "${VAT_GOOGLE_APIKEY}"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
  model: "gemini-3-flash-preview"
```

特点：

- 兼容 OpenAI SDK 语义
- 适合未来切换到其他 OpenAI-style 站点或中转服务
- 也是项目早期使用 AI Studio 时的主要方式

### 3.2 Vertex Native + API key

配置示例：

```yaml
llm:
  provider: "vertex_native"
  auth_mode: "api_key"
  api_key: "${VAT_VERTEX_APIKEY}"
  model: "gemini-3-flash-preview"
  location: "global"
```

请求路径：

```text
https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key=...
```

特点：

- 理论上配置最简单
- 适合某些 Vertex 项目或临时验证

但对**当前这套项目和账号**，实测结果是：

- `gemini-3-flash-preview`
- `provider=vertex_native`
- `auth_mode=api_key`

会返回：

```text
401 Unauthorized
API keys are not supported by this API.
```

所以它在**当前项目里不是可用的正式方案**。

### 3.3 Vertex Native + ADC

配置示例：

```yaml
llm:
  provider: "vertex_native"
  auth_mode: "adc"
  model: "gemini-3-flash-preview"
  location: "global"
  project_id: "vertex-490203"
  credentials_path: "/home/gzy/.ssh/vat_vertex.json"
```

请求路径：

```text
https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
```

特点：

- 更符合 Google 官方推荐的正式认证方式
- 更适合长期部署
- 是**当前项目实测可用**的正式方案

注意：

- `credentials_path` 建议始终写绝对路径
- 这台机器之前的 `GOOGLE_APPLICATION_CREDENTIALS=.ssh/vat_vertex.json` 是相对路径，容易在不同工作目录下失效

## 4. 这次没有做的事

以下内容目前还没有纳入默认实现：

- `streamGenerateContent` 作为默认路径
- Vertex thinking 参数控制
- `usageMetadata` 透传给上层

原因：

- VAT 当前是批处理翻译场景，不需要边生成边消费
- `streamGenerateContent` 返回结构和当前统一抽象不一致
- thinking 在 Gemini 2.5 上的 token 开销明显偏高，后续应单独控制，不适合这次顺手改语义

## 5. 当前推荐方案

### 当前实际默认方案

默认推荐：

```yaml
llm:
  provider: "vertex_native"
  auth_mode: "adc"
  api_key: ""
  base_url: ""
  model: "gemini-3-flash-preview"
  location: "global"
  project_id: "vertex-490203"
  credentials_path: "/home/gzy/.ssh/vat_vertex.json"
```

原因：

- 真实最小调用已验证可用
- `LLMTranslator.translate_subtitle()` 已验证可写出 `translated.srt`
- 同一套配置也能从 CLI `vat process -s translate` 进入真实翻译阶段
- 相比 Vertex API key，更符合当前项目的实际认证要求

### 备用方案：保留 OpenAI-compatible

如果后续找到别的 OpenAI-style 站点、模型中转服务，或重新回到某个兼容端点，可切回：

```yaml
llm:
  provider: "openai_compatible"
  auth_mode: "api_key"
  api_key: "${YOUR_API_KEY}"
  base_url: "https://your-openai-style-endpoint/v1"
  model: "gemini-3-flash-preview"
```

## 6. 本次验证

### 单元与集成测试

执行：

```bash
HOME=/tmp pytest tests/test_llm_client_vertex.py tests/test_config.py tests/test_translator_error_handling.py tests/test_vertex_translation_flow.py -q
```

结果：

- `39 passed`

### 真实流程验证

实际跑通了两条最小翻译流程：

- API key
  - 输入：`おはようございます`
  - 输出：`早上好`
  - 产物：`/tmp/vat-e2e-api-key/translated.srt`
- ADC
  - 输入：`こんばんは`
  - 输出：`晚上好`
  - 产物：`/tmp/vat-e2e-adc/translated.srt`

这说明当前改动已经能从 VAT 配置与翻译器调用链一路走到真实字幕文件产出。

### 真实 Vertex ADC 补充验证

后续追加的真实环境验证结果：

- 当前仓库默认配置当时仍是 `openai_compatible`，并未真正切到 Vertex
- 最小真实调用：
  - `vertex_native + api_key`：失败，`401 Unauthorized`
  - `vertex_native + adc`：成功
- 真实翻译器调用：
  - `LLMTranslator.translate_subtitle()` 在 `vertex_native + adc + gemini-3-flash-preview` 下成功
- 贴近真实 pipeline 的 CLI canary：
  - `python -m vat -c /tmp/vat-vertex-adc-test.yaml process -v e11fsGDFB-E -s translate`
  - 能进入真实 `translate` 阶段并调用 Vertex
  - 但长视频翻译过程中出现 `429`、空响应、TLS 握手超时，说明**链路已打通，但稳定性仍需收敛**

## 7. Benchmark 说明

现有的 `scripts/translation_benchmark.py` 现在已经支持：

- `gemini-3-flash`
  - 走 Google AI Studio 的官方 OpenAI-compatible 接口
- `gemini-3-flash-vertex`
  - 走 Google Vertex 的官方 Vertex Native 接口

因此现在可以直接在同一套 benchmark 流程里，对比：

- 旧的 Google 官方接入方式
- 新的 Google 官方接入方式

这次实际用 `gemini-3-flash-vertex` 跑过多视频 benchmark，接口可用，翻译质量整体与历史 `gemini-3-flash` 基线接近，但不是逐句完全一致，更接近“同模型不同官方入口下的自然改写”。

## 8. 后续建议

### 8.1 当前并发建议

AI Studio / OpenAI-compatible 路线下，`thread_num=10` 曾经可以稳定使用；  
但 Vertex ADC 的 `gemini-3-flash-preview` 在当前项目里并不适合直接沿用这个值。

本地最小压测结果：

- 并发 `1/3/5`：稳定成功
- 并发 `10`：仍能最终成功，但会出现大量 `429` 重试，尾请求耗时显著拉长

而真实长视频翻译时，`thread_num=10` + 多批次会进一步放大：

- `429 Rate Limit Error`
- `Invalid OpenAI API response: empty choices or content`
- `Vertex API 网络请求失败: _ssl.c:980: The handshake operation timed out`

因此当前默认建议：

- `translator.llm.thread_num: 3`

### 8.2 Vertex 是否支持多少并发

Google 官方文档并没有给出 Gemini 3 Flash 这种共享容量模式下的“固定支持 N 并发请求”数字。  
官方更强调的是：

- `429` 代表 quota / shared capacity 被打满
- 若需要更稳定、可预测的吞吐，应使用 **Provisioned Throughput**

也就是说：

- 默认 pay-as-you-go / shared capacity 模式下，没有一个可以直接写死到仓库里的官方“支持 10 并发”数字
- 对当前项目而言，应以实测和保守配置为准，而不是沿用 AI Studio 的经验值

### 8.3 官方更推荐的方式

如果只是当前仓库继续用 Gemini 3 Flash：

- 认证：`vertex_native + adc`
- 区域：`global`
- 并发：保守控制在 `3`

如果未来对高吞吐有硬要求：

- 官方更推荐考虑 **Provisioned Throughput**
- 它比共享容量模式更适合稳定的大批量调用

### 8.4 其他代码层建议

- 给 Vertex 增加更细的空响应日志：把 `promptFeedback` / `finishReason` 记录出来
- 如果后续 Web UI 有流式显示需求，再单独评估 `streamGenerateContent`
- 若后续需要统计成本或调试 token 消耗，再考虑把 `usageMetadata` 暴露出来
