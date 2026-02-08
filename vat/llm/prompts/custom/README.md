# 自定义提示词使用说明

此目录用于存放 LLM 翻译和优化的自定义提示词文件。

## 目录结构

```
custom/
├── translate/          # 翻译自定义提示词
│   └── example.md      # 示例文件
├── optimize/           # 优化自定义提示词
│   └── example.md      # 示例文件
└── README.md           # 本文件
```

## 使用方法

### 1. 创建自定义提示词文件

在 `translate/` 或 `optimize/` 目录下创建 `.md` 文件，编写你的自定义提示词内容。

**示例**：`translate/fubuki.md`

```markdown
# 白上吹雪翻译要求

## 术语表
- ガチャ: 抽卡
- 配信: 直播
- スパチャ: SC（Super Chat）

## 翻译风格
- 保持吹雪的可爱语气
- 使用第一人称"我"
```

### 2. 在配置文件中引用

编辑 `config/default.yaml` 或 `config/config.yaml`：

```yaml
translator:
  llm:
    # 翻译自定义提示词
    custom_prompt: "translate/fubuki.md"
    
    optimize:
      # 优化自定义提示词
      custom_prompt: "optimize/fubuki.md"
```

**注意**：
- 文件路径相对于 `vat/llm/prompts/custom/` 目录
- 空字符串 `""` 表示不使用自定义提示词
- 文件必须存在，否则配置加载时会报错

### 3. 文件编码

所有提示词文件必须使用 **UTF-8** 编码。

## 最佳实践

1. **按项目/主播分类**：为不同的翻译项目创建不同的提示词文件
2. **包含术语表**：列出专有名词、游戏术语等的翻译规则
3. **说明翻译风格**：描述语气、人称、特殊要求等
4. **保持简洁**：提示词不宜过长，重点突出关键信息

## 示例场景

### 游戏直播翻译
```yaml
custom_prompt: "translate/gaming.md"
```

### 歌回翻译
```yaml
custom_prompt: "translate/singing.md"
```

### 聊天回翻译
```yaml
custom_prompt: "translate/chatting.md"
```
