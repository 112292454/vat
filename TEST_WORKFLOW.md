# VAT 测试工作流

## 准备工作

1. 确保配置正确：
```bash
cat config/default.yaml | grep -A 5 "api_key"
```

2. 测试配置加载：
```bash
python -c "from vat.config import load_config; c=load_config(); print('✅ 配置OK')"
```

## 完整流程测试

### 方案 1：使用真实 YouTube 视频

```bash
# 短视频测试（推荐先用短视频测试）
vat pipeline --url "https://www.youtube.com/watch?v=SHORT_VIDEO_ID"

# 观察输出日志，确认各步骤正常：
# [下载] → [Whisper ASR] → [智能断句] → [字幕优化] → [翻译] → [嵌入]
```

### 方案 2：测试缓存机制

```bash
# 首次运行
vat asr --video-id EXISTING_VIDEO_ID
# 记录时间

# 修改配置
vim config/default.yaml
# 修改 asr.split.max_words_cjk: 24 -> 30

# 再次运行（应该跳过 Whisper）
vat asr --video-id EXISTING_VIDEO_ID
# 确认日志显示 "复用 Whisper 缓存"
```

### 方案 3：分步测试

```bash
# 1. 下载
vat download --url "URL"

# 2. 转录（含智能断句）
vat asr --all

# 检查输出文件
ls data/videos/*/original*.srt
# 应该看到：original_raw.srt, original_split.srt, original.srt

# 3. 翻译（含字幕优化）
vat translate --all

# 4. 嵌入
vat embed --all

# 5. 检查最终输出
ls data/videos/*/final.mkv
```

## 验证要点

- [ ] 配置加载正常
- [ ] Whisper 识别成功
- [ ] 智能断句生效（对比 original_raw.srt 和 original_split.srt）
- [ ] 翻译质量正常
- [ ] 缓存机制工作（修改参数后跳过 Whisper）
- [ ] 最终视频可播放且字幕正确

## 问题排查

如果遇到问题：

1. 查看日志：`tail -f vat.log`
2. 检查配置：确认嵌套结构正确
3. 测试 API：`curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"`
4. 清理重试：`rm -rf data/videos/VIDEO_ID && vat pipeline --url URL`
