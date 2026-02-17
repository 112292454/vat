# B站合集（SEASON）API 使用说明

## 概述

B站的"合集"分为两种类型：
- **SEASON（新版合集）**：每个视频独立计算数据和推荐，适合搬运/翻译场景。本项目使用此类型。
- **SERIES（旧版列表/分P）**：多个视频合并为一个推荐单位，流量弱，不适用。

本项目通过逆向 B站创作中心前端 JS 发现了正确的 API 调用格式，实现了合集的自动管理。

## 功能状态

| 功能 | 状态 | 方法 |
|------|------|------|
| 添加视频到合集 | ✅ 可用 | `add_to_season(aid, season_id)` |
| 从合集移除视频 | ✅ 可用 | `remove_from_season(aids, season_id)` |
| 获取合集视频列表 | ✅ 可用 | `get_season_episodes(season_id)` |
| 获取合集列表 | ✅ 可用 | `list_seasons()` |
| 合集内视频排序 | ✅ 可用 | `sort_season_episodes(season_id, aids_in_order)` |
| 创建合集 | ⚠ 未验证 | `create_season()` — 代码已实现但未实测 |

## 配置方式

### 全局配置

在 `config/upload.yaml` 中设置：

```yaml
bilibili:
  season_id: 7320255  # B站合集ID（从创作中心合集管理页面的URL获取）
```

### Per-Playlist 配置

在 WebUI 的 Playlist 详情页 → "上传配置" 中设置 `season_id`，优先级高于全局配置。

### 优先级

`playlist.metadata.upload_config.season_id` > `config/upload.yaml → bilibili.season_id`

## 上传流程中的合集集成

上传阶段（executor `_run_upload`）的合集处理流程：

1. 视频通过 `biliup` 库上传到 B站，获得 `bvid`
2. 如果配置了 `season_id`：
   - 通过 `bvid_to_aid` 将 BV号转为 AV号（含重试，因为新视频可能有索引延迟）
   - 调用 `add_to_season(aid, season_id)` 添加到合集
   - 成功则在视频 metadata 中记录 `bilibili_season_id`
   - 失败则记录 processing_note，不影响上传结果

## API 技术细节

### 核心概念

- **season_id**：合集ID，从创作中心 URL 或 `list_seasons()` 获取
- **section_id**：合集下的"分区"ID，每个合集至少有一个默认分区。通过 `get_season_episodes()` 获取
- **episode_id**：视频在合集中的内部ID（不是 aid），从 `get_season_episodes()` 返回的每个 episode 的 `id` 字段获取
- **csrf / bili_jct**：从 cookie 中获取的 CSRF token

### 添加视频到合集

```
POST /x2/creative/web/season/section/episodes/add?csrf={bili_jct}
Content-Type: application/json; charset=UTF-8
Referer: https://member.bilibili.com/platform/upload-manager/article/season

{
    "sectionId": 8081933,           // 注意驼峰命名
    "episodes": [{
        "title": "视频标题",
        "aid": 116084510294347,
        "cid": 36118727074,         // 从 /x/web-interface/view 获取
        "charging_pay": 0
    }],
    "csrf": "bili_jct_value"        // csrf 同时在 query string 和 body 中
}
```

**关键点**：
- 参数名是 `sectionId`（驼峰），不是 `id` 或 `section_id`
- `episodes` 是对象数组（含 aid/cid/title），不是 aid 整数数组
- 使用错误格式（如 `{id, aids}`）会返回 code=0 但不生效（"空成功"）

### 从合集移除视频

```
POST /x2/creative/web/season/section/episode/del
Content-Type: application/x-www-form-urlencoded

id={episode_id}&csrf={bili_jct}
```

**关键点**：
- 端点是 `episode/del`（单数），不是 `episodes/del`
- 参数 `id` 是 episode_id（合集内部ID），不是 aid
- 使用 form-encoded，不是 JSON
- 每次只能删除一个 episode

### 合集内排序

```
POST /x2/creative/web/season/section/edit?csrf={bili_jct}
Content-Type: application/json

{
    "section": {
        "id": 8081933,           // section_id（必须）
        "type": 1,               // 固定值 1（必须）
        "seasonId": 7320255,     // season_id（必须）
        "title": "正片"          // section 标题（必须）
    },
    "sorts": [
        {"id": episode_id_1, "sort": 1},
        {"id": episode_id_2, "sort": 2}
    ],
    "captcha_token": ""
}
```

**关键点**：
- `section` 对象必须包含 `id`、`type`、`seasonId`、`title` 四个字段，缺少任何一个都返回 -400
- body 中**不能**包含 `csrf`（只在 query string 中），否则返回 -400
- `sorts` 必须包含合集中的**所有**视频，不能只传部分
- 通过浏览器 F12 抓包发现此格式，之前仅传 `{title}` 导致持续失败

## 如何获取 season_id

1. 登录 B站创作中心：https://member.bilibili.com/platform/upload-manager/ep
2. 点击目标合集，URL 中的数字即为 season_id
3. 或通过代码：`BilibiliUploader.list_seasons()` 返回所有合集及其 ID

## 排查问题

- **添加后合集数量没变**：检查是否使用了正确的 `season_id`（不是 `section_id`）
- **bvid_to_aid 返回 None**：新上传的视频可能需要几秒到几分钟才能被 B站索引，executor 已内置重试
- **Cookie 过期**：重新运行 `scripts/bilibili_login.py` 获取新 cookie
