# SpeechJudge HTTP API 接口说明

基于 FastAPI 的单条语音自然度评分服务：进程常驻、模型预加载在 GPU，通过 HTTP 提交音频与参考文本，返回 1–10 分及可选明细。

**默认文档：** 服务启动后可在浏览器打开 `http://<host>:<port>/docs` 查看 Swagger（OpenAPI）。

---

## 启动与环境变量

```bash
cd infer
uvicorn api_service:app --host 0.0.0.0 --port 8000
```

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `SPEECHJUDGE_MODEL_PATH` | 模型目录 | `pretrained/SpeechJudge-GRM` |
| `SPEECHJUDGE_CUDA_DEVICE` | GPU 序号，如 `0` | 由框架自动选择 |
| `SPEECHJUDGE_THINKER` | 设为 `1` / `true` / `yes` 时使用 thinker 模式 | 关闭 |

---

## 评分模式 `mode` / `analysis`

- **`mode`** 取值：`fast` | `compact` | `analysis`（可选；不传时由 **`analysis`** 布尔推导：`analysis=true` → `analysis`，否则 → **`compact`**）。
- **`compact`（默认）**：结构化子分项 + 总分，生成上限约 64 tokens（与 VRAM 自动上限取 min）。
- **`fast`**：尽量只出总分，约 16 tokens，延迟更低。
- **`analysis`**：带分析文本再出分，最慢；`max_new_tokens` 可用自动上限（通常更大）。

同一次请求内 **`mode` 优先**：若显式传 `mode`，则不再仅由 `analysis` 决定模式。

**并发：** 推理在进程内用互斥锁串行执行；多请求会排队，不会在同一 GPU 上并行多路生成。

---

## `GET /health`

探活与运行时信息。

**响应示例（就绪）：**

```json
{
  "status": "ready",
  "model_path": "/root/models/SpeechJudge-GRM",
  "cuda_device": 0,
  "auto_max_new_tokens": 768,
  "loaded_at": 1710000000.0,
  "default_mode": "compact",
  "recommended_request_max_new_tokens": {
    "fast": 16,
    "compact": 64,
    "analysis": 256
  },
  "model_dtype": "torch.bfloat16"
}
```

模型仍在加载时：`{"status": "loading"}`。

---

## `POST /score-path`

**服务端可读到的本地路径**上音频评分（适合机器人在同一台机或挂载卷上读文件）。

- **Content-Type:** `application/json`

**请求体：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_path` | string | 是 | 音频文件绝对或相对路径（服务端文件系统） |
| `target_text` | string | 是 | 期望朗读/参考文本，非空 |
| `max_new_tokens` | int \| null | 否 | 覆盖解码长度上限；默认按 GPU 自动 |
| `analysis` | bool | 否 | 默认 `false`；为 `true` 且未指定 `mode` 时用 `analysis` 模式 |
| `mode` | string | 否 | `fast` / `compact` / `analysis` |

**成功：** `200`，JSON 见下文「统一响应字段」。

**错误：**

| HTTP | 说明 |
|------|------|
| `404` | 音频文件不存在 |
| `500` | 其它运行时/模型错误，`detail` 为错误信息 |

---

## `POST /score-url`

根据 **公网 HTTP(S) URL** 下载音频，在服务端 **转码为 WAV** 后评分；**临时文件在请求结束后删除**，不落库。

- **Content-Type:** `application/json`

**请求体：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_url` | string | 是 | 必须以 `http://` 或 `https://` 开头 |
| `target_text` | string | 是 | 期望文本 |
| `max_new_tokens` | int \| null | 否 | 同 `/score-path` |
| `analysis` | bool | 否 | 同 `/score-path` |
| `mode` | string | 否 | 同 `/score-path` |

**成功：** `200`，在统一响应基础上额外包含：

- `audio_url`：请求的 URL  
- `downloaded_audio_format`：下载文件的扩展名  
- `server_transcoded_to_wav`：`true`  

**错误：** 多为 `500`（下载失败、解码失败、模型错误等），`detail` 为错误字符串。下载超时为 **120 秒**。

---

## `POST /score-upload`

**multipart/form-data** 上传音频文件，写入临时文件评分后删除。

**表单字段：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | file | 是 | 音频文件 |
| `target_text` | string | 是 | 期望文本 |
| `max_new_tokens` | int | 否 | 可空 |
| `analysis` | bool | 否 | 表单里常用字符串 `"true"` / `"false"` |
| `mode` | string | 否 | `fast` / `compact` / `analysis` |

**成功：** `200`，JSON 同统一响应（无 URL 相关附加字段）。

**错误：** `500` + `detail`。

---

## 统一成功响应 JSON（字段说明）

各评分接口成功时结构一致（`/score-url` 多几个字段，见上）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | number | 总分（通常 1–10，具体以模型解析为准） |
| `reason` | string | 便于直接展示的“理由/依据”摘要：默认 **`compact`** 由 `sub_scores` 拼接；其它模式优先取 `raw_response`，否则回退到 `details`；过长会截断 |
| `details` | object \| null | `analysis` 模式：与模型输出相关的明细；`compact` 时与 `sub_scores` 同源语义；`fast` 常为 `null` |
| `sub_scores` | object \| null | **仅 `compact`**：结构化子分项；其它模式多为 `null` |
| `raw_response` | string \| null | 模型原始文本片段（依模式不同） |
| `audio_path` | string | 实际用于推理的文件路径（上传/URL 为服务端临时路径） |
| `target_text` | string | 回显请求中的参考文本 |
| `max_new_tokens` | int \| null | 本请求实际采用的生成上限（经模式裁剪后） |
| `audio_format` | string | 文件扩展名，如 `.wav` |
| `wav_recommended` | bool | 非 WAV 时建议转 WAV（路径接口）；URL 接口已转码 |
| `analysis` | bool | 是否分析模式 |
| `mode` | string | 实际使用的模式 |

---

## 调用示例

### cURL — 本地路径

```bash
curl -s -X POST "http://127.0.0.1:8000/score-path" \
  -H "Content-Type: application/json" \
  -d '{"audio_path":"/data/sample.wav","target_text":"Hello world.","mode":"compact"}'
```

### cURL — 音频 URL

```bash
curl -s -X POST "http://127.0.0.1:8000/score-url" \
  -H "Content-Type: application/json" \
  -d '{"audio_url":"https://example.com/audio.mp3","target_text":"Bonjour.","mode":"compact"}'
```

### cURL — 上传文件

```bash
curl -s -X POST "http://127.0.0.1:8000/score-upload" \
  -F "audio=@./sample.wav" \
  -F "target_text=Hello world." \
  -F "mode=compact"
```

---

## 依赖说明（服务端）

URL 下载与转码依赖 **`librosa`**、**`soundfile`**（见仓库根目录 `requirements.txt`）。仅使用 `/score-path` 且输入已为服务端可解码格式时，仍建议安装完整依赖以与部署一致。

---

## 批量客户端（参考）

仓库内脚本（非 HTTP 接口本体）：

- `infer/batch_score_upload_with_log.py` — 调 `/score-upload`，支持 manifest / 目录  
- `infer/batch_score_url_with_log.py` — 调 `/score-url`，manifest 需含 `url` 列  

详见脚本内 `--help`。
