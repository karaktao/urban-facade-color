# Urban Facade Color

街景建筑分割、去阴影、提取主色并拼接色卡的工具集，提供 Gradio 前端与 FastAPI 后端接口。

## 安装依赖

```bash
pip install -r requirements.txt
```

确保根目录存在 SegFormer ADE20K 权重文件（`segformer_mit-b0_*ade20k*.pth`）。

## 运行 FastAPI 后端

启动服务（默认 0.0.0.0:8000）：

```bash
bash start_backend.sh
```

POST `/analyze` 接收图片文件并返回去背景+色卡的 PNG（base64）和主色比例：

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@images/example.jpg"
```

示例响应：

```json
{
  "image_png_base64": "iVBORw0KGgoAAA...",
  "colors": [
    {"rgb": [188, 162, 138], "ratio": 0.42},
    {"rgb": [120, 106, 98], "ratio": 0.24}
  ]
}
```

## 运行 Gradio Demo

仍然共用同一套模型与管线：

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 上传图片可视化结果。

## 推理流程

1. 加载 SegFormer 模型并选出建筑相关类别 ID。
2. 语义分割生成建筑掩码；抠出建筑并通过 LAB 阴影检测将阴影区域透明。
3. 在建筑前景内用 KMeans 提取主色，过滤极黑/极白区域。
4. 将色卡（右侧不透明）与左侧 BGRA 结果拼接，生成 PNG 输出与颜色比例列表。