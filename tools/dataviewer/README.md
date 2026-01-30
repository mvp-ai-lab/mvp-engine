# WebDataset Viewer

一个用于可视化和浏览 WebDataset 数据的工具。

## 功能特性

- 🔍 加载并浏览 WebDataset tar 文件
- 🖼️ 显示 RGB 图像和深度图
- 📊 展示样本元数据和统计信息
- ⌨️ 支持键盘导航（左右箭头键）
- 🎨 美观的 Web 界面
- 📦 支持多种数据格式

## 安装

```bash
# 进入 dataviewer 目录
cd tools/dataviewer

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 方式一：命令行启动并指定数据集路径

```bash
python app.py --dataset-path ./data/potato_v1/data --port 5000
```

### 方式二：启动服务器后在网页中指定路径

```bash
python app.py --port 5000
```

然后在浏览器中打开 `http://127.0.0.1:5000`，在输入框中输入数据集路径。

### 命令行参数

- `--dataset-path`: WebDataset 数据集目录路径（默认: `./data/potato_v1/data`）
- `--port`: 服务器端口号（默认: 5000）
- `--host`: 服务器主机地址（默认: 127.0.0.1）
- `--debug`: 启用调试模式

## 数据格式要求

WebDataset 应包含以下内容的 tar 文件：

- `images.*`: RGB 图像文件（支持 .jpg, .png 等格式）
- `depths.*`: 深度图文件（支持 .png, .exr 等格式）
- `meta.json`: 元数据文件（可选）

每个样本应该有一个唯一的 `__key__`。

## 界面操作

1. **加载数据集**: 在顶部输入框中输入数据集路径，点击 "Load Dataset" 按钮
2. **浏览样本**: 使用 "Previous" / "Next" 按钮或键盘左右箭头键导航
3. **查看信息**: 
   - 统计卡片显示数据集概况
   - 图像区域显示 RGB 和深度图
   - 元数据区域显示详细的样本信息

## 示例

```bash
# 启动服务器查看 potato_v1 数据集
python app.py --dataset-path ./data/potato_v1/data

# 在远程服务器上运行，允许外部访问
python app.py --host 0.0.0.0 --port 8080

# 开发模式（自动重载）
python app.py --debug
```

## API 端点

- `GET /`: 主页面
- `POST /api/init`: 初始化数据集
- `GET /api/sample/<index>`: 获取指定索引的样本
- `GET /api/stats`: 获取数据集统计信息

## 技术栈

- **后端**: Flask (Python)
- **前端**: HTML5 + CSS3 + JavaScript
- **数据处理**: WebDataset, NumPy, Pillow

## 故障排除

### 找不到 tar 文件
确保数据集路径正确，且包含 `.tar` 文件。

### 图像无法显示
检查 WebDataset 中的图像数据是否包含正确的键名（如 `images.jpg`, `depths.png`）。

### 内存不足
可以修改 `load_samples()` 中的 `num_samples` 参数来减少一次加载的样本数量。

## 许可证

MIT License
