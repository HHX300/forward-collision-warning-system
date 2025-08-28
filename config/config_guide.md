# 🚗 智能车道检测与碰撞预警系统 - 配置指南

## 📋 配置文件说明

本系统使用 `config.json` 文件来管理所有配置参数。配置文件采用 JSON 格式，包含以下主要配置节：

### 🔧 系统基本配置 (SYSTEM)

```json
"SYSTEM": {
    "APP_NAME": "🚗 智能车道检测与碰撞预警系统",  // 应用程序名称
    "VERSION": "2.0.0",                        // 版本号
    "DEBUG_MODE": false,                       // 调试模式
    "LOG_LEVEL": "INFO",                       // 日志级别
    "ENABLE_EMOJI": true,                      // 启用emoji图标
    "LANGUAGE": "zh_CN"                        // 界面语言
}
```

### 📹 视频处理配置 (VIDEO)

```json
"VIDEO": {
    "DEFAULT_FPS": 30,          // 默认帧率
    "FRAME_SKIP": 1,            // 跳帧处理（1=不跳帧）
    "RESIZE_WIDTH": 1280,       // 视频宽度
    "RESIZE_HEIGHT": 720,       // 视频高度
    "ENABLE_RESIZE": true       // 启用视频缩放
}
```

### 🛣️ 车道线检测配置 (LANE_DETECTION)

```json
"LANE_DETECTION": {
    "GAUSSIAN_BLUR_KERNEL": 5,     // 高斯模糊核大小
    "CANNY_LOW_THRESHOLD": 50,     // Canny边缘检测低阈值
    "CANNY_HIGH_THRESHOLD": 150,   // Canny边缘检测高阈值
    "HOUGH_THRESHOLD": 50,         // 霍夫变换阈值
    "MIN_LINE_LENGTH": 100,        // 最小线段长度
    "MAX_LINE_GAP": 50,            // 最大线段间隙
    "ROI_TOP_RATIO": 0.6,          // 感兴趣区域顶部比例
    "LANE_LINE_THICKNESS": 8       // 车道线粗细
}
```

### 🚗 车辆检测配置 (VEHICLE_DETECTION)

```json
"VEHICLE_DETECTION": {
    "CONFIDENCE_THRESHOLD": 0.5,   // 置信度阈值
    "NMS_THRESHOLD": 0.4,          // 非极大值抑制阈值
    "MIN_VEHICLE_AREA": 1000,      // 最小车辆面积
    "MAX_VEHICLE_AREA": 50000,     // 最大车辆面积
    "DETECTION_CLASSES": [2,3,5,7] // 检测的车辆类别ID
}
```

### 📏 距离计算配置 (DISTANCE_CALCULATION)

```json
"DISTANCE_CALCULATION": {
    "METHOD": "perspective",        // 计算方法
    "CAMERA_HEIGHT": 1.5,          // 摄像头高度（米）
    "CAMERA_ANGLE": 10,            // 摄像头俯仰角（度）
    "FOCAL_LENGTH": 800,           // 焦距（像素）
    "VEHICLE_AVERAGE_WIDTH": 1.8   // 平均车宽（米）
}
```

### ⚠️ 碰撞预警配置 (COLLISION_WARNING)

```json
"COLLISION_WARNING": {
    "SAFE_DISTANCE": 30.0,         // 安全距离（米）
    "WARNING_DISTANCE": 20.0,      // 预警距离（米）
    "DANGER_DISTANCE": 10.0,       // 危险距离（米）
    "CRITICAL_DISTANCE": 5.0,      // 紧急距离（米）
    "ENABLE_AUDIO_WARNING": true   // 启用音频预警
}
```

### 🎨 界面配置 (UI)

```json
"UI": {
    "WINDOW_WIDTH": 1400,          // 窗口宽度
    "WINDOW_HEIGHT": 900,          // 窗口高度
    "THEME": "modern_dark",        // 界面主题
    "ENABLE_ANIMATIONS": true,     // 启用动画效果
    "SHOW_FPS": true,              // 显示FPS
    "SHOW_VEHICLE_COUNT": true,    // 显示车辆数量
    "ENABLE_LOG_DISPLAY": true     // 启用日志显示
}
```

### 🎨 UI样式配置 (UI_STYLES)

```json
"UI_STYLES": {
    "PRIMARY_COLOR": "#3498db",     // 主色调
    "SUCCESS_COLOR": "#27ae60",     // 成功色
    "WARNING_COLOR": "#f39c12",     // 警告色
    "DANGER_COLOR": "#e74c3c",      // 危险色
    "BORDER_RADIUS": "8px",         // 圆角半径
    "BUTTON_HEIGHT": "40px"         // 按钮高度
}
```

## 🔧 配置使用方法

### 1. 基本使用

```python
from config import Config

# 创建配置实例
config = Config()

# 获取配置值
safe_distance = config.get('COLLISION_WARNING', 'SAFE_DISTANCE')
window_width = config.WINDOW_WIDTH  # 快捷访问

# 设置配置值
config.set('UI', 'THEME', 'modern_light')

# 保存配置
config.save_config()
```

### 2. 获取专用配置

```python
# 获取UI配置
ui_config = config.get_ui_config()

# 获取窗口配置
window_config = config.get_window_config()

# 获取主题颜色
theme_colors = config.get_theme_colors()

# 获取预警距离
warning_distances = config.get_warning_distances()
```

### 3. 配置验证

```python
# 验证配置有效性
if config.validate_config():
    print("配置验证通过")
else:
    print("配置验证失败")
```

### 4. 配置导入导出

```python
# 导出配置
config.export_config('backup_config.json')

# 导入配置
config.import_config('custom_config.json')

# 重置为默认配置
config.reset_to_default()
```

## 🎨 主题配置

系统支持三种主题：

1. **modern_dark** - 现代深色主题（默认）
2. **modern_light** - 现代浅色主题
3. **default** - 经典主题

修改主题：
```json
"UI": {
    "THEME": "modern_light"
}
```

## ⚙️ 性能优化配置

```json
"PERFORMANCE": {
    "ENABLE_GPU": false,           // 启用GPU加速
    "NUM_THREADS": 4,              // 处理线程数
    "MEMORY_LIMIT_MB": 1024,       // 内存限制
    "ENABLE_CACHING": true         // 启用缓存
}
```

## 📝 日志配置

```json
"LOGGING": {
    "ENABLE_LOGGING": true,        // 启用日志
    "LOG_FILE": "system.log",      // 日志文件名
    "LOG_MAX_SIZE_MB": 10,         // 日志文件最大大小
    "LOG_BACKUP_COUNT": 5          // 日志备份数量
}
```

## 🚨 注意事项

1. **配置文件格式**：必须是有效的JSON格式
2. **数值范围**：置信度阈值必须在0.0-1.0之间
3. **距离单位**：所有距离参数单位为米
4. **颜色格式**：BGR格式，如[0, 255, 255]表示黄色
5. **备份配置**：修改前建议备份原配置文件

## 🔄 配置更新

当系统更新时，新的配置参数会自动合并到现有配置中，不会覆盖用户的自定义设置。

## 📞 技术支持

如果在配置过程中遇到问题，请参考：
- 系统日志文件
- 配置验证结果
- 控制台错误信息

---

**提示**：修改配置后需要重启应用程序才能生效。