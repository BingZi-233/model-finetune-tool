# 更新日志 (Changelog)

> model-finetune-tool 版本更新记录

所有重大变更都会记录在此文件。

格式基于 [Keep a Changelog](https://keepachangelog.com/) 规范。

---

## [v0.1.0] - 2026-02-02

### 首次发布

这是一个初始版本，包含以下功能：

#### ✨ 新功能
- 📄 **文档解析** - 支持 Word (docx)、PDF、Markdown 格式
- 🤖 **LLM 集成** - 使用 OpenAI API 生成训练数据
- 💾 **数据管理** - SQLite 数据库缓存，支持 MySQL/PostgreSQL
- ⚡ **模型训练** - 基于 LoRA 的高效微调
- 📋 **CLI 工具** - 简洁的命令行界面
- 📊 **数据集管理** - 版本控制、导出功能

#### 🐛 修复
- 修复 Markdown 解析器重复代码问题
- 添加路径遍历攻击防护
- 添加输入验证（文件大小、文本长度）
- 完善数据库索引优化

#### 🔒 安全改进
- 路径规范化处理
- 输入长度限制
- 文件大小限制
- 数据库 SQL 注入防护

#### 🚀 性能优化
- 数据库查询索引
- 缓存管理（自动清理）
- 配置热重载支持

#### 🌐 兼容性
- **Windows 兼容** - 完整的 Windows 支持
- **路径处理** - 跨平台路径规范化
- **SQLite** - Windows 绝对路径修复

#### 📚 文档
- 详细设计文档 (`docs/design.md`)
- 完整用户手册 (`docs/user-manual.md`)
- Windows 安装指南 (`docs/windows-guide.md`)

#### 🛠️ 开发改进
- Pydantic 配置验证
- 统一日志记录
- 自定义异常类
- 完整的测试覆盖

---

## 待发布

### 计划中的功能
- 🐳 Docker 支持
- 🔄 并行处理
- 📦 批量 LLM 调用
- 📝 API 文档

### 已知问题
- 暂无

---

## 贡献者

感谢所有为这个项目贡献代码的人！

---

## 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- 项目仓库: https://github.com/BingZi-233/model-finetune-tool
- 问题反馈: https://github.com/BingZi-233/model-finetune-tool/issues
