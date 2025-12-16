# 测试和代码质量检查指南

## 运行测试

> **注意**: 如果直接运行 `pytest` 命令找不到，请使用 `python -m pytest` 代替。

### 所有测试
```bash
cd backend
python -m pytest tests/ -v
```

或者（如果 pytest 在 PATH 中）：
```bash
cd backend
pytest tests/ -v
```

### 只运行单元测试
```bash
cd backend
python -m pytest tests/unit/ -v
```

### 只运行集成测试
```bash
cd backend
python -m pytest tests/integration/ -v
```

### 只运行负载测试
```bash
cd backend
python -m pytest tests/load/ -v -s
```

`-s` 参数用于显示测试中的打印输出（性能指标）。

### 运行特定测试文件
```bash
cd backend
python -m pytest tests/unit/test_storage.py -v
```

### 运行特定测试函数
```bash
cd backend
python -m pytest tests/unit/test_storage.py::test_create_job_with_unique_name -v
```

### 带代码覆盖率
```bash
cd backend
python -m pytest tests/ -v --cov=app --cov-report=html
```

查看覆盖率报告：
```bash
open htmlcov/index.html  # macOS
# 或
xdg-open htmlcov/index.html  # Linux
```

### 快速测试（简短输出）
```bash
cd backend
python -m pytest tests/ -q
```

## Lint 检查

### 检查代码质量（不修复）
```bash
cd /Users/cocktail/PycharmProjects/PythonProject
./backend/lint.sh
```

这个命令会运行：
- `ruff check` - 代码风格检查
- `black --check` - 代码格式化检查
- `mypy` - 类型检查

### 自动修复 Lint 错误
```bash
cd /Users/cocktail/PycharmProjects/PythonProject
./backend/lint.sh --fix
```

或者使用 `-f` 参数：
```bash
./backend/lint.sh -f
```

这个命令会：
- 自动修复 ruff 可以修复的问题
- 自动格式化代码（black）
- 运行 mypy 类型检查（mypy 不会自动修复，需要手动修复）

### 单独运行各个工具

#### Ruff（代码风格检查）
```bash
# 只检查
ruff check backend/

# 自动修复
ruff check --fix backend/
```

#### Black（代码格式化）
```bash
# 只检查格式
black --check backend/

# 自动格式化
black backend/
```

#### Mypy（类型检查）
```bash
mypy backend/app --ignore-missing-imports
```

## 完整工作流程

### 开发前检查
```bash
# 1. 运行所有测试
cd backend
python -m pytest tests/ -v

# 2. 检查代码质量
cd ..
./backend/lint.sh
```

### 提交代码前
```bash
# 1. 自动修复可以修复的问题
./backend/lint.sh --fix

# 2. 运行所有测试确保没有破坏功能
cd backend
python -m pytest tests/ -v

# 3. 检查代码覆盖率
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

## 负载测试

负载测试用于评估API在高并发情况下的性能表现。

### 运行所有负载测试
```bash
cd backend
python -m pytest tests/load/ -v -s
```

### 运行特定负载测试
```bash
cd backend
# 并发健康检查测试
python -m pytest tests/load/test_load.py::test_concurrent_health_checks -v -s

# 压力测试
python -m pytest tests/load/test_load.py::test_stress_test_health_endpoint -v -s

# 混合负载测试
python -m pytest tests/load/test_load.py::test_mixed_workload -v -s
```

### 负载测试包含的场景

1. **并发健康检查** (`test_concurrent_health_checks`)
   - 50个并发请求
   - 测试健康检查端点的吞吐量

2. **并发根端点** (`test_concurrent_root_endpoint`)
   - 100个并发请求
   - 测试根端点的性能

3. **并发任务状态查询** (`test_concurrent_job_status_queries`)
   - 50个并发查询请求
   - 测试任务状态查询的性能

4. **并发图片上传** (`test_concurrent_image_uploads`)
   - 20个并发上传，限制5个同时进行
   - 测试图片上传的并发处理能力

5. **混合负载** (`test_mixed_workload`)
   - 混合多种操作（健康检查、查询、上传）
   - 模拟真实使用场景

6. **压力测试** (`test_stress_test_health_endpoint`)
   - 200个请求，分批处理
   - 测试系统在高负载下的稳定性

### 性能指标

负载测试会输出以下性能指标：
- **总请求数**: 发送的请求总数
- **成功请求**: 成功响应的请求数
- **总耗时**: 完成所有请求的总时间
- **吞吐量**: 每秒处理的请求数（RPS）
- **成功率**: 成功请求的百分比

## 常见问题

### 测试失败
如果测试失败，查看详细错误信息：
```bash
python -m pytest tests/ -v --tb=long
```

### Lint 错误无法自动修复
某些 mypy 类型错误需要手动修复。查看错误信息：
```bash
mypy backend/app --ignore-missing-imports
```

### 覆盖率不足
当前要求覆盖率 >= 70%。查看哪些代码未覆盖：
```bash
python -m pytest tests/ --cov=app --cov-report=term-missing
```

