# Makefile for CorpusToolkit
# 可使用：make build / make clean / make upload / make test-upload

# Python解释器（可自定义）
PYTHON := python

# 默认目标：构建 .whl
.PHONY: build
build:
	$(PYTHON) tools/build_wheel.py

# 清理构建缓存
.PHONY: clean
clean:
	rm -rf dist build *.egg-info

# 上传到正式 PyPI
.PHONY: upload
upload:
	twine upload dist/*

# 上传到 Test PyPI（测试用）
.PHONY: test-upload
test-upload:
	twine upload --repository testpypi dist/*

# 显示帮助信息
.PHONY: help
help:
	@echo "可用命令："
	@echo "  make build        构建 wheel 包"
	@echo "  make clean        清理构建产物"
	@echo "  make upload       上传到 PyPI"
	@echo "  make test-upload  上传到 TestPyPI"
