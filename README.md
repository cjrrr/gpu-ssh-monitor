# gpu-ssh-monitor

在本地一个终端里，实时查看所有可 SSH 服务器上的 NVIDIA GPU 状态。远端只需要已有的 `nvidia-smi`，不需要安装 agent。

## 安装

需要 Python 3.9+。本项目自带无需安装依赖的启动入口：

```bash
cd gpu-ssh-monitor
./gsm
```

如需在任意目录直接执行，可将入口链接到 PATH 中已有的目录：

```bash
ln -s "$(pwd)/gsm" ~/.local/bin/gsm
```

也支持使用新版 `pip install -e .` 进行标准安装。

## 使用

自动读取 `~/.ssh/config` 中不带通配符的 `Host`：

```bash
gsm
```

只看指定服务器：

```bash
gsm gpu-a gpu-b
```

常用选项：

```bash
gsm --match 'gpu|train'      # 按主机名筛选
gsm -i 1                    # 每秒刷新
gsm -1 gpu-a gpu-b          # 查询一次
gsm --json gpu-a            # JSON 输出，适合脚本接入
gsm -t 10 -p 8              # 连接超时 10 秒，最多并发 8 台
gsm -F ~/.ssh/work-config   # 使用另一份 SSH 配置
```

持续模式用 `Ctrl-C` 退出。主机不可达、认证失败或没有 `nvidia-smi` 时，错误会显示在该主机对应的行中，不影响其他服务器刷新。

如果多个 SSH 别名解析到相同的主机名和端口（例如只是 `User` 不同），工具会自动将它们归为同一台服务器，只使用配置中最先出现的别名执行一次 `nvidia-smi`。终端会显示被合并的别名，JSON 输出则在 `aliases` 字段中保留完整列表。

## 采集内容

- GPU 型号、利用率、显存占用
- 温度、功耗和功耗上限
- 每台服务器的查询延迟
- SSH 连接错误和超时

连接默认启用 `BatchMode`，不会在刷新过程中卡住等待密码输入。请先确保 `ssh <主机别名>` 可以通过密钥或 SSH agent 非交互登录。

## 开发测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
