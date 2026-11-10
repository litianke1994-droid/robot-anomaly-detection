# Parquet 读取环境修复（2026-10-18）

## 问题与证据

运行 `01_read_data.py` 时，pandas 无法加载 Parquet 引擎。
当前系统 Python 已安装 pyarrow 26.0.0；直接导入 `pyarrow.parquet` 可复现错误。
Windows CodeIntegrity 日志事件 3077 / 3033 明确指向该版本的 `pyarrow/parquet.dll`。
`VerifiedAndReputablePolicyState=1`，对应开启的智能应用控制。
因此当前阻塞点是原生库加载时的应用控制拦截。

## 修复

从官方 PyPI 获取 pyarrow 25.0.0 并先做兼容性验证，通过后在项目中新建 `.venv`。
该虚拟环境使用 Python 3.12.10，启用 `--system-site-packages`，复用电脑现有的 pandas 和 numpy；
在环境内部单独安装 pyarrow 25.0.0。它不是所有依赖都独立安装的完整隔离环境。
系统 Python 的 pyarrow 26.0.0 保留；未修改 Windows 安全设置、用户脚本和原始数据。
读取组件版本记录于项目的 `requirements-reader.txt`。

## 验证

使用 `.venv/Scripts/python.exe` 直接运行用户原有的 `01_read_data.py`，退出码为 0：

```text
数据的行数和列数：
(2321690, 4)
前十行数据：
   sample      time  anomaly  motor_iq_1
0       0  0.000000     True    0.452609
1       0  0.010047     True    0.446010
2       0  0.020135     True    0.452627
```

## 后续运行

在项目目录的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe .\01_read_data.py
```

VS Code 运行按钮需要选择本项目 `.venv/Scripts/python.exe` 作为解释器。
这条命令直接使用解释器，不需要激活脚本或更改 PowerShell 执行策略。

如需复建同样的环境（仍需已有 pandas / numpy 的基础 Python）：

```powershell
& 'D:\06_Software\python\python_312\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-reader.txt
```

当前验证只针对本机此项目的读取流程，不推断其他机器的应用控制结果。

来源：

- https://pypi.org/project/pyarrow/25.0.0/
- https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting
