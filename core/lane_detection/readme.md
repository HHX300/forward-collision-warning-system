## ✅ 方法一：使用 `pip install` 安装本地包

这是最推荐、最干净的方法。

### 👉 步骤：

1. 在你的包目录中写一个 `setup.py` 文件（如果没有的话）：

   ```
   python复制编辑from setuptools import setup, find_packages
   
   setup(
       name='mylib',
       version='0.1',
       packages=find_packages(),
       install_requires=[],
   )
   ```

2. 使用 `pip` 安装你的包：

   - 永久安装（复制到 site-packages）：

     ```
     bash
     
     
     复制编辑
     pip install ~/my_python_libs/mylib
     ```

   - 或开发者模式安装（不会复制，而是直接指向源码）：

     ```
     bash
     
     
     复制编辑
     pip install -e ~/my_python_libs/mylib
     ```

     ✅ 修改源码后，其他项目会同步生效，适合开发阶段。