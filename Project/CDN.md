# 使用CDN来存放数据集

在bash中运行以下代码：
```bash
git clone git@github.com:IntRGit/misc.git
cd misc
git checkout gh-pages
```
该Github repo的根目录和`node0.static.jsonx.ml`的根目录建立了映射关系。  
比如，有个文件地址为./copa/copa.xml，则可以通过`node0.static.jsonx.ml/copa/copa.xml直接访问或下载该文件。

注：如果文件大小超过100M，需要git lfs扩展。