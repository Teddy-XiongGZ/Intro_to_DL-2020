# 基于半监督学习方法的因果关系提取

## Main Idea:
如果一个句子含'因为/所以/because/so'，那么它就一定包含因果这样的逻辑关系。(inspired by Distant Supervision)\\
通过爬虫生成大量训练数据，再mask掉部分句子的显式关键词，从而生成含有显式关系+隐式关系的训练集
用自动生成的数据集进行预训练，再结合已有标注数据来做fine tuning
将训练好的模型在儿童语料库上测试，探究儿童对该逻辑的语言表达的发展情况

## Unlabelled Corpus


## Annotated Dataset
### Chinese Dataset
哈工大中文篇章关系语料 http://ir.hit.edu.cn/hit-cdtb/
Chinese Discourse Treebank 0.5 https://catalog.ldc.upenn.edu/LDC2014T21 (需注册)

### English Dataset
BECAUSE https://github.com/duncanka/BECAUSE

## Children's Corpus
CHILDES https://childes.talkbank.org/

## Related Research
