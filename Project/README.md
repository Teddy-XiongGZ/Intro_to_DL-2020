# 基于半监督学习方法的因果关系抽取

## Main Idea:
- 如果一个句子含'因为/所以/because/so'，那么它就一定包含因果这样的逻辑关系。(inspired by Distant Supervision)
- 通过爬虫生成大量训练数据，再mask掉部分句子的显式关键词，从而生成含有显式关系+隐式关系的训练集
- 用自动生成的数据集进行预训练，再结合已有标注数据来做fine tuning(模型基于[BERT](https://arxiv.org/abs/1810.04805))
- 将训练好的模型在儿童语料库上测试，探究儿童对该逻辑的语言表达的发展情况

## Unlabelled Corpus


## Annotated Dataset
### Chinese Dataset
- [哈工大中文篇章关系语料](http://ir.hit.edu.cn/hit-cdtb/)
- [Chinese Discourse Treebank 0.5](https://catalog.ldc.upenn.edu/LDC2014T21) (需注册)

### English Dataset
- [BECAUSE](https://github.com/duncanka/BECAUSE)

## Children's Corpus
- [CHILDES](https://childes.talkbank.org/)

## Related Research
- [The BECauSE Corpus 2.0: Annotating Causality and Overlapping Relations](https://www.aclweb.org/anthology/W17-0812/)
- [Automatic Extraction of Causal Relations from Text using Linguistically Informed Deep Neural Networks](https://www.aclweb.org/anthology/W18-5035/)
- [Causal Relation Extraction](http://lrec-conf.org/proceedings/lrec2008/pdf/87_paper.pdf)
- [CATENA: CAusal and TEmporal relation extraction from NAtural language texts](https://www.aclweb.org/anthology/C16-1007.pdf)

## Other Materials
- [Causal Corpus 事件因果关系语料统计](https://blog.csdn.net/gao2628688/article/details/96228855)
- [Corpus-Based Language Studies](https://www.lancaster.ac.uk/fass/projects/corpus/)
- [自然语言处理之中英语料库](https://blog.csdn.net/zeng_xiangt/article/details/81572317)
- [Choice of Plausible Alternatives(COPA)](https://people.ict.usc.edu/~gordon/copa.html)
