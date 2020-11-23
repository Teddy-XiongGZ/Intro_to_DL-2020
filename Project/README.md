# 基于半监督学习方法的因果关系抽取

## Main Idea:
- 如果一个句子含'因为/所以/because/so'，那么它就一定包含因果这样的逻辑关系。(inspired by Distant Supervision)
- 通过爬虫生成大量训练数据，再mask掉部分句子的显式关键词，从而生成含有显式关系+隐式关系的训练集
- 用自动生成的数据集进行预训练，再结合已有标注数据来做fine tuning(模型基于[BERT](https://arxiv.org/abs/1810.04805))
- 将训练好的模型在儿童语料库上测试，探究儿童对该逻辑的语言表达的发展情况

## Annotated Dataset
### Chinese Dataset
- [哈工大中文篇章关系语料](http://ir.hit.edu.cn/hit-cdtb/)
- [Chinese Discourse Treebank 0.5](https://catalog.ldc.upenn.edu/LDC2014T21) (需注册)

### English Dataset
- [BECAUSE](https://github.com/duncanka/BECAUSE)

## Children's Corpus
- [CHILDES](https://childes.talkbank.org/)

## Related Research
- [Automatic Extraction of Causal Relations from Natural Language Texts: A Comprehensive Survey](https://arxiv.org/pdf/1605.07895.pdf) 2016
- [The BECauSE Corpus 2.0: Annotating Causality and Overlapping Relations](https://www.aclweb.org/anthology/W17-0812/) ACL2017
- [Automatic Detection of Causal Relations for Question Answering](https://www.aclweb.org/anthology/W03-1210.pdf) ACL2003
- [Automatic Extraction of Causal Relations from Text using Linguistically Informed Deep Neural Networks](https://www.aclweb.org/anthology/W18-5035/) ACL2018
- [Causal Relation Extraction](http://lrec-conf.org/proceedings/lrec2008/pdf/87_paper.pdf) LREC2008
- [CATENA: CAusal and TEmporal relation extraction from NAtural language texts](https://www.aclweb.org/anthology/C16-1007.pdf) ACL2016
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v3.pdf) JMLR2020 [T5](https://huggingface.co/transformers/model_doc/t5.html)
- [Identifying Causal Relations Using Parallel Wikipedia Articles](http://www.cs.columbia.edu/nlp/papers/2016/hidey_wikipedia_causality_acl2016.pdf) ACL2016

## Other Materials
- [Causal Corpus 事件因果关系语料统计](https://blog.csdn.net/gao2628688/article/details/96228855)
- [Corpus-Based Language Studies](https://www.lancaster.ac.uk/fass/projects/corpus/)
- [自然语言处理之中英语料库](https://blog.csdn.net/zeng_xiangt/article/details/81572317)
- [Choice of Plausible Alternatives(COPA)](https://people.ict.usc.edu/~gordon/copa.html)
- [Github Search](https://github.com/search?o=desc&q=Chinese+NLP&s=stars&type=Repositories)
  - for instance [大规模语料，如250万篇新闻](https://github.com/brightmart/nlp_chinese_corpus)
  - or meta-datasets like [this](https://github.com/InsaneLife/ChineseNLPCorpus)
