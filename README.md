# Self-Alignment for Factuality

To implement **Self-Knowledge Tuning (SK-Tuning)**, we utilize the **Direct Preference Optimization (DPO)** algorithm to **stabilize** the training process and enhance an LLM's self-knowledge awareness. To support the research community, we have provided pre-training data, which is publicly available at the following link:  

🔗 [Pre-Training Data](https://drive.google.com/file/d/1B18Aax0hAXPUhKmHtOmn1r9JdJYY2XzN/view?usp=sharing)  

If you find our dataset useful, please cite our work using the following reference:  

```bibtex
@inproceedings{zhang-etal-2024-self,
    title = "Self-Alignment for Factuality: Mitigating Hallucinations in {LLM}s via Self-Evaluation",
    author = "Zhang, Xiaoying  and
      Peng, Baolin  and
      Tian, Ye  and
      Zhou, Jingyan  and
      Jin, Lifeng  and
      Song, Linfeng  and
      Mi, Haitao  and
      Meng, Helen",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.107/",
    doi = "10.18653/v1/2024.acl-long.107",
    pages = "1946--1965",
    abstract = "Despite showing impressive abilities, large language models (LLMs) often struggle with factual inaccuracies, i.e., {\textquotedblright}hallucinations{\textquotedblright}, even when they hold relevant knowledge. To mitigate these hallucinations, current approaches typically necessitate high-quality human factuality annotations. In this work, we explore Self-Alignment for Factuality, where we leverage the self-evaluation capability of an LLM to provide training signals that steer the model towards factuality. Specifically, we incorporate Self-Eval, a self-evaluation component, to prompt an LLM to validate the factuality of its own generated responses solely based on its internal knowledge. Additionally, we design Self-Knowledge Tuning (SK-Tuning) to augment the LLM`s self-evaluation ability by improving the model`s confidence estimation and calibration. We then utilize these self-annotated responses to fine-tune the model via Direct Preference Optimization algorithm. We show that the proposed self-alignment approach substantially enhances factual accuracy over Llama family models across three key knowledge-intensive tasks on TruthfulQA and BioGEN."
}
