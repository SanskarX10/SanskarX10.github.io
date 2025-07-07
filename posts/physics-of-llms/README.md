# Physics of LLMS

- decompose intelligence into building blocks (structures, knowledge, reasoning)
- study in controlled, idealized environments
- highly repeatable, small experiments
- probing techniques to see inner workings?

---

# ==Knowledge?==

* knowledge extraction: pretrain on n biographies, finetune on questions and answers for N/2 individuals, and test on remaining N/2 individuals 
* need to knowledge-augment the pretrain data
* pretrain + finetune is not equal to knowledge extraction
* pretrain + knowledge augmentation >>
* why does this loss of accuracy happen without knowledge augmentation?
* do we need to augment everybody for this controlled experiment?
* more stored knowledge for celebrities helped generalize minorities who have less data
* models like Bert or DeBERTa fail to do this even with knowledge augmentation
* knowledge manipulation is impossible without CoT
* need to include CoT in both training data and CoT at inference (different CoT from reasoning)
* knowledge inverse search is impossible  
* Knowledge Capacity Scaling Laws
* knowledge is measured in bits (entropy / information theory)
* to create synthetic data for biographies with K attributes, just an example  
* This formula gives total information in bits in that dataset:

    $\log_{2} \left( \frac{N_{0}!}{N!(N_{0}-N)!} \right) + NK \log_{2} C + K \log_{2} D + K \log_{2} \left( \frac{T!}{L!(T-L)!} \right)$

    * **N:** Number of distinct names in the synthetic data  
    * **N0:** Total number of possible names  
    * **K:** Number of knowledge attributes (e.g., birthday, birth city, etc.)  
    * **T:** Vocabulary size (total number of unique words in the data)  
    * **C:** Number of values for each attribute  
    * **L:** Length of each value (e.g., number of characters in a name)  
    * **D:** Value diversity (a measure of how spread out the values are)  

* all LLMs can achieve 2 bits/param in storing knowledge that is seen for 1000 exposures (exposure ≠ epochs; exposure is like 1,000,000 occurrences of "US, capital, DC" in 1 epoch of whole internet data)
* 1 bit/param if exposure is 100 times (insufficient training / not enough occurrence); if you fix knowledge and increase data/model size, you cannot learn more—before reaching that point, 2 bit/param is followed; works for Llama, Mistral, GPT-2
* for rare knowledge or 100-exposure data (insufficient training), GPT-2 works fine but Llama and Mistral knowledge retained is 1.3x less (in bits) (for knowledge capacity only, only for rare knowledge)
* Gated MLP is the reason for worse performance in Llama and Mistral
* | ---- good data ---- | -------------------- junk data ---------------------- |
* good -------------> 20x worse performance in knowledge retention in good + junk
* how to fix this? Add a domain token in front of each piece of pretrain data (adding URL); training becomes 10x better with junk; LLMs can automatically detect which sources are good
* 2 bit/params is also valid for int8 quantization

---

# ==hidden reasoning process==

* GSM is too contaminated
* remove common sense like (if a candle burns then length shrinks)
* iGSM (imaginary) -> capture:
    * direct dependency (ex: a = 5 * x * y)
    * instance dependency (ex: x classroom has y students)
    * implicit dependency (ex: a is 3x > b, b is 3a + 2, etc.)
    * Structure Graph
    * dependency graph
    * problem description (models decide what to compute first)
    * op = number of operations

* it helps in CoT as variables can be dependent on each other
* training GPT-2 rotary on hard and medium set of iGSM
* what are solution templates?
* GPT-2 achieves level-1 reasoning:
    * level-0: brute force, compute all params
    * level-1: uses topological sort + gives shortest CoT
        * model must "process stuff mentally"
        * should know necessary and unnecessary things in solving
    * level-2: it computes all pair dependency graph before the questions is asked (not needed to solve the problems) (somehow GPT-2 also developed this skill)

* language models can learn to solve math problems
* models know what factors are important to solve or what factors are dependent on each other, what to compute next
* how do models make mistakes?
    * computing unnecessary factors (you can tell the mistake beforehand by looking into model states with probing)
    * try computing a factor beforehand

* for reasoning, depth of the model matters (they claimed)
* for the next computation step (important in reasoning)
* here, parameters mean factors; this cannot be mitigated using CoT, even before CoT you need to think for the next step to compute
* one way to improve reasoning:
    * model knows it has made reasoning mistakes
    * pretrain with mistakes and corrections
    * beam-search/finetune => no accuracy gain
    * pretrain with fake mistakes?
* model knows it has made reasoning mistakes
    * can easily finetune for error detection
    * detection only won't solve this
    * beam search doesn't give any improvement in this case
    * data with mistakes + corrections, insert a "| back |" token preceded by probability (?? of what) in mistakes so that before computation of next it knows what mistakes have happened
    * this gives huge gains
    * higher p => better improvement
    * mistakes in pretrain ≠ mistakes in inference
    * label masking is not needed
    * during inference time, it gives short solution, doesn't create long reasoning chains
    * how to obtain such data?
        * dumber idea: try to create fake mistakes, insert future sentences earlier with |back|
        * smarter idea: try to create a mistake using factors as mistakes
        * dumber > smarter
    * how to train with this type of data?
        * a model with math pretrain with no mistake + finetuning model with math data with mistake and correction => doesn't improve the model but makes mistakes instead
        * adding retry data in pretrain is important; finetune is not going to make a difference; error correction should be learned from pretrain

---

# ==Hierarchical Language Structure:==

* they generated synthetic CFGs
* if we train a model on this CFG:
    * given valid prefix,
        * result 1: accuracy  GPT (relative attention) > GPT (rotary embed) > GPT (positional)
        * result 2: diversity
        * result 3: distribution

* why RoPE? For learning structure (i -> j) it's important to learn distance between two |i - j|
* uniform version of attention is important for learning structures
* encoder-based models use masked modeling so they can't learn above