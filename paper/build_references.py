"""Writes an IEEE-formatted thebibliography block into main.tex.

The references live inside main.tex (no BibTeX step), so the paper compiles from
the .tex file alone. Entries are numbered in order of first citation, as IEEE
style requires. Every entry was checked against Crossref, the arXiv API or the
publisher page; see REFERENCE_CHECK.md. Run after editing the text:

    python paper/build_references.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")

REFS = {
    "xu2018stocknet": r"Y.~Xu and S.~B. Cohen, ``Stock movement prediction from tweets and historical prices,'' in \emph{Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers)}, 2018, pp. 1970--1979, doi: 10.18653/v1/P18-1183.",
    "feng2019advalstm": r"F.~Feng, H.~Chen, X.~He, J.~Ding, M.~Sun, and T.-S. Chua, ``Enhancing stock movement prediction with adversarial training,'' in \emph{Proc. 28th Int. Joint Conf. Artif. Intell. (IJCAI)}, 2019, pp. 5843--5849, doi: 10.24963/ijcai.2019/810.",
    "sawhney2020mansf": r"R.~Sawhney, S.~Agarwal, A.~Wadhwa, and R.~R. Shah, ``Deep attentive learning for stock movement prediction from social media text and company correlations,'' in \emph{Proc. 2020 Conf. Empirical Methods Natural Lang. Process. (EMNLP)}, 2020, pp. 8415--8426, doi: 10.18653/v1/2020.emnlp-main.676.",
    "sawhney2021fast": r"R.~Sawhney, A.~Wadhwa, S.~Agarwal, and R.~R. Shah, ``FAST: Financial news and tweet based time aware network for stock trading,'' in \emph{Proc. 16th Conf. Eur. Chapter Assoc. Comput. Linguistics: Main Volume}, 2021, pp. 2164--2175, doi: 10.18653/v1/2021.eacl-main.185.",
    "ang2022guided": r"G.~Ang and E.-P. Lim, ``Guided attention multimodal multitask financial forecasting with inter-company relationships and global and local news,'' in \emph{Proc. 60th Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers)}, 2022, pp. 6313--6326, doi: 10.18653/v1/2022.acl-long.437.",
    "chen2018gcn": r"Y.~Chen, Z.~Wei, and X.~Huang, ``Incorporating corporation relationship via graph convolutional neural networks for stock price prediction,'' in \emph{Proc. 27th ACM Int. Conf. Inf. Knowl. Manage. (CIKM)}, 2018, pp. 1655--1658, doi: 10.1145/3269206.3269269.",
    "feng2019rsr": r"F.~Feng, X.~He, X.~Wang, C.~Luo, Y.~Liu, and T.-S. Chua, ``Temporal relational ranking for stock prediction,'' \emph{ACM Trans. Inf. Syst.}, vol.~37, no.~2, pp. 1--30, 2019, doi: 10.1145/3309547.",
    "cheng2022multimodality": r"D.~Cheng, F.~Yang, S.~Xiang, and J.~Liu, ``Financial time series forecasting with multi-modality graph neural network,'' \emph{Pattern Recognit.}, vol.~121, Art. no.~108218, 2022, doi: 10.1016/j.patcog.2021.108218.",
    "wang2021review": r"J.~Wang, S.~Zhang, Y.~Xiao, and R.~Song, ``A review on graph neural network methods in financial applications,'' 2021, \emph{arXiv:2111.15367}.",
    "yoo2021dtml": r"J.~Yoo, Y.~Soun, Y.-C. Park, and U.~Kang, ``Accurate multivariate stock movement prediction via data-axis transformer with multi-level contexts,'' in \emph{Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining}, 2021, pp. 2037--2045, doi: 10.1145/3447548.3467297.",
    "zhang2025camef": r"Y.~Zhang, W.~Yang, J.~Wang, Q.~Ma, and J.~Xiong, ``CAMEF: Causal-augmented multi-modality event-driven financial forecasting by integrating time series patterns and salient macroeconomic announcements,'' in \emph{Proc. 31st ACM SIGKDD Conf. Knowl. Discovery Data Mining V.2}, 2025, pp. 3867--3878, doi: 10.1145/3711896.3736872.",
    "zhang2025gnnsentiment": r"P.~Zhang, R.~D.~F. Harris, and J.~Zheng, ``GNN-based social media sentiment analysis for stock market forecasting and trading,'' \emph{Expert Syst. Appl.}, vol.~291, Art. no.~128425, 2025, doi: 10.1016/j.eswa.2025.128425.",
    "nguyen2025dasf": r"N.-H. Nguyen, T.-T. Nguyen, and Q.~T. Ngo, ``DASF-Net: A multimodal framework for stock price forecasting with diffusion-based graph learning and optimized sentiment fusion,'' \emph{J. Risk Financial Manag.}, vol.~18, no.~8, Art. no.~417, 2025, doi: 10.3390/jrfm18080417.",
    "qayyum2025news": r"A.~Qayyum, ``News sentiment embeddings for stock price forecasting,'' 2025, \emph{arXiv:2507.01970}.",
    "harit2024breaking": r"A.~Harit, Z.~Sun, J.~Yu, and N.~Al Moubayed, ``Breaking down financial news impact: A novel AI approach with geometric hypergraphs,'' 2024, \emph{arXiv:2409.00438}.",
    "dukkipati2025predictive": r"A.~Dukkipati, K.~Mayilvaghanan, N.~K. Pallekonda, S.~P. Hadnoor, and R.~S. Ayyagari, ``Predictive AI with external knowledge infusion: Datasets and benchmarks for stock markets,'' 2025, \emph{arXiv:2504.20058}.",
    "li2024finreport": r"X.~Li, X.~Shen, Y.~Zeng, X.~Xing, and J.~Xu, ``FinReport: Explainable stock earnings forecasting via news factor analyzing model,'' in \emph{Companion Proc. ACM Web Conf. 2024}, 2024, pp. 319--327, doi: 10.1145/3589335.3648330.",
    "liu2025multimodal": r"Y.~Liu, Z.~Wei, L.~Chen, C.~Xu, and Z.~Guan, ``Multi-modal temporal dynamic graph construction for stock rank prediction,'' \emph{Mathematics}, vol.~13, no.~5, Art. no.~845, 2025, doi: 10.3390/math13050845.",
    "hu2025differential": r"L.~Hu and Q.~Wang, ``A study of dynamic stock relationship modeling and S\&P500 price forecasting based on differential graph transformer,'' 2025, \emph{arXiv:2506.18717}.",
    "niu2025ngat": r"Y.~Niu, M.~Zhao, V.~Poti, and R.~Dong, ``NGAT: A node-level graph attention network for long-term stock prediction,'' 2025, \emph{arXiv:2507.02018}.",
    "sonani2025hybrid": r"M.~S. Sonani, A.~Badii, and A.~Moin, ``Stock price prediction using a hybrid LSTM-GNN model: Integrating time-series and graph-based analysis,'' 2025, \emph{arXiv:2502.15813}.",
    "zhu2025msgformer": r"M.~Zhu, H.~Qi, S.~Ni, and Y.~Liu, ``MSGformer: A hybrid multi-scale graph--transformer architecture for unified short- and long-term financial time series forecasting,'' \emph{Electronics}, vol.~14, no.~12, Art. no.~2457, 2025, doi: 10.3390/electronics14122457.",
    "kumar2024dynamic": r"P.~N. Kumar, N.~Umeorah, and A.~Alochukwu, ``Dynamic graph neural networks for enhanced volatility prediction in financial markets,'' 2024, \emph{arXiv:2410.16858}.",
    "li2018deeper": r"Q.~Li, Z.~Han, and X.-M. Wu, ``Deeper insights into graph convolutional networks for semi-supervised learning,'' in \emph{Proc. AAAI Conf. Artif. Intell.}, vol.~32, no.~1, 2018, doi: 10.1609/aaai.v32i1.11604.",
    "li2024master": r"T.~Li, Z.~Liu, Y.~Shen, X.~Wang, H.~Chen, and S.~Huang, ``MASTER: Market-guided stock transformer for stock price forecasting,'' in \emph{Proc. AAAI Conf. Artif. Intell.}, vol.~38, no.~1, 2024, pp. 162--170, doi: 10.1609/aaai.v38i1.27767.",
    "mcmahan2017fedavg": r"B.~McMahan, E.~Moore, D.~Ramage, S.~Hampson, and B.~Aguera y Arcas, ``Communication-efficient learning of deep networks from decentralized data,'' in \emph{Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)}, ser. Proc. Mach. Learn. Res., vol.~54, 2017, pp. 1273--1282.",
    "kumarappan2024federated": r"J.~Kumarappan, E.~Rajasekar, S.~Vairavasundaram, K.~Kotecha, and A.~Kulkarni, ``Federated learning enhanced MLP--LSTM modeling in an integrated deep learning pipeline for stock market prediction,'' \emph{Int. J. Comput. Intell. Syst.}, vol.~17, no.~1, Art. no.~267, 2024, doi: 10.1007/s44196-024-00680-9.",
    "noseda2025federated": r"M.~Noseda, A.~De Luca, L.~Von Briel, and N.~Lacour, ``Federated learning for financial forecasting,'' 2025, \emph{arXiv:2509.16393}.",
    "zhong2025financial": r"Z.~Zhong, B.~Wang, and Z.~Qi, ``A financial multimodal sentiment analysis model based on federated learning,'' in \emph{Proc. 2025 6th Int. Conf. Manage. Sci. Eng. Manage. (ICMSEM 2025)}, ser. Atlantis Highlights in Economics, Business and Management, 2025, pp. 845--853, doi: 10.2991/978-94-6463-845-5\_84.",
    "tan2025graphprivacy": r"Z.~Tan, J.~Zheng, K.~Q. Zhang, and W.~Wang, ``Graph privacy: A heterogeneous federated GNN for trans-border financial data circulation,'' 2025, \emph{arXiv:2505.00257}.",
    "xia2025fingraphfl": r"Z.~Xia and S.~C. Saha, ``FinGraphFL: Financial graph-based federated learning for enhanced credit card fraud detection,'' \emph{Mathematics}, vol.~13, no.~9, Art. no.~1396, 2025, doi: 10.3390/math13091396.",
    "aljunaid2025secure": r"S.~K. Aljunaid, S.~J. Almheiri, H.~Dawood, and M.~A. Khan, ``Secure and transparent banking: Explainable AI-driven federated learning model for financial fraud detection,'' \emph{J. Risk Financial Manag.}, vol.~18, no.~4, Art. no.~179, 2025, doi: 10.3390/jrfm18040179.",
    "matthews1975": r"B.~W. Matthews, ``Comparison of the predicted and observed secondary structure of T4 phage lysozyme,'' \emph{Biochim. Biophys. Acta (BBA)---Protein Struct.}, vol.~405, no.~2, pp. 442--451, 1975, doi: 10.1016/0005-2795(75)90109-9.",
    "chicco2020mcc": r"D.~Chicco and G.~Jurman, ``The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,'' \emph{BMC Genomics}, vol.~21, no.~1, Art. no.~6, 2020, doi: 10.1186/s12864-019-6413-7.",
    "henderson2018": r"P.~Henderson, R.~Islam, P.~Bachman, J.~Pineau, D.~Precup, and D.~Meger, ``Deep reinforcement learning that matters,'' in \emph{Proc. AAAI Conf. Artif. Intell.}, vol.~32, no.~1, 2018, doi: 10.1609/aaai.v32i1.11694.",
    "hochreiter1997lstm": r"S.~Hochreiter and J.~Schmidhuber, ``Long short-term memory,'' \emph{Neural Comput.}, vol.~9, no.~8, pp. 1735--1780, 1997, doi: 10.1162/neco.1997.9.8.1735.",
    "burges2005ranknet": r"C.~Burges, T.~Shaked, E.~Renshaw, A.~Lazier, M.~Deeds, N.~Hamilton, and G.~Hullender, ``Learning to rank using gradient descent,'' in \emph{Proc. 22nd Int. Conf. Mach. Learn. (ICML)}, 2005, pp. 89--96, doi: 10.1145/1102351.1102363.",
    "kingma2015adam": r"D.~P. Kingma and J.~Ba, ``Adam: A method for stochastic optimization,'' in \emph{Proc. 3rd Int. Conf. Learn. Represent. (ICLR)}, San Diego, CA, USA, 2015.",
    "welch1947": r"B.~L. Welch, ``The generalization of `Student's' problem when several different population variances are involved,'' \emph{Biometrika}, vol.~34, no.~1--2, pp. 28--35, 1947, doi: 10.1093/biomet/34.1-2.28.",
    "kunsch1989": r"H.~R. K{\"u}nsch, ``The jackknife and the bootstrap for general stationary observations,'' \emph{Ann. Statist.}, vol.~17, no.~3, pp. 1217--1241, 1989, doi: 10.1214/aos/1176347265.",
    # added in review round 5
    "holm1979": r"S.~Holm, ``A simple sequentially rejective multiple test procedure,'' \emph{Scand. J. Statist.}, vol.~6, no.~2, pp. 65--70, 1979, doi: 10.2307/4615733.",
    "grinold2000active": r"R.~C. Grinold and R.~N. Kahn, \emph{Active Portfolio Management}, 2nd~ed. New York, NY, USA: McGraw-Hill, 2000.",
    "nosek2018": r"B.~A. Nosek, C.~R. Ebersole, A.~C. DeHaven, and D.~T. Mellor, ``The preregistration revolution,'' \emph{Proc. Nat. Acad. Sci.}, vol.~115, no.~11, pp. 2600--2606, 2018, doi: 10.1073/pnas.1708274114.",
    # added in review round 2 (verified: arXiv listing / Microsoft Research page; PMLR v37 page)
    "tang2020add": r"H.~Tang, L.~Wu, W.~Liu, and J.~Bian, ``ADD: Augmented disentanglement distillation framework for improving stock trend forecasting,'' 2020, \emph{arXiv:2012.06289}.",
    "ganin2015grl": r"Y.~Ganin and V.~Lempitsky, ``Unsupervised domain adaptation by backpropagation,'' in \emph{Proc. 32nd Int. Conf. Mach. Learn. (ICML)}, ser. Proc. Mach. Learn. Res., vol.~37, 2015, pp. 1180--1189.",
}


def main():
    tex = open(TEX, encoding="utf-8").read()
    body = tex.split("% REFERENCES-START")[0]
    order = []
    for group in re.findall(r"\\cite\{([^}]*)\}", body):
        for key in (k.strip() for k in group.split(",")):
            if key not in order:
                order.append(key)
    missing = [k for k in order if k not in REFS]
    unused = [k for k in REFS if k not in order]
    if missing:
        raise SystemExit(f"cited but not defined: {missing}")
    if unused:
        print("defined but not cited (left out):", unused)
    block = ["% REFERENCES-START (generated by paper/build_references.py; ordered by first citation)",
             "\\begin{thebibliography}{00}"]
    block += [f"\\bibitem{{{k}}} {REFS[k]}" for k in order]
    block += ["\\end{thebibliography}", "% REFERENCES-END"]
    new = re.sub(r"% REFERENCES-START.*?% REFERENCES-END", lambda _: "\n".join(block),
                 tex, flags=re.S)
    open(TEX, "w", encoding="utf-8", newline="\n").write(new)
    print(f"{len(order)} references written in citation order")


if __name__ == "__main__":
    main()
