# APPFD-FK-GMM Method for Surface-based Protein Domains Retrieval (SHREC 2021)
The APPFD-FK-GMM is a 3D shape retrieval method (algorithm) which involves the Agglomeration of Augmented Point-pair Features Descriptor with Fisher Kernel and Gaussian Mixture Model (APPFD-FK-GMM).


# [SHREC 2021, Surface-based Protein Domains Retrieval](http://shrec2021.drugdesign.fr/)
This repository presents the code-implementation of our novel 3D shape retrieval algorithm (method), the APPFD-FK-GMM. This method is applied for retrieval of 3D protein conformers, for the SHape REtrieval Contests (SHREC 2021) research tasks/challenges, Track 4. The aim of this ***Protein Domain Retrieval*** track is to assess the performance of shape retrieval methods on a dataset of related multi-domain protein surfaces.

* Further details regarding this retrieval track can be found **[here](http://shrec2021.drugdesign.fr/)**.


## [OUR TEAM](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
1. Dr. Ekpo Otu (eko@aber.ac.uk)
2. Prof. Reyer(rrz@aber.ac.uk)
3. Prof. Yonguai (liuyo@edgehill.ac.uk)
4. Dr. David (dah56@aber.ac.uk)


## [1. Introduction](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
This repository contains the following code-implementation (Python scripts) and a [Jupyter notebook](#) demonstrating some essential steps. We try as much as possible to keep things simple and straightforward, to the best of our abilities. Please feel free to contact: [Dr. Ekpo Otu](eko@aber.ac.uk) for any issue or concern regarding this work.

* [local_appfd_method.py](#)
* [ekpoUtilities.py](#)
* [fisher_vector.py](#)
* [SHREC2021_ProteinDomainsRetrieval_Track4.ipynb](#)

> The [local_appfd_method.py](#) contains our baseline algorithm that extracts 6-dimensional hand-crafted local geometric features from the surface of 3D point cloud and computes local Augmented Point-pair Feature Descriptors (APPFD) for Local Surface Patches (LSPs) around a keypoint (or interest point). Some of the supporting functions needed by this algorithm (local APPFD) are presented in [ekpoUtilities.py](#). After we have computed local APPFD 3D shape descriptors for each 3D model in the dataset, these descriptors are agglomerated using a fisher-kernel framework, implemented by a function in [fisher_vector.py](#) - following the training and fitting of a Gaussian Mixture Model (GMM) on combined locally-computed APPFDs for all database objects/models.

> Additionally, we present a Jupyter notebook: [SHREC2021_ProteinDomainsRetrieval_Track4.ipynb](#) to demonstrate some essential steps in this implementation and explain the stages on the APPFD algorithm, as well as the APPFD-FK-GMM method. **[Click Here](#)** to access the Notebook.
> 

## [2. The Research Problem](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
According to [[1](http://shrec2021.drugdesign.fr/)], proteins are primarily made of two domains (the structural as and functional sub-units of proteins), or more, which can exist independently of the rest of the proteins, and are the level at which protein interactions and functions are studied. To compare proteins at the domain level for similarities is a common task in structural biology, biochemistry or drug discovery. Proteins can be described as non-rigid surfaces representing their solvent-excluded surface (SES) as defined by Connoly (Connoly et al., J Appl Cryst. 1983). Additional, biologically-relevant information can be provided, such as electrostatics, to further describe these molecular shapes.

> The above track proposes a set of representation for the conformational space of 10 query domains, extracted from the PFAM database (El-Geabli et al., NAR, 2019) as well as 554 surfaces of multi-domain proteins. Compared to the previous Protein Shape Retrieval contests, this track aims to focus on the evaluation of the performance to retrieve 10 individual domains among a set of 554 multi-domains protein surfaces.

> Ten individual domains involved in protein-protein (7 domains) or protein-DNA (3 domains) were extracted from the PFAM database, and a representative structure of each of these domains were be provided to the participants as query for the retrieval task.


## [3. Dataset](http://shrec2021.drugdesign.fr/)
603 single-domain or multi-domain protein surfaces were provided to the participants, in two versions : a shape-only file and shape+electrostatics file (provided as .ply files). Each protein will includes at least one of the query domain, meaning that several proteins will match several queries. Additional details regarding the dataset for this retrieval challenge can be obtained **[HERE](http://shrec2021.drugdesign.fr/)**


## [4. Research Tasks](http://shrec2021.drugdesign.fr/)
Participants for this retrieval challenge were asked to produce a distance-to-the-query dissimilarity matrix, using either the shape-only, the shape+electrostatics or both versions for each query. The participants are expected to return their results as a distance matrix file in binary format.

> In addition, research participants are expected to provide runtimes and hardware specifications for their calculations since it is a critical information for processing large datasets, notably in this particular context of molecular surfaces.


## [5. Ground Truth and Evaluation](http://shrec2021.drugdesign.fr/)
The ground truth is derived from PFAM database classification; only the family level of the database are used to generate the ground truth, and will be analyzed for the final report.

Standard metrics of previous shape retrieval experiments will be used: precision - recall (PR) evaluation, Nearest Neighbor (NN), first-tier (FT), second-tier (ST), mean average precision (mAP), and confusion matrix.


## [6. Our Implementation](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
We approach this retrieval challenge by first computing a novel, knowledged-based 3D shape descriptor, the Augmented Point-pair Features Descriptor (APPFD) which is based on the extraction of 6-dimensional local hand-crafted features from keypoint regions (i.e. Local Surface Patches - LSP or regions) on the surface of a given 3D protein model. The final APPFD involves bucketing each of the 6-dimensional locally-extracted features into a 1-dimensional histogram, where number of bins, ***B = 35***. The resulting APPFD for each keypoint or LSP is a ***210-dimensional*** vector. Next, we agglomerate all computed local APPFD into a single compact code or representation, using the Fisher-Kernel (FK) and Gaussian Mixture Model (GMM) framework. The final 3D shape descriptor we present for each input 3D protein model is therefore a novel 3D shape descriptor called APPFD-FK-GMM, which is a 1-dimensional feature vector (***fv***) - compact and highly discriminative against different protein domains.

> We provide a simple and straightforward implementation pipeline for the APPFD-FK-GMM 3D shape retrieval/classification method in this **[Jupyter notebook](#)**.

> Comparing of two different protein models or domains, is reduced to finding the spatial distance/(dis)similarity between their final APPFD-FK-GMM descriptors.

This repository, therefore contains full code implementation of the APPFD-FK-GMM 3D shape retrieval (and classification) method. Detained description of the implementation are also provided below.

## [6. Implementation Guides](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
See this Jupyter notebook for a Step-by-Step implementation guidance on our novel 3D shape descriptor or retrieval method, called the APPFD-FK-GMM.

> The method we present here for this protein domains retrieval challenge are completely implemeted in Python 3.6. We strictly adopt the FOP (Functional Oriented Programming) coding style for all functions and algorithms presented here.

## [7. References](https://github.com/KoksiHub/APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval)
TBC...
