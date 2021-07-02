# APPFD_FK_GMM-Method-For-SHREC-2021-Surface-based-Protein-Domains-Retrieval
3D Shape Retrieval method (algorithm) called the Agglomeration of Augmented Point-pair Features Descriptor with Fisher Kernel and Gaussian Mixture Model (APPFD-FK-GMM).


# [SHREC 2021, Surface-based Protein Domains Retrieval](http://shrec2021.drugdesign.fr/)
This repository presents the code-implementation of our novel 3D shape retrieval algorithm (method), the APPFD-FK-GMM. This method is applied for retrieval of 3D protein conformers, for the SHape REtrieval Contests (SHREC 2021) research tasks/challenges, Track 4. The aim of this ***Protein Domain Retrieval*** track is to assess the performance of shape retrieval methods on a dataset of related multi-domain protein surfaces.

* Further details regarding this retrieval track can be found **[here](http://shrec2021.drugdesign.fr/)**.


### The Research Problem
According to [1](http://shrec2021.drugdesign.fr/), proteins are primarily made of two domains (the structural as and functional sub-units of proteins), or more, which can exist independently of the rest of the proteins, and are the level at which protein interactions and functions are studied. To compare proteins at the domain level for similarities is a common task in structural biology, biochemistry or drug discovery. Proteins can be described as non-rigid surfaces representing their solvent-excluded surface (SES) as defined by Connoly (Connoly et al., J Appl Cryst. 1983). Additional, biologically-relevant information can be provided, such as electrostatics, to further describe these molecular shapes.

The above track proposes a set of representation for the conformational space of 10 query domains, extracted from the PFAM database (El-Geabli et al., NAR, 2019) as well as 554 surfaces of multi-domain proteins. Compared to the previous Protein Shape Retrieval contests, this track aims to focus on the evaluation of the performance to retrieve 10 individual domains among a set of 554 multi-domains protein surfaces.

Ten individual domains involved in protein-protein (7 domains) or protein-DNA (3 domains) were extracted from the PFAM database, and a representative structure of each of these domains were be provided to the participants as query for the retrieval task.


### [Dataset](http://shrec2021.drugdesign.fr/)
603 single-domain or multi-domain protein surfaces were provided to the participants, in two versions : a shape-only file and shape+electrostatics file (provided as .ply files). Each protein will includes least one of the query domain, meaning that several proteins will match several queries.

The structures were retrieved and protonated using propka (Sondergaard et al., JCTC, 2011; Olsson et al., JCTC, 2011). All solvent-excluded surfaces (SES) are calculated using EDTSurf (Xu et al., Plos One, 2009; atomic partial charges were computed using APBS (Jurrus et al., Protein Sci, 2018).

Additional details regarding the dataset for this retrieval challenge can be obtained **[HERE](http://shrec2021.drugdesign.fr/)**


### [Research Tasks](http://shrec2021.drugdesign.fr/)
The participants were asked to produce a distance-to-the-query dissimilarity matrix, using either the shape-only, the shape+electrostatics or both versions for each query. 
The participants are expected to return their results as a distance matrix file in binary format.

Research participants are expected to provide runtimes and hardware specifications for their calculations since it is a critical information for processing large datasets, notably in this particular context of molecular surfaces.


### [Ground Truth and Evaluation](http://shrec2021.drugdesign.fr/)
The ground truth is derived from PFAM database classification; only the family level of the database are used to generate the ground truth, and will be analyzed for the final report.

Standard metrics of previous shape retrieval experiments will be used: precision - recall (PR) evaluation, Nearest Neighbor (NN), first-tier (FT), second-tier (ST), mean average precision (mAP), and confusion matrix.


### Our Implementation
We approach this retrieval challenge by first computing a novel, knowledged-based 3D shape descriptor, the Augmented Point-pair Features Descriptor (APPFD) which is based on hand-crafted local features, extracted from keypoint regions (i.e. Local Surface Patches - LSP or regions) on the surface of a given 3D protein model. The final APPFD here involves discritizing each of the 6-dimensional locally-extracted features into a 1-dimensional binning, where number of bind, $B = 35$. Next, we agglomerate all computed local APPFD into a single compact code or representation, using the Fisher-Kernel (FK) and Gaussian Mixture Model (GMM) framework. The final 3D shape descriptor we present for each input 3D protein model is therefore a novel 3D shape descriptor called APPFD-FK-GMM, which is a 1-dimensional feature vector ($fv$) - compact and highly discriminative against different protein domains.

Comparing of two different protein models or domains, is reduced to finding the spatial distance/(dis)similarity between their final APPFD-FK-GMM descriptors.

This repository, therefore contains full code implementation of the APPFD-FK-GMM 3D shape retrieval (and classification) method. Detained description of the implementation are also provided below.

### Additional notes
TBC...

The method we present here for this protein domains retrieval challenge are completely implemeted in Python 3.6. We strictly adopt the FOP (Functional Oriented Programming) coding style for all functions and algorithms presented here.

### References
TBC...
