These files comprise the overall steps taken to create and evaluate teh HugSelect recommender system for Hugging Face models. 

1 - is the collection and filtering of the initial 2 million HF models to around 70k useful models.

2 - is the noun phrase extraction performed on the descriptions of the retained 70k models.

4 - is the llm based evaluation of the noun phrases and retaining a semantically rich set of functional features.

5 - is the collection and filtering of reviews about models.

6 - is the mapping of review sentimenst to ISO/IEC2500 to reatin a semantically rich quantitative information of models.

7 - is the clustering of models into modality, task, model_family, model_root groups.

8 - is the acutalization of the recsystem with all of the collected information from the pipelines. 
  - unified knowledgebase, user intent extraction module, mcdm based weighting of criteria, experimentation with weights.
  - evaluationnof results against the case study dataset.

10 - is the evaluation of functional feature extraction pipeline

11 - is the creation of case study dataset for recommendation evaluation.

12 - is the evaluation of the quality attribute pipeline (sentiment analysis of reviews and quality mapping of reviews).

13 - statistical analysis and cool graphs.
