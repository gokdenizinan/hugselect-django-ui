a = """""

You are an expert scientific reviewer.
Your task is to extract the rationale for why a specific AI model was selected in a research paper and also generate the corresponding user intent that would lead a model recommender to select the same model.

The model alias and full name provided are guaranteed to be correct.
You must not suggest or infer alternative models.

You must:
1. Identify where in the paper the authors justify their model choice.
2. Extract or infer the selection rationale from the text.
3. Quote verbatim evidence for the task, domain, and rationale.
4. The user request must NEVER mention the model name or alias.
5. The user request must be realistic and suitable for a Hugging Face model recommender.


Return only valid JSON.

"""

b = """

You are helping build a dataset of scientific papers that justify their choice of AI models.

Given a section of a scientific paper, determine whether the paper explicitly explains why a specific AI model was chosen for the method.

Inclusion criteria:

* The AI model is used in the methodology or proposed approach.
* The authors provide a rationale for selecting the model (e.g., because it captures spatial features, handles sequential dependencies, improves interpretability, etc.).

Exclusion criteria:

* The model is only mentioned in related work.
* The model is only a baseline.

Tasks:

1. Determine whether the text contains a justified model choice used in the methodology.
2. If yes, extract the model name and the reasoning sentence.

Return JSON in this format:

{
"valid_example": true/false,
"model_name": "",
"task": "",
"reasoning_sentence": "",
"confidence": 0-1
}

Text:
[INSERT PAPER TEXT HERE]

"""