from EA_Features import UserQuery
import json

user_inputs = ["Give me an image classification model",
               "Show me well-maintained models for text classification in the medical domain, ideally labeled as clinical or biomedical. I want models that are popular or high-quality, with a good number of downloads or likes, and updated recently, rather than old or abandoned models. Exclude private or gated models.",
               "Find models for text generation created by Meta or closely related authors, preferably based on LLaMA or LLaMA 2, and licensed under permissive licenses like MIT or Apache-2. I don't want models that are gated or restricted, and I'm specifically looking for open, widely usable ones.",
                "Give me models designed for code or programming tasks, especially those built on top of Mistral, CodeLlama, or DeepSeek, and include advanced functional capabilities like reasoning, function calling, or tool use. Prefer models that have been downloaded frequently in the last month, showing active interest.",
                "Show models for image generation or diffusion, particularly in the fields of art or design, that are known for high performance and reliability. I’d like models that balance quality and popularity, have a strong download count, and are openly accessible rather than gated.",
                "Recommend small, efficient models for text classification that are optimized for speed and low resource usage. Prefer models with high performance efficiency, a small storage footprint, low file count, and compatibility with mainstream libraries like Transformers. Prioritize lightweight, production-friendly options.",
                "Foundation models with permissive commercial licenses (Apache-2.0, MIT, BSD)",
                "Instruction-tuned LLMs optimized for function calling and tool use",
                "Open-weight models benchmarked highly on reasoning tasks (MMLU, GSM8K, BIG-Bench)",
                "Sparse or Mixture-of-Experts (MoE) foundational models for efficient inference",
                "Foundation models suitable for continual learning or parameter-efficient fine-tuning (LoRA, QLoRA)",
                "Vision-language models pretrained for document understanding (PDFs, charts, tables)",
                "Time-series forecasting foundational models trained on multivariate sensor data",
                "Open models for mathematical reasoning and symbolic problem solving",
                "Small foundational models distilled from larger LLMs for fast inference",
                "Open-weight large language models (7B–70B) with strong zero-shot and few-shot reasoning performance, permissive commercial licensing, and documented fine-tuning recipes for downstream enterprise use.",
                "Multimodal foundation models that jointly encode text and images, support visual question answering and document understanding, and provide pretrained checkpoints suitable for domain adaptation via LoRA.",
                "Text embedding foundation models optimized for large-scale semantic search and RAG pipelines, with high MTEB scores, multilingual support (≥50 languages), and low-latency inference on CPU.",
                "Speech foundation models for automatic speech recognition trained on diverse accents and noisy environments, offering both streaming and offline inference modes with open-source weights.",
                "Efficient diffusion or transformer-based generative models for text-to-image synthesis that balance visual fidelity and inference speed, include safety filtering components, and can run on consumer-grade GPUs (≤8GB VRAM)."
            ]

with open("11-RECOMMENDATION_EVALUATION/OUTPUT_ZZZZ_VVV.json", "r") as f:
    rationale_input_full = json.load(f)

import os
import json

rationale_input = {}

# yes (target) ; the system is evaluated on this selected model
reasons = ["no", "introduced", "baseline", "comparison", 
           "experiment (benchmarkish)", "no (embeddings)", "no (backbone popular)", "no (evaluation)" ] # "no (backbone popular)", "component"  # Define the reasons to check for
for paper_key, paper in rationale_input_full.items():

    if not paper:
        continue
    if paper.get("manuel_check", "").lower() in reasons:
        continue  
    if paper.get("manuel_check", "").lower() == "":
        if paper.get("in approach", "").lower() == "no":
            continue  
    try:
        paper_id = paper_key.split("_")[1]
    except IndexError:
        continue

    eval_filename = f"eval_H{paper_id}"


    # keep valid item
    rationale_input[paper_key] = paper
# save new json

with open("11-RECOMMENDATION_EVALUATION/OUTPUT_F.json", "w") as f:
    json.dump(rationale_input, f, indent=4)

print(f"Saved {len(rationale_input)} papers to OUTPUT_F.json")