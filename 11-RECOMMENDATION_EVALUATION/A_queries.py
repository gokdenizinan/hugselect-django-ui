
queries_100_broad = [
# --- BERT family ---
"BERT classification",
"BERT question answering",
"BERT named entity recognition",
"BERT sentiment analysis",
"BERT text classification",
"BERT relation extraction",
"BERT document classification",
"BERT biomedical classification",
"BERT legal text classification",
"BERT fine tuning classification",

"RoBERTa classification",
"RoBERTa sentiment analysis",
"RoBERTa question answering",
"RoBERTa text classification",
"RoBERTa fine tuning classification",

"DeBERTa classification",
"DeBERTa question answering",
"DeBERTa text classification",
"DeBERTa fine tuning classification",

# --- T5 / seq2seq ---
"T5 summarization",
"T5 translation",
"T5 question answering",
"T5 text generation",
"T5 fine tuning summarization",

"FLAN T5 summarization",
"FLAN T5 instruction tuning",
"FLAN T5 question answering",
"FLAN T5 text generation",

"BART summarization",
"BART text generation",
"BART summarization fine tuning",

# --- GPT / LLMs ---
"GPT summarization",
"GPT question answering",
"GPT text generation",
"GPT classification",
"GPT few shot classification",

"LLaMA text generation",
"LLaMA instruction tuning",
"LLaMA question answering",
"LLaMA fine tuning",

"Mistral text generation",
"Mistral instruction tuning",
"Mistral question answering",

"Falcon text generation",
"Falcon instruction tuning",

# --- Diffusion / generative ---
"Stable Diffusion image generation",
"Stable Diffusion text to image",
"Stable Diffusion fine tuning",
"diffusion model image generation",
"diffusion model text to image",

# --- Vision models ---
"ViT image classification",
"ViT fine tuning classification",
"vision transformer image classification",

"CLIP image classification",
"CLIP zero shot classification",
"CLIP image text retrieval",

# --- Speech ---
"Whisper speech recognition",
"Whisper transcription",
"speech recognition transformer model",

# --- Generic pretrained / foundation ---
"pretrained model classification",
"pretrained model text generation",
"pretrained model detection",
"pretrained model question answering",
"pretrained model summarization",

"foundation model classification",
"foundation model text generation",
"foundation model question answering",
"foundation model summarization",

# --- Fine-tuning focused ---
"fine tuning BERT classification",
"fine tuning T5 summarization",
"fine tuning GPT generation",
"fine tuning LLaMA instruction",
"fine tuning transformer classification",

# --- HuggingFace / implementation ---
"huggingface transformers classification",
"huggingface transformers summarization",
"huggingface fine tuning BERT",
"huggingface fine tuning T5",
"huggingface pipeline classification",
"transformers library text classification",
"transformers library summarization",
"diffusers stable diffusion pipeline",

# --- Domain-specific ---
"BERT biomedical named entity recognition",
"BERT clinical text classification",
"transformer model medical image classification",

"legal text classification BERT",
"legal document summarization GPT",

"financial text classification transformer",
"finance sentiment analysis BERT",

"social media sentiment analysis BERT",
"twitter sentiment classification transformer",

"remote sensing image classification transformer",
"satellite image classification ViT",

"education text classification transformer",
"student performance prediction transformer",

"robotics vision transformer perception",
"autonomous driving object detection transformer"
]
queries_100_specific_1 = [
"huggingface transformers classification",
"huggingface transformers summarization",
"huggingface transformers question answering",
"huggingface transformers text generation",
"huggingface transformers retrieval",
"huggingface transformers named entity recognition",
"huggingface transformers sentiment analysis",
"huggingface transformers translation",
"huggingface transformers pipeline task",
"huggingface transformers dataset application",

"huggingface diffusers image generation",
"huggingface diffusers image synthesis",
"stable diffusion generation application",
"stable diffusion dataset generation",
"latent diffusion image synthesis",
"diffusion model image generation task",

"transformers library classification",
"transformers library summarization",
"transformers library question answering",
"transformers library text generation",
"transformers pipeline classification",
"transformers pipeline question answering",
"transformers pipeline summarization",

"bert text classification dataset",
"bert sentiment analysis dataset",
"bert question answering dataset",
"bert named entity recognition dataset",
"bert text classification application",

"roberta text classification dataset",
"roberta sentiment analysis dataset",
"roberta question answering dataset",

"t5 summarization dataset",
"t5 text generation dataset",
"t5 translation dataset",

"gpt text generation dataset",
"gpt language modeling application",

"sentence transformers retrieval dataset",
"sentence transformers semantic search",
"sentence transformers embeddings clustering",

"embedding model semantic search transformer",
"embedding retrieval transformer dataset",
"rag retrieval transformer application",

"biobert clinical classification dataset",
"pubmedbert biomedical classification dataset",
"clinical text classification transformer",

"legalbert legal classification dataset",
"contract classification transformer dataset",

"finbert financial sentiment analysis dataset",
"financial text classification transformer",

"scibert citation classification dataset",
"scientific text classification transformer",

"codebert code classification dataset",
"codet5 code generation dataset",
]
queries_100_specific = [
"education tutoring language model application",
"intelligent tutoring transformer dataset",

"multilingual transformer classification dataset",
"low resource language transformer task",
"cross lingual transformer classification",

"medical imaging vit classification dataset",
"radiology transformer classification",
"pathology image classification transformer",

"satellite image classification transformer",
"remote sensing transformer dataset",
"geospatial transformer classification",

"vision transformer classification dataset",
"vit image classification application",
"cnn transformer hybrid classification",

"image captioning transformer dataset",
"vision language model captioning task",

"video understanding transformer dataset",
"video classification transformer application",

"audio speech recognition transformer dataset",
"asr transformer application",

"time series forecasting transformer dataset",
"time series transformer prediction",

"anomaly detection transformer dataset",
"fraud detection transformer application",

"search ranking transformer embeddings",
"information retrieval transformer dataset",

"recommendation system transformer embeddings",
"recommender system transformer dataset",

"clustering embeddings transformer dataset",
"similarity search transformer embeddings",

"feature extraction transformer dataset",
"representation learning transformer dataset",

"tokenizer huggingface transformers usage",
"transformers inference pipeline dataset",

"training pipeline transformer dataset",
"evaluation pipeline transformer dataset",

"real world application transformer model",
"deployment transformer model application",
"production system transformer model",

"end to end system transformer application",
"framework system huggingface transformers",

"case study transformer application dataset",
"industry application transformer model",

"user study transformer application",
"experiment transformer dataset application",

"pretrained model classification dataset",
"pretrained transformer application dataset",

"fine tuned transformer classification dataset",
"fine tuning transformer task dataset",

"transfer learning transformer dataset",
"downstream task transformer dataset",

"zero shot transformer task dataset",
"few shot transformer classification",

"efficient transformer deployment model",
"lightweight transformer model application",

"small transformer model deployment",
"large language model application dataset",

"foundation model downstream task dataset",

"domain specific transformer dataset",
"task specific transformer application",

"ensemble transformer model application",
"model mixture transformer dataset",

"cross domain transformer generalization",
"robust transformer model evaluation"
]

queries_domain = [
"biomedical bert classification dataset",
"biomedical roberta classification",
"biomedical t5 summarization",
"clinical bert classification dataset",
"clinical roberta text classification",
"ehr bert classification dataset",
"medical text t5 summarization",
"healthcare transformer classification dataset",

"radiology vit classification dataset",
"pathology vit classification",
"medical imaging resnet classification",
"medical imaging transformer segmentation",
"xray classification vit dataset",
"ct scan transformer classification",

"legal bert classification dataset",
"legal roberta classification",
"contract analysis bert classification",
"case law transformer classification",
"legislation roberta classification",

"financial bert sentiment analysis",
"financial roberta classification",
"earnings call bert analysis",
"stock market transformer prediction",
"finance t5 summarization dataset",

"scientific scibert classification dataset",
"scientific roberta classification",
"citation classification scibert dataset",
"research paper classification transformer",

"chemistry transformer molecules prediction",
"smiles transformer classification",
"drug discovery transformer dataset",
"molecular property prediction transformer",

"codebert code classification dataset",
"codet5 code generation dataset",
"program analysis transformer dataset",
"source code classification bert",

"education transformer classification dataset",
"tutoring llm application dataset",
"student performance transformer prediction",
"education bert classification",

"multilingual bert classification dataset",
"xlm roberta classification dataset",
"cross lingual bert classification",
"low resource language transformer dataset",

"social media bert sentiment analysis",
"twitter roberta sentiment dataset",
"reddit transformer classification",
"social network transformer analysis",

"news bert classification dataset",
"news summarization t5 dataset",
"article classification roberta",
"news transformer summarization",

"customer reviews bert sentiment",
"product reviews roberta classification",
"ecommerce transformer recommendation",
"review classification transformer dataset",

"search transformer ranking dataset",
"information retrieval bert dataset",
]

queries_domain_2 = [
"semantic search sentence transformers",

"recommendation transformer embeddings dataset",
"recommender system bert embeddings",
"user behavior transformer prediction",
"content recommendation transformer",

"cybersecurity transformer anomaly detection",
"intrusion detection transformer dataset",
"malware classification bert dataset",
"fraud detection roberta dataset",

"time series transformer forecasting dataset",
"stock prediction transformer dataset",
"energy consumption transformer forecasting",
"weather prediction transformer dataset",

"audio transformer speech recognition",
"asr transformer dataset",
"speech classification transformer dataset",
"audio classification transformer",

"video transformer classification dataset",
"video understanding transformer",
"action recognition transformer dataset",
"video captioning transformer",

"image captioning transformer dataset",
"vision language model captioning dataset",
"image text transformer alignment",
"multimodal transformer captioning",

"satellite transformer classification dataset",
"remote sensing transformer dataset",
"geospatial transformer prediction",
"earth observation transformer dataset",

"agriculture transformer crop classification",
"crop disease detection transformer dataset",
"plant classification transformer dataset",
"precision agriculture transformer",

"transportation transformer traffic prediction",
"traffic forecasting transformer dataset",
"mobility transformer dataset",
"route prediction transformer",

"health monitoring transformer prediction",
"wearable sensor transformer dataset",
"patient monitoring transformer model",
"clinical time series transformer",

"question answering bert dataset domain",
"qa roberta dataset domain specific",
"faq transformer question answering",
"customer support qa transformer",

"summarization t5 domain specific dataset",
"legal document summarization t5",
"medical summarization transformer",
"financial report summarization transformer",

"translation transformer domain dataset",
"multilingual translation bert dataset",
"cross lingual translation transformer",
"low resource translation transformer"
]