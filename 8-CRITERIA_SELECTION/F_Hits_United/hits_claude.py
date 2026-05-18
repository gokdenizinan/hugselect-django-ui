CLAUDE_RECOMMENDATIONS  ={
    "Sample_8": [
    "facebook/nllb-200-distilled-600M",           # NLLB-200: 200 langs, Arabic-first, best for span translation
    "facebook/nllb-200-1.3B",                     # NLLB-200 larger variant, higher quality for rare lang pairs
    "facebook/m2m100_418M",                        # M2M100: true many-to-many, 100 langs, no English pivot
    "facebook/m2m100_1.2B",                        # M2M100 larger: better quality across 9900 directions
    "facebook/mbart-large-50-many-to-many-mmt",   # mBART-50: seq2seq, Arabic included, fine-tuning friendly
    "Helsinki-NLP/opus-mt-ar-en",                  # Compact Arabic→English, fast inference for span swapping
    "Helsinki-NLP/opus-mt-tc-big-ar-en",          # Larger OPUS Arabic→English with better fluency
    "google/madlad400-3b-mt",                      # MADLAD-400: 400+ langs including Arabic dialects, T5-based
    "UBC-NLP/turjuman",                            # Arabic-centric many-to-many MT, built for MSA and dialects
    "microsoft/Azure-AI-translator-mt-ar",         # Azure-backed Arabic multilingual seq2seq fine-tune target
],
    "Sample_9":[
    "google/mt5-base",
    "google/mt5-large",
    "google/mt5-xl",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "bigscience/mt0-large",
    "bigscience/mt0-xl",
    "facebook/mbart-large-cc25",
    "ai4bharat/IndicBART",
    "UBC-NLP/AraT5v2-base-1024",
],
    "Sample_18": [
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "google/pegasus-large",
    "google/pegasus-xsum",
    "google/pegasus-x-large",
    "sshleifer/distilbart-cnn-12-6",
    "philschmid/bart-large-cnn-samsum",
    "google/long-t5-tglobal-large",
    "allenai/led-large-16384",
    "pszemraj/led-large-book-summary",
],
    "Sample_19": [
    "Qwen/Qwen2.5-7B",                           # Best overall Chinese base: SFT/PPO/RLHF-ready, explicit support
    "Qwen/Qwen2.5-14B",                          # Larger Qwen2.5 base for higher-capacity medical adaptation
    "internlm/internlm2-base-7b",                # Natively trained with RLHF+PPO, 200K context, strong Chinese
    "internlm/internlm2-base-20b",               # InternLM2 at 20B scale, same full training regime
    "THUDM/chatglm3-6b-base",                    # ChatGLM3 base: Chinese-first, bilingual, safety-aligned
    "baichuan-inc/Baichuan2-13B-Base",           # Trained on 2.6T tokens; proven Chinese medical baseline
    "deepseek-ai/deepseek-llm-7b-base",          # DeepSeek-LLM: strong reasoning, full SFT/RLHF pipeline
    "deepseek-ai/deepseek-llm-67b-base",         # DeepSeek 67B for high-capacity continued pretraining
    "FreedomIntelligence/HuatuoGPT2-7B",         # Chinese medical SFT+RLHF model, direct domain reference
    "BAAI/AquilaMed-RL",                         # Full pretrain→SFT→DPO pipeline on Chinese medical data
],
    "Sample_24": [
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "google/vit-base-patch16-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-huge-patch14-224-in21k",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-large-patch16-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
],
    "Sample_55":  [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large",
    "thenlper/gte-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
],
    "Sample_62": [
    "stabilityai/stable-diffusion-x4-upscaler",
    "stabilityai/sd-x2-latent-upscaler",
    "stabilityai/stable-diffusion-2-1",
    "diffusers/stable-diffusion-xl-refiner-1.0",
    "ai-forever/Real-ESRGAN",
    "sberbank-ai/Real-ESRGAN",
    "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "CompVis/ldm-super-resolution-4x-openimages",
    "gilesagnel/SD-upscaler-pipeline",
    "konieshadow/fooocus-api-realistic",
],

    "sample_86":[
    "facebook/wav2vec2-base",                        # 95M params – strong general-purpose audio encoder, fast base variant
    "facebook/hubert-base-ls960",                    # 94M params – HuBERT base, rich self-supervised audio representations
    "microsoft/wavlm-base",                          # 94M params – WavLM base, excels at diverse audio understanding tasks
    "openai/whisper-tiny",                           # 39M params – smallest Whisper encoder, very fast, good feature backbone
    "openai/whisper-base",                           # 74M params – slightly richer than tiny, still real-time capable
    "MIT/ast-finetuned-audioset-10-10-0.4593",       # Audio Spectrogram Transformer, strong AudioSet representations
    "facebook/wav2vec2-base-960h",                   # wav2vec2 fine-tuned on LibriSpeech, speech-optimised embeddings
    "microsoft/wavlm-base-plus",                     # WavLM base+ with extra unlabelled data, better generalization
    "Anton-Sizykh/audio-embeddings-model",           # Dedicated lightweight audio embedding model
    "laion/larger_clap_general",                     # CLAP: joint audio-language embeddings, great for AV alignment
],
    "Sample_90":  [
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "roberta-large",
    "ALBERT/albert-xxlarge-v2",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-v3-base",
    "deepset/roberta-large-squad2",
    "LIAMF-USP/roberta-large-finetuned-race",
    "ALBERT/albert-xxlarge-v2-finetuned-race",
    "tasksource/deberta-large-long-nli",
],
    "Sample_98": [
    "allenai/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_cased",
    "allenai/longformer-base-4096",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-v1.1",
    "sultan/BioM-ELECTRA-Large-SQuAD2",
    "NLP4Science/scibert-acronym-disambiguation",
    "allenai/specter2_base",
    "malteos/scincl",
    "gsarti/scibert-nli",
],
    "Sample_99":  [
    "emilyalsentzer/Bio_ClinicalBERT",
    "medicalai/ClinicalBERT",
    "obi/deid_roberta_i2b2",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "sultan/BioM-ELECTRA-Large-SQuAD2",
    "dmis-lab/biobert-v1.1",
    "cambridgeltl/SapBERT-from-PubMed-self-supervized",
    "UFNLP/gatortron-base",
    "StanfordAIMI/stanford-deidentifier-base",
    "healx/gpt-2-pubmed-medium",
],
    "Sample_116":  [
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-roberta-base",
    "facebook/bart-large-mnli",
    "typeform/distilbart-mnli-12-3",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "microsoft/deberta-large-mnli",
    "roberta-large-mnli",
    "cross-encoder/nli-MiniLM2-L6-H768",
],
"Sample_130": [
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-large",
    "microsoft/deberta-base",
    "microsoft/deberta-xlarge-v2",
    "microsoft/deberta-v2-xxlarge",
    "google/bigbird-roberta-large",
    "allenai/longformer-large-4096",
    "studio-ousia/luke-large",
],
    "Sample_134": [
    "facebook/bart-large",
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-50",
    "google/t5-v1_1-large",
    "google/t5-v1_1-base",
    "google/flan-t5-large",
    "microsoft/prophetnet-large-uncased",
    "Yale-LILY/blink-biencoder-bi-encoder",
],
    "Sample_135": [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/bart-large-mnli",
    "sshleifer/distilbart-cnn-6-6",
    "sshleifer/distilbart-cnn-12-6",
    "sshleifer/distilbart-xsum-12-3",
    "lucadiliello/bart-small",
    "ainize/bart-base-cnn",
],
    "Sample_136":[
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/bart-large",
    "facebook/bart-base",
    "google/pegasus-large",
    "google/pegasus-xsum",
    "google/pegasus-x-large",
    "sshleifer/distilbart-cnn-12-6",
    "google/flan-t5-large",
    "allenai/led-large-16384",
],
    "Sample_146": [
    "facebook/contriever-msmarco",               # Unsupervised contrastive retriever – top zero-shot generalization on BEIR
    "facebook/contriever",                        # Base Contriever without MS MARCO fine-tuning – purer unsupervised representations
    "BAAI/bge-base-en-v1.5",                      # BGE base – excellent query/doc asymmetric encoding, strong MTEB performance
    "BAAI/bge-m3",                                # BGE-M3 – dense + sparse + multi-vector, strong cross-domain & multilingual retrieval
    "sentence-transformers/msmarco-distilbert-base-tas-b",  # TAS-B dual-encoder – optimised for passage retrieval, fast inference
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",     # MPNet trained on 215M QA pairs – generalises well to unseen query types
    "intfloat/e5-base-v2",                        # E5 base – instruction-aware embeddings, strong zero-shot retrieval on BEIR
    "intfloat/e5-large-v2",                       # E5 large – higher-capacity variant, top BEIR scores for unseen domains
    "facebook/dpr-question_encoder-single-nq-base",  # DPR query encoder – classic dual-encoder for dialogue QA retrieval
    "sentence-transformers/all-mpnet-base-v2",    # General-purpose MPNet bi-encoder – robust cross-domain sentence embeddings
],
    "Sample_158": [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-distilroberta-v1",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "declare-lab/flan-alpaca-base",
],
    "Sample_177": [
    "Qwen/Qwen2.5-Coder-7B-Instruct",            # Top-tier code+math instruct model; decontaminated on HumanEval/MATH for fair eval
    "Qwen/Qwen2.5-Math-7B-Instruct",              # Dedicated math instruct model; SOTA on GSM8K, MATH, AIME, OlympiadBench
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # R1 reasoning distilled into 7B; strong chain-of-thought on math and code
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # R1 distill at 32B; outperforms OpenAI-o1-mini on reasoning benchmarks
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",# Specialist code instruct model; strong HumanEval/MBPP generalization
    "meta-llama/Llama-3.1-8B-Instruct",           # Llama 3.1 instruct; 128k context, reliable zero-shot reasoning baseline
    "meta-llama/Llama-3.3-70B-Instruct",          # Larger Llama instruct; strong cross-task generalization, broad eval coverage
    "Qwen/Qwen2.5-72B-Instruct",                  # Qwen2.5 flagship instruct; matches Llama-3.1-405B on math and coding tasks
    "microsoft/phi-4",                             # 14B dense model; exceptional math reasoning and code for its size class
    "MathLLMs/MathCoder2-DeepSeekMath-7B",        # MathCoder2 fine-tune; interleaves NL+code+execution for math problem solving
],
    "Sample_178": [
    "intfloat/e5-large-v2",                          # E5-large: first zero-shot model to beat BM25 on full BEIR benchmark
    "intfloat/e5-mistral-7b-instruct",               # E5 backed by Mistral-7B; top MTEB retrieval scores, instruction-aware
    "BAAI/bge-large-en-v1.5",                        # BGE-large: RetroMAE pre-trained, strong zero-shot QA passage retrieval
    "BAAI/bge-m3",                                   # BGE-M3: dense + sparse + multi-vector; scales to large web corpora
    "facebook/contriever-msmarco",                   # Contriever: unsupervised, robust zero-shot BEIR generalization
    "facebook/dpr-question_encoder-single-nq-base",  # DPR: the canonical dual-encoder for open-domain QA over Wikipedia
    "facebook/rag-token-nq",                         # RAG-Token: end-to-end retriever+reader, native LM integration via FAISS
    "sentence-transformers/msmarco-distilbert-base-tas-b",  # TAS-B: optimised dual-encoder for passage retrieval, fast inference
    "Alibaba-NLP/gte-large-en-v1.5",                 # GTE-large: top MTEB retrieval model from Alibaba, strong on QA domains
    "nvidia/dragon-plus-query-encoder",              # DRAGON+: progressive augmentation training, strong cross-domain QA retrieval
],
    "Sample_179": [
    "microsoft/swin-large-patch4-window12-384-in22k",
    "microsoft/swin-base-patch4-window12-384-in22k",
    "OpenGVLab/InternViT-6B-448px-V2_5",
    "OpenGVLab/InternVL2_5-8B",
    "OpenGVLab/InternVL2-1B",
    "llava-hf/llava-1.5-7b-hf",
    "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "microsoft/Florence-2-large",
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
],
    "Sample_184": [
    "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft",
    "microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft",
    "microsoft/swin-large-patch4-window12-384-in22k",
    "OpenGVLab/InternViT-6B-448px-V2_5",
    "OpenGVLab/InternVL2_5-8B",
    "google/siglip-so400m-patch14-384",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "microsoft/Florence-2-large",
    "llava-hf/llava-1.5-13b-hf"
],
    "Sample_194": [
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/bart-base",
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "google/mt5-base",
    "Helsinki-NLP/opus-mt-en-de",
    "facebook/mbart-large-cc25"
],
    "Sample_212": [
    "MingZhong/unieval-intermediate",
    "MingZhong/unieval-sum",
    "MingZhong/unieval-dialog",
    "MingZhong/unieval-fact",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/t5-v1_1-large",
    "google/t5-v1_1-xl"
],
    "Sample_213": [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "lmsys/vicuna-7b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
],
    "Sample_216":  [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "CompVis/stable-diffusion-v1-4",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "kandinsky-community/kandinsky-2-2-decoder",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "dataautogpt3/OpenDalleV1.1"
],
    "Sample_217": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-base-patch32",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch16-384",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip-image-captioning-large",
    "facebook/flava-full",
    "kakaobrain/align-base"
],
    "Sample_220": [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-so400m-patch16-384",
    "google/owlvit-large-patch14",
    "google/owlv2-large-patch14-ensemble",
    "facebook/metaclip-h14-fullcc2.5b",
    "Salesforce/blip2-opt-2.7b"
],
    "Sample_223": [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "kandinsky-community/kandinsky-2-2-decoder"
],
    "Sample_224": [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "Qwen/Qwen2.5-7B-Instruct",
    "openai/clip-vit-large-patch14",
    "google/siglip-so400m-patch14-384"
],
    "Sample_229": [
    "google-t5/t5-large",
    "google-t5/t5-3b",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "facebook/bart-large",
    "facebook/bart-base",
    "tuner007/pegasus_paraphrase",
    "Salesforce/codet5-large",
    "google/t5-v1_1-large",
    "microsoft/GODEL-v1_1-large-seq2seq"
],
    "Sample_230": [
    "google/vit-base-patch16-224-in21k",        # Canonical ViT-Base, ImageNet-21k pretrained
    "google/vit-large-patch16-224-in21k",        # ViT-Large for scale ablations
    "microsoft/swin-base-patch4-window7-224-22k",# Swin Transformer (hierarchical ViT variant)
    "microsoft/swin-large-patch4-window12-384-22k", # Swin-Large, high-res generalization
    "facebook/deit-base-distilled-patch16-224",  # DeiT-Base, distillation-based ViT
    "facebook/dinov2-base",                      # DINOv2 self-supervised ViT backbone
    "facebook/dinov2-large",                     # DINOv2-Large, SOTA self-supervised features
    "openai/clip-vit-base-patch16",              # CLIP ViT-Base, vision-language pretraining
    "openai/clip-vit-large-patch14",             # CLIP ViT-Large, richer patch embeddings
    "timm/vit-base-patch16-clip-224.openai",     # timm-wrapped CLIP ViT for easy PEFT integration
],
    "Sample_231": [
    "allenai/specter2",
    "allenai/specter2_base",
    "allenai/specter2_classification",
    "allenai/scibert_scivocab_uncased",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "NeuML/pubmedbert-base-embeddings",
    "BioMistral/BioMistral-7B",
    "stanford-crfm/BioMedLM",
    "abhinand/MedEmbed-large-v0.1"
],
    "Sample_237": [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-instruct",
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-base",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "hkunlp/instructor-base",
    "intfloat/multilingual-e5-base"
],
    "Sample_246": [
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "lllyasviel/sd-controlnet-canny",
    "diffusers/controlnet-canny-sdxl-1.0",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "timbrooks/instruct-pix2pix",
    "kandinsky-community/kandinsky-2-2-decoder",
    "thu-ml/unidiffuser-v1"
],
    "Sample_247": [
    "stabilityai/stable-diffusion-2-inpainting",
    "runwayml/stable-diffusion-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Fantasy-Studio/Paint-by-Example",
    "lllyasviel/control_v11p_sd15_inpaint",
    "h94/IP-Adapter",
    "TencentARC/BrushNet",
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    "xichenhku/AnyDoor",
    "Huggingface/MAT"
],
    "Sample_264": [
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "microsoft/git-large-coco",
    "OFA-Sys/ofa-large",
    "Salesforce/instructblip-vicuna-7b",
    "llava-hf/llava-1.5-7b-hf",
    "nlpconnect/vit-gpt2-image-captioning",
    "google/paligemma-3b-pt-224",
    "HuggingFaceM4/idefics2-8b"
],
    "Sample_265": [
    "OFA-Sys/ofa-large",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "microsoft/git-large-textcaps",
    "google/paligemma-3b-mix-224",
    "j-min/VL-T5",
    "microsoft/unilm-base-cased",
    "MAGAer13/mplug-owl2-llama2-7b",
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    "allenai/uio2-large"
],
    "Sample_268":  [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "google/electra-large-discriminator",
    "google/electra-base-discriminator",
    "FacebookAI/roberta-large",
    "google-bert/bert-large-uncased",
    "distilbert/distilbert-base-uncased",
    "albert/albert-xxlarge-v2",
    "xlnet/xlnet-large-cased",
    "google/mobilebert-uncased"
],
    "Sample_281": [
    "allenai/specter2_base",
    "allenai/scibert_scivocab_uncased",
    "allenai/specter",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-large-cased-v1.1",
    "malteos/scincl",
    "allenai/specter2_proximity",
    "globuslabs/ScholarBERT",
    "michiyasunaga/BioLinkBERT-large",
    "allenai/cs_roberta_base"
],
    "Sample_290": [
    "facebook/bart-large-cnn",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "pszemraj/led-large-book-summary",
    "google/bigbird-pegasus-large-arxiv",
    "google-t5/t5-large",
    "google/flan-t5-large",
    "facebook/bart-large-xsum",
    "google/long-t5-tglobal-large",
    "google/mt5-large"
],
    "Sample_292": [
    "MBZUAI/geochat-7B",
    "Salesforce/blip2-flan-t5-xxl",
    "openai/clip-vit-large-patch14",
    "flax-community/remoteclip",
    "OpenGVLab/InternVL3-8B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-7b-hf",
    "microsoft/Florence-2-large",
    "Qwen/Qwen2-VL-7B-Instruct",
],
    "Sample_293": [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    "microsoft/udop-large",
    "naver-clova-ix/donut-base",
    "google/pix2struct-base",
    "microsoft/dit-base",
    "microsoft/dit-large",
    "microsoft/layoutlmv2-base-uncased",
    "google/pix2struct-docvqa-base",
    "Qwen/Qwen2-VL-7B-Instruct",
],
    "Sample_296": [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "google/flan-t5-xl",
    "allenai/led-base-16384",
    "google/flan-t5-xxl",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]
}