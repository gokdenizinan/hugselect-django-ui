#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
improved_family_discovery_with_relations.py

Goal
----
Cluster foundation models into meaningful model families with higher precision.

Main improvements over the earlier version
------------------------------------------
1. No more "first surviving token" fallback for family extraction.
2. Strong validation layer to reject nonsense candidate families.
3. Discovery runs only on unresolved rows, not the full dataset.
4. Candidate promotion requires multiple signals beyond raw frequency.
5. Better separation of family / child / variant attributes.
6. More conservative fuzzy matching.
7. Optional coherence checks for discovered families.
8. model_type is now a first-class family hint source.
9. assignments CSV now includes family_source = seeded/discovered/unresolved.

Expected outputs
----------------
- family_assignments.csv
- family_relations.csv
- discovered_family_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# User-provided model_type counts
# ============================================================

MODEL_TYPE_COUNTS = {
    "mistral": 5308,
    "qwen2": 3100,
    "roberta": 1400,
    "llama": 12150,
    "xlm-roberta": 745,
    "siglip": 69,
    "bert": 3771,
    "t5": 1685,
    "albert": 141,
    "qwen2_vl": 170,
    "gemma2": 903,
    "phi3": 303,
    "whisper": 1021,
    "cohere2": 14,
    "gpt2": 1438,
    "llava": 279,
    "modernbert": 169,
    "wav2vec2": 849,
    "mixtral": 951,
    "vit": 640,
    "capai": 1,
    "esm": 41,
    "deberta-v2": 357,
    "opt": 132,
    "sam2_video": 8,
    "mllama": 68,
    "gptj": 111,
    "deberta": 55,
    "yolos": 27,
    "gemma": 370,
    "qwen2_5_vl": 104,
    "deepseek_v2": 46,
    "marian": 505,
    "idefics2": 23,
    "longt5": 42,
    "mt5": 309,
    "van": 5,
    "mbart": 160,
    "chatglm": 71,
    "llama4": 44,
    "RefinedWebModel": 49,
    "orion": 12,
    "blip": 34,
    "smolvlm": 9,
    "stablelm_epoch": 70,
    "gpt_neox": 351,
    "tapas": 25,
    "baichuan": 57,
    "qwen": 99,
    "gemma3": 144,
    "speech-encoder-decoder": 22,
    "layoutlmv3": 34,
    "timesformer": 9,
    "vmistral": 2,
    "bart": 470,
    "electra": 160,
    "mpt": 112,
    "QH_360VL": 3,
    "vision-encoder-decoder": 210,
    "phi3_v": 25,
    "span-marker": 45,
    "gpt_neo": 135,
    "mistral3": 10,
    "blip-2": 43,
    "LanguageBindImage": 1,
    "bloom": 209,
    "distilbert": 872,
    "centurio": 2,
    "mpnet": 155,
    "wav2vec2-bert": 40,
    "llava_phi": 8,
    "eagle_2_5_vl": 2,
    "xlnet": 19,
    "clip": 91,
    "llava_jamba": 3,
    "camembert": 113,
    "llava_qwen2": 5,
    "facebook/opt": 1,
    "llava_next": 35,
    "florence2": 74,
    "zamba": 2,
    "pegasus": 84,
    "videollama2_qwen2": 4,
    "xmod": 8,
    "git": 30,
    "dbrx": 14,
    "m2m_100": 138,
    "mamba": 34,
    "swin2sr": 10,
    "videomae": 30,
    "codet5p": 4,
    "speecht5": 96,
    "vitpose": 12,
    "ultravox": 14,
    "bunny-qwen2": 3,
    "internvl_chat": 135,
    "mobilenet_v1": 2,
    "telechat": 8,
    "xmodel": 1,
    "new": 47,
    "mobilevitv2": 3,
    "swin": 88,
    "liger_gsa": 1,
    "apriel": 3,
    "phi": 127,
    "udop": 4,
    "rwkv7": 14,
    "idefics3": 22,
    "hunyuan": 4,
    "hifigan": 1,
    "hyenadna": 5,
    "cohere": 104,
    "miewid": 2,
    "dinov2_with_registers": 7,
    "cellrepDINO": 1,
    "llava_onevision": 13,
    "ernie-pixel": 3,
    "swinv2": 36,
    "longformer": 54,
    "granite": 51,
    "glm4": 17,
    "codegen": 51,
    "dragonfly": 4,
    "convnextv2": 35,
    "canine": 12,
    "aya_vision": 11,
    "stablelm": 43,
    "GLiClass": 13,
    "olmo": 26,
    "dinov2": 29,
    "reformer": 2,
    "LUAR": 2,
    "minicpm": 6,
    "meralion": 1,
    "vit_b": 3,
    "bunny-llama": 6,
    "doge": 19,
    "sa2va_chat": 4,
    "falcon": 78,
    "jamba": 33,
    "marqo-chimera-arctic-bge-m": 1,
    "hierarchical-transformer": 4,
    "resnet": 55,
    "upcycling-qwen2-moe": 3,
    "gpt_bigcode": 84,
    "internlmxcomposer2": 5,
    "clip_vision_model": 10,
    "internlm2": 72,
    "encoder-decoder": 69,
    "MobileNetV1": 1,
    "eurobert": 11,
    "detr": 45,
    "led": 39,
    "audio-spectrogram-transformer": 26,
    "paligemma": 78,
    "nanogpt-j": 2,
    "llaaa": 2,
    "beit": 53,
    "mask2former": 25,
    "exaone": 26,
    "squeezebert": 1,
    "mambavision": 11,
    "olmo2": 29,
    "intern_vit_6b": 7,
    "depth_anything": 22,
    "openvla": 8,
    "v8": 53,
    "xglm": 28,
    "glm": 17,
    "vits": 124,
    "gemma3_text": 53,
    "phi3small": 6,
    "xverse": 11,
    "encodec": 4,
    "perception_lm": 3,
    "diva": 1,
    "cable": 4,
    "otter": 4,
    "GOT": 3,
    "HelpingAI": 7,
    "bailing_moe": 7,
    "convnext": 31,
    "segformer": 69,
    "lean_albert": 1,
    "TransHLA": 2,
    "rwkv5": 8,
    "hubert": 91,
    "ultra": 2,
    "internlm": 31,
    "CXR-LLAVA": 1,
    "diffrhythm": 2,
    "dual_ar": 11,
    "nvembed": 3,
    "sam": 21,
    "lumenspark": 1,
    "aquilamoe": 2,
    "phi-msft": 57,
    "longllama": 5,
    "index": 5,
    "vilt": 7,
    "detikzify": 6,
    "stripedhyena": 7,
    "regnet": 7,
    "multi_modality": 16,
    "mixformer-sequential": 28,
    "switch_transformers": 12,
    "timm_wrapper": 2,
    "xlm-token": 6,
    "kimi_vl": 3,
    "speech_to_text": 13,
    "ovis": 25,
    "mctct": 1,
    "flux-rectified-flow": 1,
    "xgenmm": 8,
    "t2v": 10,
    "musicgen": 19,
    "luke": 33,
    "relik-reader": 1,
    "T-CLM2": 1,
    "starcoder2": 33,
    "llada": 5,
    "llava_llama": 90,
    "davit": 1,
    "staticvectors": 8,
    "sparsetral": 7,
    "musilingo": 3,
    "enct5": 1,
    "molmo": 9,
    "jetmoe": 3,
    "emova_qwen2": 3,
    "creek": 1,
    "yi": 21,
    "dallebart": 4,
    "jina_clip": 3,
    "fuxitranyu": 3,
    "zhinao": 11,
    "recurrent_gemma": 7,
    "vgcn-bert": 1,
    "set-encoder": 2,
    "deit": 7,
    "llava_phi3": 3,
    "mineru": 1,
    "swiftformer": 5,
    "mobilevit": 11,
    "bigbird_pegasus": 6,
    "flmr": 4,
    "roformer": 16,
    "theta": 2,
    "mobilebert": 16,
    "qwen2_audio": 5,
    "gpt2l": 1,
    "gpt_jx": 1,
    "blenderbot": 12,
    "hunyuan_v1_dense": 2,
    "clipseg": 4,
    "deepseek_v3": 60,
    "atomformer": 1,
    "sew-d": 6,
    "neobert": 3,
    "parler_tts": 30,
    "wavlm": 17,
    "convbert": 12,
    "minicpmv": 21,
    "i2v": 13,
    "xclip": 14,
    "focalnet": 5,
    "data2vec-vision": 6,
    "owlv2": 7,
    "rwkv_hybrid": 5,
    "VideoMAEv2_Base": 4,
    "bunny-stablelm": 1,
    "pegasus_x": 9,
    "granitemoe": 14,
    "mrt5": 2,
    "graphormer": 2,
    "conditional_detr": 7,
    "RefinedWeb": 29,
    "bit": 3,
    "nomic_bert": 29,
    "align": 1,
    "eagle_llama": 6,
    "fasttext_jp": 1,
    "big_bird": 27,
    "rwkv": 14,
    "prompt_depth_anything": 2,
    "musicgen_melody": 9,
    "japanese_stable_clip": 1,
    "transformer": 2,
    "megatron-bert": 19,
    "speech_language_model": 4,
    "YOLO11n-seg": 1,
    "layoutlmv2": 22,
    "aligngpt": 1,
    "llava_qwen1_5": 3,
    "aquila": 16,
    "roberta-prelayernorm": 10,
    "gpt_refact": 2,
    "olmoe": 10,
    "aria": 8,
    "oryx_llama": 2,
    "bunny-phi": 8,
    "fnet": 6,
    "seamless_m4t": 2,
    "birefnet": 1,
    "nllb-llm2vec": 1,
    "videollama2_mixtral": 2,
    "upernet": 8,
    "ESMplusplus": 2,
    "amber": 2,
    "phonelm": 3,
    "deepseek": 14,
    "jais": 31,
    "decision_transformer": 5,
    "tinyllava": 13,
    "efficientnet": 12,
    "llava_mpt": 1,
    "lilt": 8,
    "videochat_flash_qwen": 3,
    "rwkv6": 7,
    "acip_model": 9,
    "efficientformer": 4,
    "vit_msn": 4,
    "tinyllama": 2,
    "latent_recurrent_depth": 1,
    "moondream1": 15,
    "oryx_qwen": 4,
    "umt5": 7,
    "video_llava": 1,
    "ChatUniVi": 3,
    "LEGO": 1,
    "xlm-prophetnet": 1,
    "ernie_m": 3,
    "table-transformer": 9,
    "llava_qwen": 12,
    "zamba2": 10,
    "gemmoe": 5,
    "qwen2_moe": 19,
    "SENTINEL-SRC-DA": 1,
    "cpmbee": 4,
    "rembert": 3,
    "custom-clip-model": 2,
    "mmMamba_chat": 2,
    "caduceus": 10,
    "mvp": 14,
    "fegeo-qwen2": 1,
    "splinter": 2,
    "nanbeige": 8,
    "moe_llava_stablelm": 3,
    "internimage": 11,
    "video_blip": 3,
    "layoutlm": 14,
    "instructblip": 12,
    "markuplm": 5,
    "funnel": 10,
    "openelm": 16,
    "batgpt": 1,
    "taivisionlm": 2,
    "pvt_v2": 7,
    "deformable_detr": 5,
    "doubutsu_next": 1,
    "moduleformer": 3,
    "bd3lm": 4,
    "vita-Qwen2": 1,
    "BlueLM": 6,
    "cambrian_qwen": 3,
    "yolov10": 8,
    "minicpm3": 4,
    "retnet": 10,
    "phi4mm": 11,
    "hf_olmo": 6,
    "biogpt": 9,
    "ibm-granite-code": 2,
    "gpt_jiang": 2,
    "pagnolxl": 1,
    "TeleFLM": 3,
    "pix2struct": 30,
    "eve": 3,
    "share4v": 5,
    "moe_llava_phi": 3,
    "dpr": 10,
    "colpali": 2,
    "sparsellama": 6,
    "capx-llama": 1,
    "deci": 9,
    "rexseek_qwen": 1,
    "nemotron": 10,
    "shuka": 1,
    "chinese_clip": 8,
    "aquila3": 1,
    "nezha": 2,
    "llava-qwen2": 4,
    "isoformer": 2,
    "hiera": 6,
    "aimv2_vision_model": 10,
    "mplug-owl": 2,
    "bunny-qwen": 1,
    "GPNRoFormer": 1,
    "ernie": 20,
    "mobilevlm": 5,
    "lrm_generator": 2,
    "clap": 6,
    "chameleon": 15,
    "perceiver": 9,
    "qwen3": 15,
    "gbswt5": 2,
    "bge-m3": 1,
    "moe": 1,
    "Yi": 32,
    "SENTINEL-SRC-MQM": 1,
    "dab-detr": 3,
    "nystromformer": 1,
    "seamless_m4t_v2": 4,
    "lite-whisper": 9,
    "graphs_gpt": 1,
    "ser": 5,
    "monet": 8,
    "helium": 5,
    "clip_text_camembert": 1,
    "omni_speech2s_llama": 1,
    "gptsan-japanese": 1,
    "spatiallm_qwen": 1,
    "aquiladense": 1,
    "instella": 4,
    "RWKV-6": 1,
    "moe_llava_qwen": 2,
    "bitnet": 3,
    "flaubert": 13,
    "IndicTrans": 7,
    "walsh-causal-v1": 1,
    "grounding-dino": 5,
    "Tanuki": 3,
    "baichuan_m1": 3,
    "zoedepth": 3,
    "m2_bert": 6,
    "TAAS": 1,
    "pixel": 2,
    "ngen3": 1,
    "nemotron-nas": 6,
    "linglong": 1,
    "plapt": 1,
    "data2vec-audio": 10,
    "bailing_moe_linear": 1,
    "protst": 1,
    "mobilenet_v2": 18,
    "yolov9": 3,
    "vipllava": 2,
    "gte": 3,
    "deepseek_vl_v2": 5,
    "fsmt": 13,
    "yuan": 6,
    "idefics2_moe": 1,
    "LanguageBindAudio": 2,
    "ddllama": 1,
    "mplugowl3": 4,
    "timer": 1,
    "inflm": 4,
    "spatialvla": 2,
    "hindi_causal_lm": 1,
    "ijepa": 2,
    "minicpmo": 4,
    "u2net": 2,
    "EMOVASpeechTokenizer": 1,
    "hymba": 3,
    "CodeT5": 1,
    "oneformer": 6,
    "spec-1-mini": 1,
    "bert_crf": 1,
    "ullava": 1,
    "vlm": 5,
    "ESGBertReddit": 1,
    "superpoint": 2,
    "apollo": 4,
    "mobilellm": 7,
    "rwkv6qwen2": 3,
    "geochat": 1,
    "extended-mpt": 3,
    "idefics": 8,
    "xcodec2": 1,
    "flf2v": 1,
    "llava_mistral": 22,
    "ced": 3,
    "vivit": 4,
    "Deltalm": 3,
    "clip-encoder-decoder": 1,
    "extra_trees": 3,
    "internlm3": 6,
    "lsw_transformer": 1,
    "chatts": 1,
    "typhoon2audio": 1,
    "rt_detr_v2": 4,
    "altclip": 3,
    "tranception": 2,
    "mimi": 2,
    "mistral-lmm": 5,
    "progen": 8,
    "meralion_bestrq": 1,
    "Emu3": 3,
    "stablelm_alpha": 3,
    "mmalaya": 1,
    "bert_vae": 1,
    "gobbledygook": 1,
    "HumanOmni_qwen2": 1,
    "codexembed2b": 1,
    "persimmon": 2,
    "gpt_pangu": 3,
    "SegformerForSemanticSegmentation": 3,
    "xlm": 5,
    "llava_gemma": 2,
    "fastspeech2": 1,
    "cxr_basic": 1,
    "fastspeech2_conformer": 1,
    "codeshell": 2,
    "emu3": 1,
    "sew": 5,
    "tiny_llava": 1,
    "poolformer": 5,
    "mistral_denseformer": 2,
    "jukebox": 2,
    "falcon_mamba": 6,
    "liger_gla": 1,
    "depth_pro": 2,
    "mc-llava": 1,
    "qwen3_moe": 2,
    "keras": 1,
    "midi_model": 1,
    "llama_deepseek": 1,
    "SENTINEL-REF-DA": 1,
    "model2vec": 6,
    "mllama_text_model": 1,
    "maskformer": 9,
    "gliner": 2,
    "viscpmchatbee": 1,
    "dac": 2,
    "colongpt-phi": 1,
    "trillsson_efficient": 1,
    "sapiens": 2,
    "custom_model": 2,
    "bert_vits2": 2,
    "bamboo": 3,
    "llava-jp": 7,
    "simple_image_classification": 1,
    "vcoder_llava": 2,
    "arctic": 2,
    "owlvit": 4,
    "codet5p_bimodal": 2,
    "segvol": 1,
    "tapex": 1,
    "mono": 2,
    "cambrian_phi3": 1,
    "solar": 11,
    "monkey": 2,
    "plm": 2,
    "phimoe": 3,
    "prokbert": 1,
    "cvt": 3,
    "VideoAutoencoderPipeline": 1,
    "univnet": 1,
    "turbosparsemixtral": 2,
    "bilingual": 4,
    "hgrn": 3,
    "diffusion_cond": 1,
    "quiet": 3,
    "nvretriever": 1,
    "extended-llama": 3,
    "emova": 3,
    "eagle_chat": 3,
    "prophetnet": 4,
    "mixsense_llama": 1,
    "moe_llava_mistral": 1,
    "h2ovl_chat": 2,
    "segment_enformer": 1,
    "boomer": 1,
    "argonne": 1,
    "deberta_arg_classifier": 1,
    "openlm": 3,
    "vcoder_ds_llava": 1,
    "prot2text": 2,
    "adaptformer": 1,
    "git_llama": 3,
    "mitsua_japanese_clip": 1,
    "aragpt2": 2,
    "gau_alpha": 1,
    "plamo2": 7,
    "mobilenet_v3": 2,
    "modnet": 2,
    "clip_text_model": 2,
    "dpt": 14,
    "encoder_decoder": 3,
    "STDiT3": 1,
    "minigpt4_video": 1,
    "kanana2vec": 1,
    "clip-vision-bert": 3,
    "liltrobertalike": 2,
    "mm_llms": 2,
    "skywork_moe": 2,
    "deta": 4,
    "ferret_llama": 1,
    "ExtendedVGCNBert": 1,
    "hubert_ecg": 2,
    "ibert": 2,
    "ChatNT": 1,
    "mplug_owl2": 6,
    "csg-vl-wukong": 1,
    "ma2za/roberta-emotion": 1,
    "transfo-xl": 1,
    "chexagent": 1,
    "btlm": 4,
    "maira2": 1,
    "siglip_vision_model": 3,
    "travisionlm": 3,
    "stable-diffusion": 1,
    "flamingo": 3,
    "pixtral": 3,
    "llava_cohere": 1,
    "chadavit": 1,
    "bunny-phi3": 2,
    "vision-text-dual-encoder": 8,
    "DINO-HuVITS": 1,
    "cambrian_llama": 3,
    "gan": 1,
    "mipha_phi": 1,
    "mplug_docowl": 1,
    "etchat_phi3": 1,
    "flm": 1,
    "pulse2pulse-2": 1,
    "fuyu": 2,
    "data2vec-text": 3,
    "skywork": 11,
    "ResNet": 2,
    "rnabert": 1,
    "vargpt_qwen2_vl": 2,
    "mega": 7,
    "Dream": 2,
    "kosmos-2": 3,
    "chatrex": 1,
    "LanguageBindVideo": 5,
    "transnormer": 2,
    "gr00t_n1": 2,
    "zhiyin": 1,
    "crystalcoder": 2,
    "cxrmate-ed": 1,
    "flaubert2": 1,
    "lxmert": 1,
    "xlm-roberta-xl": 3,
    "cxr-bert": 1,
    "seq2seq": 1,
    "scold": 1,
    "rt_detr": 9,
    "ProbUNet": 1,
    "kphi3": 1,
    "mosaic_gpt": 3,
    "imp": 2,
    "custom_decoder_only_t5": 1,
    "roc_bert": 1,
    "switchgpt2": 1,
    "gla": 2,
    "moshi": 2,
    "sarashina2_vision": 2,
    "style_text_to_speech_2": 3,
    "songgen": 1,
    "hyperclovax_vlm": 1,
    "mgm": 2,
    "image-classification": 1,
    "superglue": 3,
    "omdet-turbo": 2,
    "llamavision": 5,
    "longcoder": 1,
    "videollama3_vision_encoder": 2,
    "llava_stablelm_epoch": 2,
    "pytorch": 1,
    "llava_next_video": 6,
    "beit3_llava": 2,
    "wav2vec2-conformer": 5,
    "ola_qwen": 3,
    "huginn_raven": 2,
    "sam_hq": 1,
    "rag": 2,
    "vit_mae": 2,
    "TransformerTextClassificationModel": 1,
    "hierarchical-xlm-roberta-xl": 1,
    "replit_lm": 2,
    "infimm-zephyr": 1,
    "llava_crystal": 1,
    "imagegpt": 1,
    "dna_encoder": 1,
    "mra": 1,
    "openba": 2,
    "any_model": 1,
    "bbsnet": 1,
    "groupvit": 2,
    "fegeo-llama": 1,
    "YOLO11n": 2,
    "fcn4flare": 1,
    "mammo": 1,
    "vitmatte": 1,
    "deberta_semantic_similarity": 1,
    "T5": 2,
    "textnet": 3,
    "gpt_neox_japanese": 1,
    "spatiallm_llama": 1,
    "bit_llama": 2,
    "visual_bert": 1,
    "tinyllm": 1,
    "gigaam-ctc": 1,
    "speech_llama": 2,
    "re_gpt": 1,
    "vargpt_llava": 1,
    "MAELM": 1,
    "BERT_CRF": 3,
    "patchtsmixer": 1,
    "tinytimemixer": 3,
    "ubke": 1,
    "moss": 10,
    "plamo": 5,
    "ConvNet": 1,
    "shikra": 1,
    "gpt2a": 3,
    "mingru": 3,
    "layoutdm_fidnet_v3": 1,
    "dasheng": 1,
    "omnilmm": 2,
    "space_explore_ai_financial": 1,
    "spark-tts": 1,
    "gazelle": 3,
    "magma": 1,
    "patchtst": 3,
    "moonshine": 5,
    "videorefer_qwen2": 3,
    "starvector": 2,
    "openai-gpt": 3,
    "llama_moe": 8,
    "xgboost": 1,
    "time_series_transformer": 1,
    "hummingbird": 1,
    "pubmedbert-bio-ext-summ": 1,
    "Phi-3": 1,
    "indic_vits_model": 1,
    "internvl": 5,
    "dass": 1,
    "COCOM": 2,
    "yolov8-seg": 1,
    "VQVAE": 1,
    "nllb-moe": 2,
    "vila_u_llama": 1,
    "evomistral": 1,
    "libra": 2,
    "starcoder": 7,
    "llava_mini_llama": 1,
    "aimv2": 2,
    "retrieva-bert": 2,
    "videollama2_mistral": 5,
    "Sparrow": 1,
    "mini_gemini_qwen2": 1,
    "llava_vistral": 1,
    "distilwhisper": 3,
    "transcorem": 1,
    "NVLM_D": 2,
    "SciDFM": 1,
    "vit-hybrid": 1,
    "rnaernie": 1,
    "fastconformer": 1,
    "dinat": 1,
    "BiGS": 3,
    "BertABSAForSequenceClassification": 1,
    "isnet": 1,
    "IAA": 1,
    "bark": 8,
    "megrezo": 1,
    "instructcir_llava_phi35": 1,
    "ben": 1,
    "dptdepth": 1,
    "ofa": 1,
    "SE.02": 1,
    "mert_model": 4,
    "gpt-neox": 4,
    "cogvlm2": 2,
    "bros": 2,
    "ModularStarEncoder": 1,
    "seamlessm4t-v2-large-speech_encoder": 1,
    "orion_moe": 1,
    "char_xmod": 1,
    "rf_detr": 2,
    "col": 1,
    "timesfm": 1,
    "elasticbert": 3,
    "crammedBERT": 2,
    "InternLMXComposer": 4,
    "mlcd": 1,
    "chartmoe": 1,
    "japanese_clip": 1,
    "olmo_1124": 1,
    "ltgbert": 1,
    "dcformer": 1,
    "rene": 1,
    "grinmoe": 1,
    "hibiki": 2,
    "pathumma_audio": 1,
    "shape_opt": 2,
    "M-CLIP": 1,
    "gpt": 1,
    "SENTINEL-CAND-MQM": 1,
    "blenderbot-small": 2,
    "fairseq_t5": 3,
    "csm": 2,
    "phi3_v_moe": 1,
    "proprime": 1,
    "llava_mixtral": 2,
    "LanguageBindThermal": 1,
    "donut": 3,
    "bamba": 6,
    "xlstm": 2,
    "Emu3VisionVQ": 1,
    "pmod_llava_llama": 2,
    "kclgpt": 1,
    "prismatic": 1,
    "perceiver-ar-symbolic-audio-model": 1,
    "dolphin": 1,
    "pico_decoder": 1,
    "siglip2": 2,
    "cased": 1,
    "blazeface": 1,
    "shieldgemma2": 1,
    "piguard": 1,
    "SMI-TED": 1,
    "rnafm": 1,
    "ferret_gemma": 1,
    "SPIDERPatchClassifier": 4,
    "perceiver-ar-causal-language-model": 1,
    "infimm-vicuna": 1,
    "ai-image-detector": 1,
    "memory_transformer": 1,
    "glpn": 3,
    "flash_t5": 1,
    "Or4cl3": 1,
    "pharia-v1": 2,
    "ardisplay": 1,
    "inf5": 1,
    "parakeet_ctc": 2,
    "mdlm": 2,
    "gigarembed": 1,
    "clip-vision-mbart": 1,
    "G2PTL": 1,
    "bleurt": 1,
    "segment_borzoi": 1,
    "typhoonaudio": 1,
    "qwen2idae": 1,
    "Bilingual": 2,
    "backpack-gpt2": 1,
    "euclid_qwen2": 2,
    "original_transformer": 1,
    "efficientnetv25": 1,
    "geov": 2,
    "step1": 2,
    "gemini_nano_v2": 1,
    "EfficientNetB0": 1,
    "lola_v1": 1,
    "yayi": 1,
    "tiny_llava_phi": 1,
    "sprite_generator": 1,
    "gptj_moe": 1,
    "mle": 1,
    "akshara": 1,
    "lir-dpr": 1,
    "grok": 2,
    "hybrid-clip": 2,
    "camelidae": 3,
    "phi-llava": 2,
    "evabyte": 2,
    "llava-chat-vector": 1,
    "m3d_clip": 1,
    "yolov8": 1,
    "Soundwave": 1,
    "ProSST": 2,
    "skywork_chat": 1,
    "exp": 1,
    "stdit": 2,
    "SententenceTransformerSentimentClassifier": 1,
    "midm-bitext-S": 1,
    "world_model": 1,
    "mitre": 2,
    "pyTorchModel": 1,
    "cerule-gemma": 2,
    "infimm-hd": 1,
    "diffllama": 1,
    "kosmos-2.5": 2,
    "sagvit": 1,
    "pop2piano": 1,
    "kraken": 3,
    "phi2": 1,
    "GraphLlama": 1,
    "agent_qwen2_vl": 2,
    "AutoModelForCausalLM": 1,
    "grok-1": 1,
    "lingowhale": 1,
    "videollama3_qwen2": 4,
    "blast_llama": 1,
    "SMT": 1,
    "turingMM": 1,
    "omni": 1,
    "imp_phi3": 1,
    "hlm": 1,
    "unispeech-sat": 1,
    "qwen2_vl_vpt": 1,
    "valley": 1,
    "got_ocr2": 1,
    "molformer": 1,
    "granite_speech": 2,
    "ctrl": 2,
    "fibonacci": 1,
    "mgp-str": 2,
    "qwen2moe": 1,
    "yolo": 1,
    "UNet": 1,
    "clyp": 1,
    "levit": 2,
    "lamed_llama": 1,
    "YAYIUIE": 1,
    "manta": 2,
    "neuralnet": 1,
    "e5rope": 1,
    "detime": 1,
    "jasper_vl": 1,
    "codify": 2,
    "time_moe": 2,
    "finvoc2vec": 1,
    "gpt_neox_reward_model": 1,
    "rita": 2,
    "multimodal_llama": 1,
    "Transformer3DModel": 1,
    "jat": 1,
    "cobald_parser": 1,
    "custom_gpt": 1,
    "git_japanese_stablelm_alpha": 2,
    "tiny_llava_stablelm": 1,
    "loconet": 1,
    "ristretto": 1,
    "voice_restore": 1,
    "SENTINEL-REF-MQM": 1,
    "HHEMv2Config": 1,
    "tiny_transformer": 1,
    "rna_torsionbert": 1,
    "v-jepa": 1,
    "Multimodal AI Model": 1,
    "cloob": 1,
    "cetaceanet": 1,
    "vstream": 1,
    "mplugdocowl": 1,
    "resnet50": 1,
    "darwinlm": 1,
    "SENTINEL-CAND-DA": 1,
    "MoE++": 1,
    "fast_esm": 1,
    "agimodel": 1,
    "custom": 1,
    "bucket-memory-model3": 1,
    "xmodelvlm": 1,
    "moonvit": 1,
    "moe_llava_qwen1_5": 1,
    "OneChart": 1,
    "llavallama": 1,
    "tcmoe": 1,
    "MAE": 1,
    "splade": 1,
    "mova": 1,
    "pvt": 1,
    "pega": 1,
    "VAE": 1,
    "trol": 1,
    "nue_asr": 1,
    "sapnous_t1": 1,
    "roberta_for_cl": 1,
    "autoencoder": 1,
    "shrink": 1,
    "imp_qwen2": 1,
    "magi": 1,
    "Vietnamese": 1,
    "elysium": 1,
    "cost_wise_gemma": 1,
    "minimax_vl_01": 1,
    "mot": 1,
    "qwen2_5_omni": 2,
    "clip_qwen2vl": 1,
    "swin_transformer": 1,
    "geblm": 1,
    "relation_detr": 1,
    "longformer-bio-ext-summ": 1,
    "codet5p_embedding": 1,
    "CSDModel": 1,
    "deci_lm": 1,
    "spec_vision": 1,
    "emuru": 1,
    "flava": 1,
    "augvit": 1,
    "Provence": 1,
    "simple_stories_4m": 1,
    "minimax_text_01": 1,
    "pvc_internvl": 1,
    "tio": 1,
    "llava_monet": 1,
    "titan": 1,
    "sesame": 1,
    "gemma3mm": 1,
    "zero_swot_encoder": 1,
    "en2fr_transformer": 1,
    "unispeech": 1,
    "spec2": 1,
    "italia": 1,
    "nabla_vl": 1,
    "Tharo.G-Eco": 1
}


# ============================================================
# Deterministic recovery maps
# ============================================================

MODEL_TYPE_DIRECT_MAP = {
    "electra": ("ELECTRA", ""),
    "mpnet": ("MPNet", ""),
    "m2m-100": ("M2M-100", ""),
    "longformer": ("Longformer", ""),
    "led": ("LED", ""),
    "layoutlm": ("LayoutLM", ""),
    "layoutlmv2": ("LayoutLM", "LayoutLMv2"),
    "layoutlmv3": ("LayoutLM", "LayoutLMv3"),
    "tapas": ("TAPAS", ""),
    "ernie": ("ERNIE", ""),
    "git": ("GIT", ""),
    "videomae": ("VideoMAE", ""),
    "audio-spectrogram-transformer": ("AST", ""),
    "vision-encoder-decoder": ("VisionEncoderDecoder", ""),
    "stablelm": ("StableLM", ""),
    "stablelm-epoch": ("StableLM", "Epoch"),
    "stablelm-alpha": ("StableLM", "Alpha"),
    "depth-anything": ("Depth Anything", ""),
}

WRAPPER_FAMILIES = {
    "llava": ("LLaVA", ""),
    "llava-next": ("LLaVA", "LLaVA-Next"),
    "llava-onevision": ("LLaVA", "LLaVA-OneVision"),
    "video-llava": ("Video-LLaVA", ""),
    "videollama": ("VideoLLaMA", ""),
    "videollama2": ("VideoLLaMA", "VideoLLaMA2"),
    "videollama3": ("VideoLLaMA", "VideoLLaMA3"),
    "tinyllava": ("TinyLLaVA", ""),
    "tiny-llava": ("TinyLLaVA", ""),
    "bunny": ("Bunny", ""),
    "vipllava": ("LLaVA", "VIP-LLaVA"),
    "moe-llava": ("LLaVA", "MoE-LLaVA"),
}

BACKBONE_HINTS = {
    "qwen2": ("Qwen", "Qwen2"),
    "qwen2-5": ("Qwen", "Qwen2.5"),
    "qwen3": ("Qwen", "Qwen3"),
    "qwen": ("Qwen", ""),
    "mistral": ("Mistral", ""),
    "mixtral": ("Mistral", "Mixtral"),
    "llama": ("LLaMA", ""),
    "phi3": ("Phi", "Phi-3"),
    "phi4": ("Phi", "Phi-4"),
    "phi": ("Phi", ""),
    "gemma3": ("Gemma", "Gemma 3"),
    "gemma2": ("Gemma", "Gemma 2"),
    "gemma": ("Gemma", ""),
    "stablelm-epoch": ("StableLM", "Epoch"),
    "stablelm": ("StableLM", ""),
    "mpt": ("GPT", "MPT"),
    "jamba": ("GPT", "Jamba"),
    "cohere": ("GPT", "Cohere"),
}

# ============================================================
# Utilities
# ============================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def lower_clean(text: str) -> str:
    return normalize_space(text).lower()


def parse_listish(value) -> List[str]:
    s = safe_str(value)
    if not s:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [normalize_space(x) for x in parsed if normalize_space(x)]
        except Exception:
            pass

    parts = re.split(r"[|,;/]", s)
    return [normalize_space(p) for p in parts if normalize_space(p)]


def strip_org_prefix(name: str) -> str:
    name = safe_str(name)
    if "/" in name:
        return name.split("/")[-1]
    return name


def normalize_model_name(name: str) -> str:
    s = strip_org_prefix(name).lower()
    s = s.replace("_", "-")
    s = s.replace(".", "-")
    s = re.sub(r"[^a-z0-9\-\+]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def normalize_model_type_name(s: str) -> str:
    s = safe_str(s)
    if not s:
        return ""
    s = s.replace("/", "-")
    s = s.replace("_", "-")
    s = s.replace(".", "-")
    s = re.sub(r"[^A-Za-z0-9\-\+]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s.lower()


def title_case_slug(s: str) -> str:
    s = safe_str(s).replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in s.split())


def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, safe_str(a).lower(), safe_str(b).lower()).ratio()


def parse_composite_family(*values: str) -> Optional[Dict[str, str]]:
    text = " ".join(safe_str(v) for v in values if safe_str(v))
    if not text:
        return None

    nm = normalize_model_name(text)
    if not nm:
        return None

    wrapper = None
    wrapper_key = ""
    backbone = None

    wrapper_candidates = sorted(WRAPPER_FAMILIES.items(), key=lambda x: len(x[0]), reverse=True)
    for key, val in wrapper_candidates:
        if re.search(rf"(?:^|-)({re.escape(key)})(?:-|$)", nm):
            wrapper = val
            wrapper_key = key
            break

    if not wrapper:
        return None

    backbone_candidates = sorted(BACKBONE_HINTS.items(), key=lambda x: len(x[0]), reverse=True)
    for key, val in backbone_candidates:
        if key == wrapper_key:
            continue
        if re.search(rf"(?:^|-)({re.escape(key)})(?:-|$)", nm):
            backbone = val
            break

    out = {
        "family_root": wrapper[0],
        "family_child": wrapper[1],
        "backbone_family": backbone[0] if backbone else "",
        "backbone_child": backbone[1] if backbone else "",
        "family_confidence": 0.85,
    }
    return out


def candidate_model_type_hit_rate(df: pd.DataFrame, cand: str) -> float:
    if df.empty or "model_type" not in df.columns:
        return 0.0

    vals = df["model_type"].fillna("").astype(str).map(normalize_model_type_name)
    if vals.empty:
        return 0.0

    cand = normalize_model_name(cand)
    if not cand:
        return 0.0

    return float(vals.str.contains(re.escape(cand), regex=True).mean())


# ============================================================
# Noise / stop words / variants
# ============================================================

STOP_TOKENS = {
    "base", "large", "small", "medium", "tiny", "mini", "micro", "nano",
    "xl", "xxl", "xlarge", "xxlarge",
    "v1", "v2", "v3", "v4", "v5",
    "preview", "latest", "experimental",

    "chat", "instruct", "instruction", "sft", "rlhf", "dpo",
    "aligned", "alignment", "uncensored", "reasoning",
    "tool", "function", "functioncall",

    "hf", "fp16", "fp32", "bf16", "int4", "int8", "gguf", "awq", "gptq", "exl2",

    "1b", "2b", "3b", "4b", "5b", "6b", "7b", "8b", "9b", "10b",
    "11b", "12b", "13b", "14b", "15b", "20b", "22b", "27b", "30b",
    "32b", "34b", "40b", "65b", "70b", "72b", "110b",

    "model", "models", "text", "image", "vision", "audio", "video",
    "multimodal", "vl", "vlm",
    "classifier", "classification", "generation", "generator",
    "embedding", "embeddings", "embed", "retrieval", "reranker", "ranker",
    "reward", "detector", "segmentation", "captioning", "forecasting",

    "finetuned", "fine", "tuned", "adapter", "lora", "merged",
    "checkpoint", "weights", "pretrained", "distilled", "moe", "dense",

    "meta", "facebook", "microsoft", "google", "openai", "nvidia", "huggingface",

    "multilingual", "english", "japanese", "chinese", "arabic",
    "medical", "finance", "legal", "code", "general",
}

BAD_FAMILY_TOKENS = set(STOP_TOKENS) | {
    "other", "unclear", "new", "best", "fast", "faster", "pro", "plus",
}

NEGATIVE_HINTS = {
    "chat": 0.05,
    "instruct": 0.05,
    "classifier": 0.10,
    "classification": 0.10,
    "embedding": 0.10,
    "reranker": 0.10,
    "ranker": 0.10,
    "awq": 0.08,
    "gptq": 0.08,
    "gguf": 0.08,
    "fp16": 0.06,
    "int8": 0.06,
    "int4": 0.06,
    "lora": 0.08,
    "adapter": 0.08,
}


# ============================================================
# Parent-child hints
# ============================================================

CHILD_HINTS = {
    "llama-2": ("LLaMA", "LLaMA 2"),
    "llama-3": ("LLaMA", "LLaMA 3"),
    "llama-3-1": ("LLaMA", "LLaMA 3.1"),
    "llama-3-2": ("LLaMA", "LLaMA 3.2"),
    "llama4": ("LLaMA", "LLaMA 4"),
    "llama-4": ("LLaMA", "LLaMA 4"),
    "code-llama": ("LLaMA", "CodeLlama"),
    "codellama": ("LLaMA", "CodeLlama"),
    "tinyllama": ("LLaMA", "TinyLLaMA"),
    "mllama": ("LLaMA", "mLLaMA"),
    "longllama": ("LLaMA", "LongLLaMA"),

    "stable-diffusion-xl": ("Stable Diffusion", "SDXL"),
    "sdxl": ("Stable Diffusion", "SDXL"),
    "stable-diffusion-3": ("Stable Diffusion", "SD3"),
    "stable-diffusion-2": ("Stable Diffusion", "SD 2.x"),
    "stable-diffusion-1": ("Stable Diffusion", "SD 1.x"),

    "flan-t5": ("T5", "FLAN-T5"),
    "mt5": ("T5", "mT5"),
    "byt5": ("T5", "ByT5"),
    "longt5": ("T5", "LongT5"),
    "umt5": ("T5", "UMT5"),

    "xlm-roberta": ("RoBERTa", "XLM-RoBERTa"),
    "xlm-roberta-xl": ("RoBERTa", "XLM-RoBERTa XL"),

    "deberta-v2": ("DeBERTa", "DeBERTa v2"),
    "deberta-v3": ("DeBERTa", "DeBERTa v3"),

    "mixtral": ("Mistral", "Mixtral"),
    "mistral3": ("Mistral", "Mistral 3"),
    "vmistral": ("Mistral", "VMistral"),
    "pixtral": ("Mistral", "Pixtral"),

    "qwen2": ("Qwen", "Qwen2"),
    "qwen2-5": ("Qwen", "Qwen2.5"),
    "qwen-2": ("Qwen", "Qwen2"),
    "qwen-2-5": ("Qwen", "Qwen2.5"),
    "qwen2-vl": ("Qwen", "Qwen2-VL"),
    "qwen2-5-vl": ("Qwen", "Qwen2.5-VL"),
    "qwen2-audio": ("Qwen", "Qwen2-Audio"),
    "qwen3": ("Qwen", "Qwen3"),

    "phi-2": ("Phi", "Phi-2"),
    "phi-3": ("Phi", "Phi-3"),
    "phi-4": ("Phi", "Phi-4"),
    "phi2": ("Phi", "Phi-2"),
    "phi3": ("Phi", "Phi-3"),
    "phi4": ("Phi", "Phi-4"),
    "phi3small": ("Phi", "Phi-3-small"),

    "dinov2": ("ViT", "DINOv2"),
    "dinov2-with-registers": ("ViT", "DINOv2 with registers"),

    "mobile-sam": ("SAM", "MobileSAM"),
    "sam2-video": ("SAM", "SAM2 Video"),

    "gemma2": ("Gemma", "Gemma 2"),
    "gemma3": ("Gemma", "Gemma 3"),
    "paligemma": ("Gemma", "PaliGemma"),
    "recurrent-gemma": ("Gemma", "RecurrentGemma"),

    "llava-next": ("LLaVA", "LLaVA-Next"),
    "llava-onevision": ("LLaVA", "LLaVA-OneVision"),

    "idefics2": ("IDEFICS", "IDEFICS2"),
    "idefics3": ("IDEFICS", "IDEFICS3"),

    "florence2": ("Florence", "Florence-2"),
    "modernbert": ("BERT", "ModernBERT"),
    "nomic-bert": ("BERT", "NomicBERT"),
    "eurobert": ("BERT", "EuroBERT"),
    "megatron-bert": ("BERT", "Megatron-BERT"),
    "mobilebert": ("BERT", "MobileBERT"),
    "convbert": ("BERT", "ConvBERT"),
    "camembert": ("BERT", "CamemBERT"),
    "flaubert": ("BERT", "FlauBERT"),
    "xlnet": ("Transformer-XL family", "XLNet"),
    "rwkv5": ("RWKV", "RWKV-5"),
    "rwkv6": ("RWKV", "RWKV-6"),
    "rwkv7": ("RWKV", "RWKV-7"),
}


# ============================================================
# Explicit family patterns
# ============================================================

FAMILY_PATTERNS = [
    r"(llama(?:-[0-9](?:-[0-9])?)?)",
    r"(llama4|llama-4|mllama|longllama|code-llama|codellama|tinyllama)",
    r"(mistral|mixtral|mistral3|vmistral|pixtral)",
    r"(qwen(?:[0-9](?:-[0-9])?)?)",
    r"(qwen-[0-9](?:-[0-9])?)",
    r"(qwen2-vl|qwen2-5-vl|qwen2-audio|qwen3)",
    r"(phi(?:-[0-9])?|phi2|phi3|phi4|phi3small)",
    r"(gemma|gemma2|gemma3|codegemma|recurrentgemma|pali(?:-)?gemma)",
    r"(stable-diffusion(?:-[a-z0-9]+)?)",
    r"(sdxl|sd-?1-?5|sd-?2(?:-[a-z0-9]+)?|sd-?3(?:-[a-z0-9]+)?)",
    r"(controlnet)",
    r"(bert|distilbert|roberta|modernbert|nomic-bert|eurobert|xlm-roberta|deberta|albert|scibert|biobert|pubmedbert|clinicalbert|camembert|mobilebert|megatron-bert|convbert)",
    r"(t5|flan-t5|mt5|byt5|longt5|umt5|bart|mbart|marian|pegasus)",
    r"(gpt2|gpt-j|gptj|gpt-neo|gpt-neox|gpt-neox|gpt|opt|bloom|falcon|mpt|mamba|olmo|olmo2|dbrx|granite|cohere|cohere2|deepseek|deepseek-v2|deepseek-v3|chatglm|internlm|internlm2|internlm3|jamba|exaone|jais|nemotron|moss|openelm|baichuan|yi|zamba|zamba2|orion|aquila|xverse|yuan|starcoder|starcoder2|codegen|gpt-bigcode)",
    r"(clip|openclip|siglip|siglip2|vit|vision-transformer|deit|beit|dino|dinov2)",
    r"(yolo(?:v[0-9]+)?|yolos|yolov10|yolov9|yolov8|rt-detr|detr)",
    r"(sam|segment-anything|mobile-sam|sam2-video)",
    r"(resnet|resnext|unet|u-net|convnext|convnextv2|swin|swinv2|mobilenet-v1|mobilenet-v2|mobilenet-v3|mobilevit|mobilevitv2|efficientnet|efficientformer|segformer)",
    r"(whisper|distilwhisper|lite-whisper|faster-whisper|wav2vec2?|hubert|xls-r|speecht5|audioldm|wavlm|parler-tts|musicgen|bark|vits)",
    r"(blip(?:-2)?|llava(?:-next)?|idefics|kosmos|instructblip|paligemma|florence2|internvl|minicpmv|mobilevlm|ovis|xgenmm|fuyu|pix2struct|chameleon|cogvlm2|mplug-owl)",
    r"(esm2?|proteinbert|protbert|openfold|alphafold|esmfold)",
    r"(graphsage|graph-attention-network|gat|graph-convolutional-network|gcn|graphormer)",
    r"(timegpt|patchtst|timesfm)",
]


# ============================================================
# Seed catalog
# ============================================================

def seed_family_catalog() -> Dict[str, Dict]:
    return {
        "BERT": {
            "aliases": [
                "bert", "bert-base", "bert-large", "modernbert", "nomic-bert",
                "eurobert", "megatron-bert", "mobilebert", "convbert", "camembert",
                "flaubert"
            ],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "RoBERTa": {
            "aliases": ["roberta", "xlm-roberta", "xlmr", "xlm-roberta-xl"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "DistilBERT": {
            "aliases": ["distilbert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "DeBERTa": {
            "aliases": ["deberta", "deberta-v2", "deberta-v3"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "ALBERT": {
            "aliases": ["albert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing"],
        },
        "BioBERT": {
            "aliases": ["biobert", "clinicalbert", "pubmedbert", "bluebert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing", "Scientific and Domain-specific AI"],
        },
        "SciBERT": {
            "aliases": ["scibert"],
            "parent": "BERT",
            "domain_hints": ["Natural Language Processing", "Scientific and Domain-specific AI"],
        },
        "T5": {
            "aliases": ["t5", "flan-t5", "mt5", "byt5", "longt5", "umt5"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "BART": {
            "aliases": ["bart", "mbart", "marian", "pegasus"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "GPT-2": {
            "aliases": ["gpt2", "openai-gpt", "backpack-gpt2"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "GPT-Neo": {
            "aliases": ["gpt-neo", "gpt_neo"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "GPT-J": {
            "aliases": ["gpt-j", "gptj", "nanogpt-j"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "GPT-NeoX": {
            "aliases": ["gpt-neox", "gpt_neox", "gpt-neox-japanese"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "GPT": {
            "aliases": [
                "gpt", "gpt_bigcode", "gpt-bigcode", "opt", "mpt", "mamba",
                "openelm", "moss", "granite", "cohere", "cohere2",
                "deepseek", "deepseek-v2", "deepseek-v3", "chatglm",
                "internlm", "internlm2", "internlm3", "exaone", "jamba",
                "jais", "nemotron", "xverse", "yuan", "dbrx", "zamba",
                "zamba2", "orion", "aquila", "baichuan", "yi", "olmo", "olmo2"
            ],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "ELECTRA": {
            "aliases": ["electra"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "MPNet": {
            "aliases": ["mpnet"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "M2M-100": {
            "aliases": ["m2m-100", "m2m_100"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "Longformer": {
            "aliases": ["longformer"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "LED": {
            "aliases": ["led"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "LayoutLM": {
            "aliases": ["layoutlm", "layoutlmv2", "layoutlmv3"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Computer Vision", "Multimodal AI"],
        },
        "TAPAS": {
            "aliases": ["tapas", "tapex"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "ERNIE": {
            "aliases": ["ernie", "ernie-m", "rnaernie"],
            "parent": "",
            "domain_hints": ["Natural Language Processing"],
        },
        "GIT": {
            "aliases": ["git", "git-llama"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Multimodal AI"],
        },
        "VideoMAE": {
            "aliases": ["videomae", "videomaev2-base"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Video AI"],
        },
        "AST": {
            "aliases": ["audio-spectrogram-transformer"],
            "parent": "",
            "domain_hints": ["Audio"],
        },
        "StableLM": {
            "aliases": ["stablelm", "stablelm-epoch", "stablelm-alpha"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Depth Anything": {
            "aliases": ["depth-anything", "prompt-depth-anything"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "VisionEncoderDecoder": {
            "aliases": ["vision-encoder-decoder"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Multimodal AI"],
        },
        "LLaMA": {
            "aliases": [
                "llama", "llama-2", "llama-3", "llama-3.1", "llama-3.2", "llama4",
                "llama-4", "code-llama", "codellama", "tinyllama", "mllama", "longllama"
            ],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Mistral": {
            "aliases": ["mistral", "mixtral", "mistral3", "vmistral", "pixtral"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Falcon": {
            "aliases": ["falcon"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "BLOOM": {
            "aliases": ["bloom", "bloomz"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Qwen": {
            "aliases": ["qwen", "qwen2", "qwen2.5", "qwen-2", "qwen-2-5", "qwen2-vl", "qwen2.5-vl", "qwen3"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Phi": {
            "aliases": ["phi", "phi-2", "phi-3", "phi-4", "phi2", "phi3", "phi4", "phi3small"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
        "Gemma": {
            "aliases": ["gemma", "gemma2", "gemma3", "codegemma", "recurrentgemma", "paligemma"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI", "Multimodal AI"],
        },
        "Stable Diffusion": {
            "aliases": ["stable-diffusion", "stable-diffusion-xl", "sdxl", "sd-1-5", "sd-2", "sd-3", "latent-diffusion"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Generative AI"],
        },
        "ControlNet": {
            "aliases": ["controlnet"],
            "parent": "Stable Diffusion",
            "domain_hints": ["Computer Vision", "Generative AI"],
        },
        "ViT": {
            "aliases": ["vit", "vision-transformer", "deit", "beit", "dino", "dinov2"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "CLIP": {
            "aliases": ["clip", "openclip", "siglip", "siglip2"],
            "parent": "",
            "domain_hints": ["Computer Vision", "Multimodal AI"],
        },
        "YOLO": {
            "aliases": ["yolo", "yolov5", "yolov8", "yolov9", "yolov10", "yolos"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "SAM": {
            "aliases": ["segment-anything", "sam", "mobile-sam", "sam2-video"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "ResNet": {
            "aliases": ["resnet", "resnext", "convnext", "convnextv2", "regnet"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "U-Net": {
            "aliases": ["unet", "u-net"],
            "parent": "",
            "domain_hints": ["Computer Vision"],
        },
        "Whisper": {
            "aliases": ["whisper", "faster-whisper", "distilwhisper", "lite-whisper"],
            "parent": "",
            "domain_hints": ["Speech and Audio", "Generative AI"],
        },
        "wav2vec": {
            "aliases": ["wav2vec", "wav2vec2", "hubert", "xls-r", "wavlm"],
            "parent": "",
            "domain_hints": ["Speech and Audio"],
        },
        "SpeechT5": {
            "aliases": ["speecht5"],
            "parent": "",
            "domain_hints": ["Speech and Audio"],
        },
        "AudioLDM": {
            "aliases": ["audioldm", "musicgen", "bark", "parler-tts", "vits"],
            "parent": "",
            "domain_hints": ["Speech and Audio", "Generative AI"],
        },
        "BLIP": {
            "aliases": ["blip", "blip-2", "instructblip"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "LLaVA": {
            "aliases": ["llava", "llava-next", "llava-onevision"],
            "parent": "LLaMA",
            "domain_hints": ["Multimodal AI"],
        },
        "IDEFICS": {
            "aliases": ["idefics", "idefics2", "idefics3"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "Kosmos": {
            "aliases": ["kosmos", "kosmos-2", "kosmos-2.5"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "Florence": {
            "aliases": ["florence2"],
            "parent": "",
            "domain_hints": ["Multimodal AI"],
        },
        "ESM": {
            "aliases": ["esm", "esm2", "proteinbert", "protbert"],
            "parent": "",
            "domain_hints": ["Scientific and Domain-specific AI"],
        },
        "AlphaFold-like": {
            "aliases": ["alphafold", "openfold", "esmfold"],
            "parent": "",
            "domain_hints": ["Scientific and Domain-specific AI"],
        },
        "GraphSAGE": {
            "aliases": ["graphsage"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "GAT": {
            "aliases": ["gat", "graph-attention-network"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "GCN": {
            "aliases": ["gcn", "graph-convolutional-network"],
            "parent": "",
            "domain_hints": ["Graph and Relational Learning"],
        },
        "TimeGPT": {
            "aliases": ["timegpt"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
        "PatchTST": {
            "aliases": ["patchtst"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
        "TimesFM": {
            "aliases": ["timesfm"],
            "parent": "",
            "domain_hints": ["Time Series and Forecasting"],
        },
        "RWKV": {
            "aliases": ["rwkv", "rwkv5", "rwkv6", "rwkv7"],
            "parent": "",
            "domain_hints": ["Natural Language Processing", "Generative AI"],
        },
    }


def build_alias_to_family(catalog: Dict[str, Dict]) -> Dict[str, str]:
    alias_to_family = {}
    for fam, cfg in catalog.items():
        alias_to_family[normalize_model_name(fam)] = fam
        for alias in cfg.get("aliases", []):
            alias_to_family[normalize_model_name(alias)] = fam
    return alias_to_family


def build_model_type_hints(
    model_type_counts: Dict[str, int],
    alias_to_family: Dict[str, str],
    min_count: int = 2,
) -> Dict[str, Dict[str, str]]:
    """
    Maps normalized model_type -> {family_root, family_child, source}
    source is:
      - seeded     if it canonicalizes to a seed family
      - discovered if it looks like a valid new root/child family
    """
    catalog = seed_family_catalog()
    hints: Dict[str, Dict[str, str]] = {}

    for raw_mt, count in model_type_counts.items():
        if int(count) < min_count:
            continue

        mt = normalize_model_type_name(raw_mt)
        if not mt:
            continue

        if mt in MODEL_TYPE_DIRECT_MAP:
            fam, child = MODEL_TYPE_DIRECT_MAP[mt]
            hints[mt] = {
                "family_root": fam,
                "family_child": child,
                "source": "seeded",
            }
            continue

        composite = parse_composite_family(raw_mt)
        if composite:
            hints[mt] = {
                "family_root": composite["family_root"],
                "family_child": composite["family_child"],
                "source": "seeded",
            }
            continue

        if mt in alias_to_family:
            fam = alias_to_family[mt]
            parent = catalog.get(fam, {}).get("parent", "")
            if parent:
                hints[mt] = {
                    "family_root": parent,
                    "family_child": fam,
                    "source": "seeded",
                }
            else:
                hints[mt] = {
                    "family_root": fam,
                    "family_child": "",
                    "source": "seeded",
                }
            continue

        _, parent, child = normalize_child_family_candidate(mt)
        if parent:
            hints[mt] = {
                "family_root": parent,
                "family_child": child,
                "source": "seeded" if normalize_model_name(parent) in alias_to_family else "discovered",
            }
            continue

        if mt in CHILD_HINTS:
            parent, child = CHILD_HINTS[mt]
            hints[mt] = {
                "family_root": parent,
                "family_child": child,
                "source": "seeded" if normalize_model_name(parent) in alias_to_family else "discovered",
            }
            continue

        cand = extract_candidate_from_name(mt)
        cand = normalize_model_name(cand)

        if cand and is_valid_family_candidate(cand, alias_to_family):
            known_root, _ = canonicalize_candidate(cand, alias_to_family, discovered_roots=None)
            if known_root:
                hints[mt] = {
                    "family_root": known_root,
                    "family_child": "",
                    "source": "seeded",
                }
            else:
                hints[mt] = {
                    "family_root": title_case_slug(cand),
                    "family_child": "",
                    "source": "discovered",
                }

    return hints


# ============================================================
# Attribute parsing
# ============================================================

def parse_attributes(name: str) -> Dict[str, str]:
    nm = normalize_model_name(name)
    out = {
        "size_variant": "",
        "tuning_variant": "",
        "quantization_variant": "",
        "domain_variant": "",
    }

    m = re.search(r"\b(\d+(?:-\d+)?b)\b", nm)
    if m:
        out["size_variant"] = m.group(1)

    for tok in ["chat", "instruct", "base", "sft", "rlhf", "dpo"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["tuning_variant"] = tok
            break

    for tok in ["awq", "gptq", "gguf", "fp16", "fp32", "bf16", "int8", "int4", "exl2"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["quantization_variant"] = tok
            break

    for tok in ["multilingual", "medical", "legal", "finance", "code"]:
        if re.search(rf"\b{re.escape(tok)}\b", nm):
            out["domain_variant"] = tok
            break

    return out


# ============================================================
# Candidate extraction
# ============================================================

def tokens_from_name(s: str) -> List[str]:
    s = normalize_model_name(s)
    if not s:
        return []
    return [t for t in s.split("-") if t]


def remove_noise_tokens(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if tok in STOP_TOKENS:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        if re.fullmatch(r"v\d+", tok):
            continue
        if re.fullmatch(r"\d+x\d+", tok):
            continue
        if re.fullmatch(r"\d+(?:-\d+)?b", tok):
            continue
        out.append(tok)
    return out


def infer_child_from_name(name: str) -> Tuple[str, str]:
    nm = normalize_model_name(name)
    if not nm:
        return ("", "")

    for key, val in CHILD_HINTS.items():
        if key in nm:
            return val

    _, parent, child = normalize_child_family_candidate(nm)
    if parent:
        return (parent, child)

    return ("", "")


def extract_candidate_from_name(name: str) -> str:
    """
    Precision-first extraction:
    - pattern-first
    - child hint aware
    - no arbitrary fallback to first token
    """
    nm = normalize_model_name(name)
    if not nm:
        return ""

    parent, child = infer_child_from_name(nm)
    if parent and child:
        for key in CHILD_HINTS:
            if key in nm:
                return key

    for pat in FAMILY_PATTERNS:
        m = re.search(pat, nm)
        if m:
            return m.group(1)

    toks = remove_noise_tokens(tokens_from_name(nm))
    joined = "-".join(toks)

    for pat in [
        r"stable-diffusion(?:-[a-z0-9]+)?",
        r"latent-diffusion",
        r"segment-anything",
        r"vision-transformer",
        r"graph-attention-network",
        r"graph-convolutional-network",
        r"xlm-roberta",
        r"faster-whisper",
    ]:
        m = re.search(pat, joined)
        if m:
            return m.group(0)

    return ""


def is_valid_family_candidate(cand: str, alias_to_family: Optional[Dict[str, str]] = None) -> bool:
    c = normalize_model_name(cand)
    if looks_like_child_not_root(c):
        return False
    if not c:
        return False

    if alias_to_family and c in alias_to_family:
        return True

    if c in CHILD_HINTS:
        return True

    toks = [t for t in c.split("-") if t]
    if not toks:
        return False

    if len(c) < 3:
        return False

    if re.fullmatch(r"[0-9\-\.]+", c):
        return False

    if any(tok in BAD_FAMILY_TOKENS for tok in toks):
        return False

    if all(re.fullmatch(r"(v\d+|\d+|\d+(?:-\d+)?b)", t) for t in toks):
        return False

    if len(toks) == 1 and toks[0] in {
        "text", "image", "audio", "vision", "base", "chat", "instruct", "model"
    }:
        return False

    return True


def extract_candidate_root(row: pd.Series, alias_to_family: Dict[str, str]) -> str:
    """
    Priority:
    1. base_model
    2. model_type
    3. model_name
    4. model_id

    But only return valid family candidates.
    """
    for field in ["base_models", "model_type", "model_name", "model_id"]:
        raw = safe_str(row.get(field, ""))
        cand = extract_candidate_from_name(raw)
        if cand and is_valid_family_candidate(cand, alias_to_family):
            return cand
    return ""


def normalize_child_family_candidate(cand: str) -> Tuple[str, str, str]:
    """
    Returns (normalized_candidate, parent_root, child_family)
    If cand looks like a child/version of a known family, normalize it here.
    Otherwise returns (cand, "", "")
    """
    c = normalize_model_name(cand)
    if not c:
        return ("", "", "")

    sd_patterns = [
        (r"^sd[- ]?1[- ]?5$", "Stable Diffusion", "SD 1.5"),
        (r"^sd15$", "Stable Diffusion", "SD 1.5"),
        (r"^stable-diffusion[- ]?1[- ]?5$", "Stable Diffusion", "SD 1.5"),

        (r"^sd[- ]?2[- ]?1$", "Stable Diffusion", "SD 2.1"),
        (r"^sd21$", "Stable Diffusion", "SD 2.1"),
        (r"^stable-diffusion[- ]?2[- ]?1$", "Stable Diffusion", "SD 2.1"),

        (r"^sd[- ]?3$", "Stable Diffusion", "SD 3"),
        (r"^stable-diffusion[- ]?3$", "Stable Diffusion", "SD 3"),

        (r"^sd[- ]?3[- ]?5$", "Stable Diffusion", "SD 3.5"),
        (r"^sd35$", "Stable Diffusion", "SD 3.5"),
        (r"^stable-diffusion[- ]?3[- ]?5$", "Stable Diffusion", "SD 3.5"),

        (r"^sdxl$", "Stable Diffusion", "SDXL"),
        (r"^stable-diffusion-xl$", "Stable Diffusion", "SDXL"),
    ]

    for pat, parent, child in sd_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    llama_patterns = [
        (r"^llama[- ]?1$", "LLaMA", "LLaMA 1"),
        (r"^llama[- ]?2$", "LLaMA", "LLaMA 2"),
        (r"^llama[- ]?3$", "LLaMA", "LLaMA 3"),
        (r"^llama[- ]?3[- ]?1$", "LLaMA", "LLaMA 3.1"),
        (r"^llama[- ]?3[- ]?2$", "LLaMA", "LLaMA 3.2"),
        (r"^llama[- ]?4$", "LLaMA", "LLaMA 4"),
        (r"^llama4$", "LLaMA", "LLaMA 4"),
        (r"^code[- ]?llama$", "LLaMA", "CodeLlama"),
        (r"^codellama$", "LLaMA", "CodeLlama"),
        (r"^tinyllama$", "LLaMA", "TinyLLaMA"),
        (r"^mllama$", "LLaMA", "mLLaMA"),
        (r"^longllama$", "LLaMA", "LongLLaMA"),
    ]

    for pat, parent, child in llama_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    qwen_patterns = [
        (r"^qwen[- ]?2$", "Qwen", "Qwen2"),
        (r"^qwen2$", "Qwen", "Qwen2"),
        (r"^qwen[- ]?2[- ]?5$", "Qwen", "Qwen2.5"),
        (r"^qwen2[- ]?5$", "Qwen", "Qwen2.5"),
        (r"^qwen2[- ]?vl$", "Qwen", "Qwen2-VL"),
        (r"^qwen2[- ]?5[- ]?vl$", "Qwen", "Qwen2.5-VL"),
        (r"^qwen2[- ]?audio$", "Qwen", "Qwen2-Audio"),
        (r"^qwen3$", "Qwen", "Qwen3"),
    ]

    for pat, parent, child in qwen_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    phi_patterns = [
        (r"^phi[- ]?2$", "Phi", "Phi-2"),
        (r"^phi[- ]?3$", "Phi", "Phi-3"),
        (r"^phi[- ]?4$", "Phi", "Phi-4"),
        (r"^phi2$", "Phi", "Phi-2"),
        (r"^phi3$", "Phi", "Phi-3"),
        (r"^phi4$", "Phi", "Phi-4"),
        (r"^phi3small$", "Phi", "Phi-3-small"),
    ]

    for pat, parent, child in phi_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    gemma_patterns = [
        (r"^gemma2$", "Gemma", "Gemma 2"),
        (r"^gemma3$", "Gemma", "Gemma 3"),
        (r"^paligemma$", "Gemma", "PaliGemma"),
        (r"^recurrent[- ]?gemma$", "Gemma", "RecurrentGemma"),
    ]

    for pat, parent, child in gemma_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    llava_patterns = [
        (r"^llava[- ]?next$", "LLaVA", "LLaVA-Next"),
        (r"^llava[- ]?onevision$", "LLaVA", "LLaVA-OneVision"),
    ]

    for pat, parent, child in llava_patterns:
        if re.fullmatch(pat, c):
            return (c, parent, child)

    return (c, "", "")


# ============================================================
# Matching
# ============================================================

@dataclass
class FamilyMatch:
    family_root: str = ""
    family_child: str = ""
    method: str = ""
    confidence: float = 0.0
    evidence: Optional[Dict] = None


def build_row_text_for_match(row: pd.Series) -> str:
    pieces = [
        safe_str(row.get("model_id", "")),
        safe_str(row.get("model_name", "")),
        safe_str(row.get("short_description", "")),
        safe_str(row.get("pipeline_tag", "")),
        safe_str(row.get("library_name", "")),
        safe_str(row.get("model_type", "")),
        safe_str(row.get("base_models", "")),
        " ".join(parse_listish(row.get("tags", ""))),
    ]
    return lower_clean(" | ".join([p for p in pieces if p]))


def score_negative_hints(text: str) -> float:
    penalty = 0.0
    for tok, val in NEGATIVE_HINTS.items():
        if re.search(rf"\b{re.escape(tok)}\b", text):
            penalty += val
    return min(penalty, 0.25)


def known_family_match(
    row: pd.Series,
    catalog: Dict[str, Dict],
    alias_to_family: Dict[str, str],
    model_type_hints: Optional[Dict[str, Dict[str, str]]] = None,
) -> FamilyMatch:
    domain = safe_str(row.get("assigned_modality", ""))
    model_name = safe_str(row.get("model_name", row.get("model_id", "")))
    model_id = safe_str(row.get("model_id", ""))
    base_model = safe_str(row.get("base_models", ""))
    model_type = safe_str(row.get("model_type", ""))
    norm_model_type = normalize_model_type_name(model_type)
    text = build_row_text_for_match(row)

    if model_type_hints and norm_model_type in model_type_hints:
        mt_hint = model_type_hints[norm_model_type]
        return FamilyMatch(
            family_root=mt_hint["family_root"],
            family_child=mt_hint["family_child"],
            method="model_type_hint",
            confidence=0.93,
            evidence={"matched_on_model_type": norm_model_type, "family_source": mt_hint["source"]},
        )

    fields = [base_model, model_name, model_id, model_type]
    norm_fields = [normalize_model_name(f) for f in fields if safe_str(f)]

    for f in norm_fields:
        parent, child = infer_child_from_name(f)
        if parent:
            return FamilyMatch(
                family_root=parent,
                family_child=child,
                method="known_child_pattern",
                confidence=0.97,
                evidence={"matched_on": f}
            )

    scores = defaultdict(float)

    for fam, cfg in catalog.items():
        domain_hints = cfg.get("domain_hints", [])
        domain_bonus = 0.15 if (not domain_hints or domain in domain_hints) else 0.0

        aliases = [normalize_model_name(a) for a in cfg.get("aliases", [])] + [normalize_model_name(fam)]
        aliases = sorted(set([a for a in aliases if a]))

        for alias in aliases:
            if normalize_model_name(base_model) and alias in normalize_model_name(base_model):
                scores[fam] += 0.85 + domain_bonus

            if normalize_model_name(model_name) and alias in normalize_model_name(model_name):
                scores[fam] += 0.50 + domain_bonus

            if normalize_model_name(model_type) and alias in normalize_model_name(model_type):
                scores[fam] += 0.65 + domain_bonus

            if alias in text:
                scores[fam] += 0.18 + domain_bonus

    if not scores:
        return FamilyMatch()

    penalty = score_negative_hints(text)
    for fam in list(scores.keys()):
        scores[fam] = max(0.0, scores[fam] - penalty)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fam1, score1 = ranked[0]
    fam2, score2 = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = score1 - score2

    if score1 >= 0.85:
        method = "known_base_model_or_strong_alias"
    elif score1 >= 0.45:
        method = "known_name_alias"
    elif score1 >= 0.25:
        method = "known_metadata_alias"
    else:
        return FamilyMatch()

    if score1 < 0.28 or margin < 0.08:
        return FamilyMatch()

    parent = catalog.get(fam1, {}).get("parent", "")
    family_root = parent if parent else fam1
    family_child = fam1 if parent else ""

    return FamilyMatch(
        family_root=family_root,
        family_child=family_child,
        method=method,
        confidence=round(min(score1, 0.99), 6),
        evidence={"top_score": score1, "margin": margin, "penalty": penalty}
    )


# ============================================================
# Canonicalization
# ============================================================

def fuzzy_threshold(s: str) -> float:
    n = len(s)
    if n <= 4:
        return 0.985
    if n <= 7:
        return 0.96
    return 0.93


def canonicalize_candidate(
    cand: str,
    alias_to_family: Dict[str, str],
    discovered_roots: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    c = normalize_model_name(cand)
    if not c:
        return ("", "")

    if c in alias_to_family:
        return (alias_to_family[c], "seed_alias_exact")

    if c in CHILD_HINTS:
        return (CHILD_HINTS[c][0], "child_hint_parent")

    best_family = ""
    best_score = 0.0
    for alias, fam in alias_to_family.items():
        score = seq_sim(c, alias)
        if score > best_score:
            best_family = fam
            best_score = score
    if best_score >= fuzzy_threshold(c):
        return (best_family, "seed_alias_fuzzy")

    if discovered_roots:
        if c in discovered_roots:
            return (discovered_roots[c], "discovered_exact")

        best_root = ""
        best_score = 0.0
        for k, v in discovered_roots.items():
            score = seq_sim(c, k)
            if score > best_score:
                best_root = v
                best_score = score
        if best_score >= max(0.96, fuzzy_threshold(c)):
            return (best_root, "discovered_fuzzy")

    return ("", "")


# ============================================================
# Discovery statistics / coherence
# ============================================================

def token_positions_for_candidate(series: pd.Series, cand: str) -> List[int]:
    positions = []
    root_token = cand.split("-")[0]
    for x in series.fillna("").astype(str):
        toks = normalize_model_name(x).split("-")
        if root_token in toks:
            positions.append(toks.index(root_token))
    return positions


def candidate_base_model_hit_rate(df: pd.DataFrame, cand: str) -> float:
    if df.empty:
        return 0.0
    vals = df["base_models"].fillna("").astype(str).map(normalize_model_name)
    return float(vals.str.contains(re.escape(cand), regex=True).mean())


def candidate_neighbor_coherence(series: pd.Series, cand: str) -> float:
    left = []
    right = []
    root_tok = cand.split("-")[0]

    for x in series.fillna("").astype(str):
        toks = normalize_model_name(x).split("-")
        for i, tok in enumerate(toks):
            if tok == root_tok:
                if i > 0:
                    left.append(toks[i - 1])
                if i < len(toks) - 1:
                    right.append(toks[i + 1])

    def top_share(vals: List[str]) -> float:
        vals = [v for v in vals if v and v not in STOP_TOKENS]
        if not vals:
            return 0.0
        c = Counter(vals)
        return c.most_common(1)[0][1] / max(1, len(vals))

    return max(top_share(left), top_share(right))


def candidate_is_heterogeneous(
    subset: pd.DataFrame,
    catalog: Dict[str, Dict],
    alias_to_family: Dict[str, str],
    model_type_hints: Optional[Dict[str, Dict[str, str]]] = None,
) -> bool:
    roots = []
    for _, row in subset.iterrows():
        km = known_family_match(row, catalog, alias_to_family, model_type_hints=model_type_hints)
        if km.family_root:
            roots.append(km.family_root)
    return len(set(roots)) >= 4


# ============================================================
# Promotion / discovery
# ============================================================

def promote_discovered_candidates(
    df_unresolved: pd.DataFrame,
    catalog: Dict[str, Dict],
    alias_to_family: Dict[str, str],
    model_type_hints: Optional[Dict[str, Dict[str, str]]] = None,
    min_count_global: int = 12,
    min_count_per_domain: int = 5,
    min_base_model_rate: float = 0.35,
    max_median_position: float = 2.0,
    min_coherence: float = 0.30,
) -> pd.DataFrame:
    """
    Discover new family roots only from unresolved rows.
    This is much safer than mining from the whole dataset.
    """
    work = df_unresolved.copy()
    if work.empty:
        return pd.DataFrame()

    work["candidate_root_raw"] = work.apply(lambda r: extract_candidate_root(r, alias_to_family), axis=1)
    work["candidate_root_norm"] = work["candidate_root_raw"].apply(normalize_model_name)
    work = work[work["candidate_root_norm"] != ""].copy()

    if work.empty:
        return pd.DataFrame()

    rows = []

    domain_counts = (
        work.groupby(["assigned_modality", "candidate_root_norm"])
        .size()
        .reset_index(name="domain_count")
    )
    global_counts = (
        work.groupby("candidate_root_norm")
        .size()
        .reset_index(name="global_count")
    )
    merged = domain_counts.merge(global_counts, on="candidate_root_norm", how="left")

    for _, r in merged.iterrows():
        cand = safe_str(r["candidate_root_norm"])
        domain = safe_str(r["assigned_modality"])
        global_count = int(r["global_count"])
        domain_count = int(r["domain_count"])

        if not cand:
            continue

        if not is_valid_family_candidate(cand, alias_to_family):
            continue

        if global_count < min_count_global and domain_count < min_count_per_domain:
            continue

        subset = work[work["candidate_root_norm"] == cand].copy()
        if subset.empty:
            continue

        if candidate_is_heterogeneous(subset, catalog, alias_to_family, model_type_hints=model_type_hints):
            continue

        base_rate = candidate_base_model_hit_rate(subset, cand)
        model_type_rate = candidate_model_type_hit_rate(subset, cand)
        positions = token_positions_for_candidate(subset["model_name"], cand)
        median_pos = median(positions) if positions else 999
        coherence = candidate_neighbor_coherence(subset["model_name"], cand)

        relaxed_ok = (
            global_count >= 4
            and model_type_rate >= 0.60
        )

        if not relaxed_ok and base_rate < min_base_model_rate and median_pos > max_median_position:
            continue

        if not relaxed_ok and coherence < min_coherence and global_count < (min_count_global * 2):
            continue

        cand_norm2, forced_parent, forced_child = normalize_child_family_candidate(cand)
        if forced_parent:
            rows.append({
                "assigned_modality": domain,
                "candidate_root_norm": cand,
                "candidate_global_count": global_count,
                "candidate_domain_count": domain_count,
                "base_model_hit_rate": round(base_rate, 4),
                "model_type_hit_rate": round(model_type_rate, 4),
                "median_name_token_position": round(float(median_pos), 4) if median_pos != 999 else 999.0,
                "neighbor_coherence": round(coherence, 4),
                "canonical_or_discovered_root": forced_parent,
                "canonicalization_source": "forced_child_parent_mapping",
                "is_new_discovered_root": False,
                "family_source": "seeded" if normalize_model_name(forced_parent) in alias_to_family else "discovered",
                "suggested_parent_root": forced_parent,
                "suggested_child_family": forced_child,
            })
            continue

        canonical_root, source = canonicalize_candidate(cand, alias_to_family, discovered_roots=None)

        if canonical_root:
            label = canonical_root
            promoted = False
            family_source = "seeded"
        else:
            label = title_case_slug(cand)
            promoted = True
            family_source = "discovered"

        parent, child = infer_child_from_name(cand)

        rows.append({
            "assigned_modality": domain,
            "candidate_root_norm": cand,
            "candidate_global_count": global_count,
            "candidate_domain_count": domain_count,
            "base_model_hit_rate": round(base_rate, 4),
            "median_name_token_position": round(float(median_pos), 4) if median_pos != 999 else 999.0,
            "neighbor_coherence": round(coherence, 4),
            "canonical_or_discovered_root": label,
            "canonicalization_source": source if source else "new_discovered_root",
            "is_new_discovered_root": promoted,
            "family_source": family_source,
            "suggested_parent_root": parent,
            "suggested_child_family": child,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["is_new_discovered_root", "candidate_global_count", "candidate_domain_count", "base_model_hit_rate"],
        ascending=[False, False, False, False]
    ).drop_duplicates(
        subset=["assigned_modality", "candidate_root_norm", "canonical_or_discovered_root"]
    )

    return out


# ============================================================
# Relations
# ============================================================

def build_family_relations(
    catalog: Dict[str, Dict],
    discovered_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for fam, cfg in catalog.items():
        parent = safe_str(cfg.get("parent", ""))
        if parent:
            rows.append({
                "parent_family": parent,
                "child_family": fam,
                "relation_source": "seed_catalog",
                "assigned_modality": "|".join(cfg.get("domain_hints", [])),
            })

    if discovered_df is not None and not discovered_df.empty:
        for _, r in discovered_df.iterrows():
            root = safe_str(r["canonical_or_discovered_root"])
            parent = safe_str(r.get("suggested_parent_root", ""))
            child = safe_str(r.get("suggested_child_family", ""))
            domain = safe_str(r.get("assigned_modality", ""))

            if parent and child:
                rows.append({
                    "parent_family": parent,
                    "child_family": child,
                    "relation_source": "discovered_child_hint",
                    "assigned_modality": domain,
                })

            cand = safe_str(r.get("candidate_root_norm", ""))
            p2, c2 = infer_child_from_name(cand)
            if p2 and c2:
                rows.append({
                    "parent_family": p2,
                    "child_family": c2,
                    "relation_source": "candidate_pattern",
                    "assigned_modality": domain,
                })

            if parent and root and root != parent and root != child:
                rows.append({
                    "parent_family": parent,
                    "child_family": root,
                    "relation_source": "discovered_root_parent_inference",
                    "assigned_modality": domain,
                })

    if not rows:
        return pd.DataFrame(columns=["parent_family", "child_family", "relation_source", "assigned_modality"])

    rel = pd.DataFrame(rows).drop_duplicates()
    rel = rel[rel["parent_family"] != rel["child_family"]].copy()
    rel = rel.sort_values(["parent_family", "child_family", "relation_source"])
    return rel


# ============================================================
# Assignment
# ============================================================

def build_discovered_root_lookup(discovered_df: pd.DataFrame) -> Dict[str, str]:
    lookup = {}
    if discovered_df is None or discovered_df.empty:
        return lookup

    for _, r in discovered_df.iterrows():
        cand = normalize_model_name(r["candidate_root_norm"])
        root = safe_str(r["canonical_or_discovered_root"])
        if cand and root:
            lookup[cand] = root
    return lookup

def build_model_family_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Map normalized model_name / model_id -> assigned family fields.
    Used for base_model inheritance in a second pass.
    """
    lookup = {}

    for _, r in df.iterrows():
        fam = safe_str(r.get("family_root", ""))
        if not fam or fam == "Other / Unclear":
            continue

        payload = {
            "family_root": fam,
            "family_child": safe_str(r.get("family_child", "")),
            "backbone_family": safe_str(r.get("backbone_family", "")),
            "backbone_child": safe_str(r.get("backbone_child", "")),
        }

        for key in ["model_name", "model_id"]:
            nm = normalize_model_name(r.get(key, ""))
            if nm:
                lookup[nm] = payload

    return lookup


def inherit_from_base_models(row, model_family_lookup: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    If this row is unresolved but has base_models pointing to a model with a known family,
    inherit that family.
    """
    base_models = parse_listish(row.get("base_models", ""))

    for bm in base_models:
        bm_norm = normalize_model_name(bm)
        if bm_norm in model_family_lookup:
            hit = model_family_lookup[bm_norm]
            return {
                "family_root": hit.get("family_root", ""),
                "family_child": hit.get("family_child", ""),
                "backbone_family": hit.get("backbone_family", ""),
                "backbone_child": hit.get("backbone_child", ""),
                "assignment_method": "base_model_inherited",
                "family_confidence": 0.85,
                "family_source": "inherited",
            }

    return None

def assign_with_discovery(
    df: pd.DataFrame,
    catalog: Dict[str, Dict],
    discovered_df: pd.DataFrame,
    model_type_hints: Optional[Dict[str, Dict[str, str]]] = None,
    model_family_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    
    alias_to_family = build_alias_to_family(catalog)
    discovered_lookup = build_discovered_root_lookup(discovered_df)
    if model_family_lookup is None:
        model_family_lookup = {}

    rel_hint_map = {}
    if discovered_df is not None and not discovered_df.empty:
        for _, r in discovered_df.iterrows():
            cand = normalize_model_name(r["candidate_root_norm"])
            rel_hint_map[cand] = {
                "parent": safe_str(r.get("suggested_parent_root", "")),
                "child": safe_str(r.get("suggested_child_family", "")),
                "root": safe_str(r.get("canonical_or_discovered_root", "")),
                "family_source": safe_str(r.get("family_source", "")),
            }

    results = []

    for _, row in df.iterrows():
        rowd = row.to_dict()
        rowd.setdefault("backbone_family", "")
        rowd.setdefault("backbone_child", "")

        attrs = parse_attributes(" ".join([
            safe_str(row.get("model_name", "")),
            safe_str(row.get("model_id", "")),
            safe_str(row.get("base_models", "")),
            safe_str(row.get("model_type", "")),
        ]))
        rowd.update(attrs)

        km = known_family_match(row, catalog, alias_to_family, model_type_hints=model_type_hints)
        if km.family_root:
            rowd["family_root"] = km.family_root
            rowd["family_child"] = km.family_child
            rowd["assignment_method"] = km.method
            rowd["family_confidence"] = km.confidence
            rowd["family_source"] = "seeded"
            rowd["candidate_root_raw"] = extract_candidate_root(row, alias_to_family)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            rowd.setdefault("backbone_family", "")
            rowd.setdefault("backbone_child", "")
            results.append(rowd)
            continue

        mt = normalize_model_type_name(row.get("model_type", ""))
        if mt in MODEL_TYPE_DIRECT_MAP:
            fam, child = MODEL_TYPE_DIRECT_MAP[mt]
            rowd["family_root"] = fam
            rowd["family_child"] = child
            rowd["assignment_method"] = "model_type_direct"
            rowd["family_confidence"] = 0.90
            rowd["family_source"] = "seeded"
            rowd["candidate_root_raw"] = extract_candidate_root(row, alias_to_family)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            results.append(rowd)
            continue

        composite = parse_composite_family(
            row.get("model_name", ""),
            row.get("model_id", ""),
            row.get("model_type", ""),
            row.get("base_models", ""),
        )
        if composite:
            rowd.update(composite)
            rowd["assignment_method"] = "composite"
            rowd["family_source"] = "seeded"
            rowd["candidate_root_raw"] = extract_candidate_root(row, alias_to_family)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            results.append(rowd)
            continue

        # NEW: inherit from base_models if possible
        inherited = inherit_from_base_models(row, model_family_lookup)
        if inherited:
            rowd.update(inherited)
            rowd["candidate_root_raw"] = extract_candidate_root(row, alias_to_family)
            rowd["candidate_root_norm"] = normalize_model_name(rowd["candidate_root_raw"])
            results.append(rowd)
            continue

        cand_raw = extract_candidate_root(row, alias_to_family)
        cand_norm = normalize_model_name(cand_raw)

        p, c = infer_child_from_name(cand_norm)
        if p:
            rowd["family_root"] = p
            rowd["family_child"] = c
            rowd["assignment_method"] = "candidate_child_pattern"
            rowd["family_confidence"] = 0.90
            rowd["family_source"] = "seeded" if normalize_model_name(p) in alias_to_family else "discovered"
            rowd["candidate_root_raw"] = cand_raw
            rowd["candidate_root_norm"] = cand_norm
            results.append(rowd)
            continue

        root, source = canonicalize_candidate(cand_norm, alias_to_family, discovered_lookup)

        if root:
            hint = rel_hint_map.get(cand_norm, {})
            child = safe_str(hint.get("child", ""))
            parent = safe_str(hint.get("parent", ""))
            source_flag = "discovered" if "discovered" in source else "seeded"

            if parent and not child and root != parent:
                child = root
                root = parent

            rowd["family_root"] = root
            rowd["family_child"] = child
            rowd["assignment_method"] = f"candidate_{source}"
            rowd["family_confidence"] = 0.76 if "discovered" in source else 0.84
            rowd["family_source"] = source_flag
            rowd["candidate_root_raw"] = cand_raw
            rowd["candidate_root_norm"] = cand_norm
            results.append(rowd)
            continue

        rowd["family_root"] = "Other / Unclear"
        rowd["family_child"] = ""
        rowd["assignment_method"] = "unresolved"
        rowd["family_confidence"] = 0.0
        rowd["family_source"] = "unresolved"
        rowd["candidate_root_raw"] = cand_raw
        rowd["candidate_root_norm"] = cand_norm
        results.append(rowd)

    return pd.DataFrame(results)


def prune_bad_discovered_roots(discovered_df: pd.DataFrame) -> pd.DataFrame:
    if discovered_df is None or discovered_df.empty:
        return discovered_df

    bad_root_patterns = [
        r"^Sd\d+.*$",
        r"^SD\d+.*$",
        r"^Llama \d+.*$",
        r"^Phi[- ]?\d+.*$",
        r"^Qwen[- ]?\d+.*$",
    ]

    keep_rows = []
    for _, row in discovered_df.iterrows():
        root = safe_str(row.get("canonical_or_discovered_root", ""))
        if any(re.fullmatch(p, root) for p in bad_root_patterns):
            if safe_str(row.get("suggested_parent_root", "")):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
        else:
            keep_rows.append(True)

    return discovered_df.loc[keep_rows].copy()


# ============================================================
# Two-stage pipeline
# ============================================================

def stage1_known_assignments(
    df: pd.DataFrame,
    catalog: Dict[str, Dict],
    model_type_hints: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    alias_to_family = build_alias_to_family(catalog)
    rows = []

    for _, row in df.iterrows():
        km = known_family_match(row, catalog, alias_to_family, model_type_hints=model_type_hints)
        rows.append({
            "model_id": safe_str(row.get("model_id", "")),
            "known_family_root": km.family_root,
            "known_family_child": km.family_child,
            "known_match_method": km.method,
            "known_match_confidence": km.confidence,
        })

    return pd.DataFrame(rows)


def looks_like_child_not_root(c: str) -> bool:
    c = normalize_model_name(c)
    child_like_patterns = [
        r"^sd\d+$",
        r"^sd[- ]?\d[- ]?\d$",
        r"^sd[- ]?\d(?:[- ]?\d)?$",
        r"^llama[- ]?\d(?:[- ]?\d)?$",
        r"^qwen[- ]?\d(?:[- ]?\d)?$",
        r"^phi[- ]?\d$",
    ]
    return any(re.fullmatch(p, c) for p in child_like_patterns)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with Level 1 modalities already assigned")
    parser.add_argument("--out_assignments", default="family_assignmecnts.csv")
    parser.add_argument("--out_relations", default="family_relations.csv")
    parser.add_argument("--out_candidates", default="discovered_family_candidates.csv")
    parser.add_argument("--min_count_global", type=int, default=12)
    parser.add_argument("--min_count_per_domain", type=int, default=5)
    parser.add_argument("--min_base_model_rate", type=float, default=0.35)
    parser.add_argument("--max_median_position", type=float, default=2.0)
    parser.add_argument("--min_coherence", type=float, default=0.30)

    args = parser.parse_args([
        "--input_csv", "7-CLUSTERING_MODELS/level1_domain_assignments_improved.csv",
        "--out_assignments", "7-CLUSTERING_MODELS/clusters_improved/family_assignments.csv",
        "--out_relations", "7-CLUSTERING_MODELS/clusters_improved/family_relations.csv",
        "--out_candidates", "7-CLUSTERING_MODELS/clusters_improved/discovered_family_candidates.csv"
    ])

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print("uploaded csv")
    if "assigned_modality" not in df.columns:
        raise ValueError("Missing required column: assigned_modality")

    catalog = seed_family_catalog()
    alias_to_family = build_alias_to_family(catalog)

    model_type_hints = build_model_type_hints(
        MODEL_TYPE_COUNTS,
        alias_to_family,
        min_count=2,
    )

    known_stage = stage1_known_assignments(df, catalog, model_type_hints=model_type_hints)
    df_work = df.copy()

    if "model_id" in df_work.columns and "model_id" in known_stage.columns:
        df_work = df_work.merge(known_stage, on="model_id", how="left")
    else:
        known_stage = known_stage.reset_index(drop=True)
        df_work = df_work.reset_index(drop=True)
        df_work = pd.concat([df_work, known_stage], axis=1)

    unresolved_mask = df_work["known_family_root"].fillna("").eq("")
    df_unresolved = df_work[unresolved_mask].copy()

    print(f"Loaded rows: {len(df)}")
    print(f"Known-family resolved in stage 1: {len(df) - len(df_unresolved)}")
    print(f"Unresolved rows for discovery: {len(df_unresolved)}")

    discovered_df = promote_discovered_candidates(
        df_unresolved=df_unresolved,
        catalog=catalog,
        alias_to_family=alias_to_family,
        model_type_hints=model_type_hints,
        min_count_global=args.min_count_global,
        min_count_per_domain=args.min_count_per_domain,
        min_base_model_rate=args.min_base_model_rate,
        max_median_position=args.max_median_position,
        min_coherence=args.min_coherence,
    )
    discovered_df = prune_bad_discovered_roots(discovered_df)

    # Pass 1: assign everything without base_model inheritance
    assigned_pass1 = assign_with_discovery(
        df_work,
        catalog,
        discovered_df,
        model_type_hints=model_type_hints,
        model_family_lookup=None,
    )

    # Build lookup from already assigned canonical models
    model_family_lookup = build_model_family_lookup(assigned_pass1)

    # Pass 2: re-run assignment with base_model inheritance enabled
    assigned = assign_with_discovery(
        df_work,
        catalog,
        discovered_df,
        model_type_hints=model_type_hints,
        model_family_lookup=model_family_lookup,
    )

    relations = build_family_relations(
        catalog=catalog,
        discovered_df=discovered_df,
    )

    preferred = [
        "model_id",
        "model_name",
        "assigned_modality",
        "family_root",
        "family_child",
        "family_source",
        "assignment_method",
        "family_confidence",
        "candidate_root_raw",
        "candidate_root_norm",
        "backbone_family",
        "backbone_child",
        "size_variant",
        "tuning_variant",
        "quantization_variant",
        "domain_variant",
        "base_models",
        "model_type",
        "pipeline_tag",
        "library_name",
        "tags",
        "short_description",
    ]
    existing = [c for c in preferred if c in assigned.columns]
    others = [c for c in assigned.columns if c not in existing]
    assigned = assigned[existing + others]

    assigned.to_csv(args.out_assignments, index=False)
    relations.to_csv(args.out_relations, index=False)

    if discovered_df is None or discovered_df.empty:
        pd.DataFrame(columns=[
            "assigned_modality",
            "candidate_root_norm",
            "candidate_global_count",
            "candidate_domain_count",
            "base_model_hit_rate",
            "model_type_hit_rate",
            "median_name_token_position",
            "neighbor_coherence",
            "canonical_or_discovered_root",
            "canonicalization_source",
            "is_new_discovered_root",
            "family_source",
            "suggested_parent_root",
            "suggested_child_family",
        ]).to_csv(args.out_candidates, index=False)
    else:
        discovered_df.to_csv(args.out_candidates, index=False)

    print(f"Saved assignments: {args.out_assignments} ({len(assigned)} rows)")
    print(f"Saved relations:   {args.out_relations} ({len(relations)} rows)")
    print(f"Saved candidates:  {args.out_candidates} ({0 if discovered_df is None else len(discovered_df)} rows)")

    final_resolved = assigned["family_root"].fillna("").ne("").sum()
    final_unresolved = assigned["family_root"].fillna("").eq("").sum()

    print(f"Final resolved after assignment: {final_resolved}")
    print(f"Final unresolved after assignment: {final_unresolved}")
    
if __name__ == "__main__":
    print("doing stuf")
    main()