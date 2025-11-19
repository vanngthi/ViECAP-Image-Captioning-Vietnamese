# import os
# import phonlp
# import py_vncorenlp
         

# def extract_caption(caption: str, vncorenlp_model, phonlp_model):
#     segmented = vncorenlp_model.word_segment(caption)[0]
    
#     tokens, pos_tags, ner_tags, deps = phonlp_model.annotate(text=segmented)
#     print(tokens, pos_tags, ner_tags, deps)
#     tokens = tokens[0]
#     pos_tags = [p[0] for p in pos_tags[0]]
#     ner_tags = ner_tags[0]
#     deps = deps[0]

#     raw_entities = []
#     for tok, pos, ner, dep in zip(tokens, pos_tags, ner_tags, deps):
#         head_idx, dep_label = dep
#         if (
#             pos in {"N", "Nc", "Np"} or 
#             dep_label in {"nmod", "sub", "dob", "pob"} or
#             ner != "O"
#         ):
#             raw_entities.append(tok.strip().lower())

#     merged_entities = []
#     temp = []
#     for tok, pos in zip(tokens, pos_tags):
#         if pos in {"N", "Nc", "Np"}:
#             temp.append(tok.lower())
#         else:
#             if temp:
#                 merged_entities.append(" ".join(temp))
#                 temp = []
#     if temp:
#         merged_entities.append(" ".join(temp))

#     merged_entities = [m.replace("_", " ").strip() for m in merged_entities]

#     filtered_raw = []
#     for r in raw_entities:
#         r_clean = r.replace("_", " ").strip()
#         if not any(r_clean in m for m in merged_entities):
#             filtered_raw.append(r_clean)

#     final = list(dict.fromkeys(merged_entities + filtered_raw))
#     return final


# jar_dir = os.path.abspath("../../pretrained/nlp_models/vncorenlp")
# pho_dir = os.path.abspath("../../pretrained/nlp_models/phonlp")

# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# os.environ["CLASSPATH"] = os.path.join(jar_dir, "VnCoreNLP-1.2.jar")

# vncorenlp_model = py_vncorenlp.VnCoreNLP(
#     save_dir=jar_dir,
#     annotators=["wseg"],
#     max_heap_size='-Xmx2g'
# )

# phonlp_model = phonlp.load(pho_dir)
        
# extract_caption("gậy bóng chày đang được vung lên bởi cầu thủ", vncorenlp_model=vncorenlp_model, phonlp_model=phonlp_model)

people_vocabs = [
    "người", "cầu thủ", "vận động viên", "đàn ông", "phụ nữ",
    "cô gái", "chàng trai", "bé trai", "bé gái", "cậu bé", "cô bé"
    # thêm tùy ý
]

def entities_process(
    detected_entities,
    people_vocabs,
):
    process_entities = []
    for i in range(len(detected_entities)):

        detected_entity = detected_entities[i].lower().strip()

        # 🌟 NEW: nếu entity chứa bất kỳ pattern người → chuẩn hóa thành "người"
        if any(p in detected_entity for p in people_vocabs):
            detected_entity = "người"
        process_entities.append(detected_entity)
    return process_entities

detected_entities = [
    "cầu thủ bóng đá",
    "vận động viên bóng rổ",
    "gậy bóng chày",
    "quả bóng rổ"
]

print(entities_process(
    detected_entities,
    people_vocabs))