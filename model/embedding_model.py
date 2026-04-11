from sentence_transformers import SentenceTransformer

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentences = ["Cô ấy là một người vui_tính .", "Cô ấy cười nói suốt cả ngày ."]

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
embeddings = model.encode(sentences)
print(embeddings)
