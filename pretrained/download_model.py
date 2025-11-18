import phonlp
import py_vncorenlp

phonlp.download(save_dir='./nlp_models/phonlp')
py_vncorenlp.download_model(save_dir='./nlp_models/vncorenlp')