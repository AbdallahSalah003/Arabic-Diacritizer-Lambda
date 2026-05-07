import numpy as np
import os
import onnxruntime as ort
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
ONNX_FILE = MODEL_DIR / "arabic_diacritizer.onnx"

#==================================================================================
DIACRITICS = {
    0x0652, 
    0x0651, 
    0x064D, 
    0x064E,  
    0x064C, 
    0x064F,  
    0x0650, 
    0x064B,
}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
NONE_LABEL = "<NONE>"
ARABIC_CHARS = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, " ",
    "أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", 
    "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ة", "ى", 
    "ء", "ؤ", "ئ", "إ", "آ",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", "،", "؟", "!", ":", "؛", "-", "(", ")", "[", "]"
]

CHAR_TO_ID = {char: idx for idx, char in enumerate(ARABIC_CHARS)}
ID_TO_CHAR = {idx: char for char, idx in CHAR_TO_ID.items()}
DIACRITIC_LABELS = [
    PAD_TOKEN,      
    NONE_LABEL,    
    chr(0x0652),    
    chr(0x064D),    
    chr(0x064E),
    chr(0x064C),    
    chr(0x064F),    
    chr(0x0650),    
    chr(0x064B),    
    chr(0x0651),    
    "".join(sorted([chr(0x0651), chr(0x064E)])), 
    "".join(sorted([chr(0x0651), chr(0x064F)])),
    "".join(sorted([chr(0x0651), chr(0x0650)])), 
    "".join(sorted([chr(0x0651), chr(0x064B)])), 
    "".join(sorted([chr(0x0651), chr(0x064C)])), 
    "".join(sorted([chr(0x0651), chr(0x064D)]))
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(DIACRITIC_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
#==================================================================================
class ONNXDiacritizer:
    def __init__(self, onnx_model_path: str, max_length: int = 512):
        so = ort.SessionOptions()
        
        # critical for Lambda: we limit threads to match the function's allocated vCPU
        # Lambda vCPUs = memory (MB) / 1769, roughly
        # for example, 3008 MB = ~2 vCPUs so 2-4 threads is optimal
        so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        so.inter_op_num_threads = 1 
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = False  # Saves memory during graph optimization
        so.enable_cpu_mem_arena = False  # Prevents ORT from hoarding memory

        self.session = ort.InferenceSession(
            onnx_model_path, 
            providers=['CPUExecutionProvider'],
            sess_options=so
        )
        
        self.max_length = max_length
        self.diacritics = DIACRITICS
        
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        

    def extract_chars_and_labels(self, text: str):
        chars = []
        
        for char in text:
            if ord(char) not in self.diacritics:
                chars.append(char)
        
        return chars

    def preprocess(self, text: str) -> dict:
        """
        Converts raw text into NumPy arrays for ONNX
        """
        chars = self.extract_chars_and_labels(text.strip())
        char_seq = [CLS_TOKEN] + chars + [SEP_TOKEN]
        input_ids = [CHAR_TO_ID.get(c, CHAR_TO_ID[UNK_TOKEN]) for c in char_seq]
        pad_len = self.max_length - len(input_ids)
        
        if pad_len > 0:
            input_ids.extend([CHAR_TO_ID[PAD_TOKEN]] * pad_len)
            attention_mask = [1] * len(char_seq) + [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length

        inputs = {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64)
        }
        
        return inputs, chars  

    def postprocess(self, logits: np.ndarray, original_chars: list) -> str:
        """
        Merges predicted diacritics back with the original characters
        """
        predicted_ids = np.argmax(logits, axis=-1)[0] 
        
        diacritized_chars = []
        for i, char in enumerate(original_chars):
            diacritic_id = predicted_ids[i + 1]
            diacritic = ID_TO_LABEL.get(diacritic_id, "")
            
            if diacritic in [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, UNK_TOKEN]:
                diacritic = ""
            elif diacritic == NONE_LABEL:
                diacritic = ""
            
            diacritized_chars.append(char + diacritic)
        
        return "".join(diacritized_chars)

    def __call__(self, text: str) -> str:
        if not text.strip():
            return ""
        
        ort_inputs, original_chars = self.preprocess(text)
        
        logits = self.session.run([self.output_name], ort_inputs)[0]
        
        return self.postprocess(logits, original_chars)

diacritizer = ONNXDiacritizer(ONNX_FILE, max_length=128)

def infer(text: str) -> str:
    return diacritizer(text)