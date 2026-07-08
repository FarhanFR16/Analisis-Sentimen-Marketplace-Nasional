import re
import os
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# List of negation words that should NOT be removed as stopwords
NEGATION_WORDS = {
    'tidak', 'tidaklah', 'belum', 'belumlah', 'kurang', 'jangan', 
    'janganlah', 'bukan', 'bukanlah', 'tidak', 'enggak', 'ga', 
    'gak', 'gk', 'nda', 'kagak', 'tapi'
}

# Standard Indonesian Stopwords (excluding negation words)
INDONESIAN_STOPWORDS = {
    'yang', 'untuk', 'pada', 'ke', 'dari', 'di', 'dan', 'atau', 'ini', 'itu',
    'dengan', 'adalah', 'yaitu', 'yakni', 'seperti', 'sebagai', 'bagi', 'oleh',
    'untuk', 'telah', 'sudah', 'akan', 'bahwa', 'secara', 'tentang', 'terhadap',
    'dalam', 'luar', 'karena', 'olehnya', 'sehingga', 'maka', 'namun', 'tetapi',
    'serta', 'saja', 'hanya', 'pun', 'juga', 'lah', 'kah', 'tah', 'sih', 'kok',
    'loh', 'dong', 'deh', 'ya', 'ada', 'adalah', 'adanya', 'adapun', 'begitu',
    'begitupun', 'begitulah', 'bila', 'bilamana', 'demikian', 'demikianlah',
    'ia', 'ialah', 'kami', 'kamu', 'saya', 'dia', 'mereka', 'kita', 'anda',
    'kalian', 'kita', 'tahu', 'tahu-tahu', 'tahunya', 'tentu', 'tentunya',
    'tersebut', 'tersebutlah', 'tentang', 'tentang-tentang'
} - NEGATION_WORDS

# Common Marketplace Slang & Abbreviation Normalization Dictionary
SLANG_DICT = {
    'yg': 'yang', 'yangg': 'yang',
    'trs': 'terus', 'trus': 'terus',
    'tp': 'tapi', 'tapii': 'tapi',
    'dgn': 'dengan',
    'sy': 'saya', 'aq': 'saya', 'gw': 'saya', 'gua': 'saya',
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'ndak': 'tidak', 'kagak': 'tidak', 'enggak': 'tidak',
    'brg': 'barang', 'barangg': 'barang',
    'bgs': 'bagus', 'baguss': 'bagus', 'bgus': 'bagus',
    'lemot': 'lambat', 'lola': 'lambat', 'lelet': 'lambat',
    'males': 'malas',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah', 'udahh': 'sudah',
    'klo': 'kalau', 'kalo': 'kalau',
    'bgt': 'banget', 'bngt': 'banget', 'bangett': 'banget',
    'cpt': 'cepat', 'cepet': 'cepat', 'cpet': 'cepat',
    'krn': 'karena', 'karna': 'karena',
    'jd': 'jadi', 'jadii': 'jadi',
    'pas': 'saat',
    'nyampe': 'sampai', 'sampe': 'sampai',
    'dpt': 'dapat', 'dapet': 'dapat',
    'ok': 'oke', 'okei': 'oke', 'tq': 'terima kasih', 'thanks': 'terima kasih',
    'makasih': 'terima kasih', 'nuhun': 'terima kasih',
    'kirim': 'kirim', 'pengiriman': 'kirim',
    'bener': 'benar', 'beneran': 'benar',
    'ori': 'original', 'asli': 'original',
    'kw': 'palsu', 'imitasi': 'palsu',
    'nyesel': 'sesal', 'menyesal': 'sesal',
    'mantap': 'mantap', 'mantul': 'mantap', 'top': 'mantap',
    'kecewa': 'kecewa', 'kecewaa': 'kecewa',
    'pake': 'pakai', 'pakek': 'pakai',
    'bisa': 'bisa', 'bisanya': 'bisa',
    'coba': 'coba',
    'baru': 'baru',
    'mati': 'mati', 'rusak': 'rusak', 'pecah': 'rusak',
    'hancur': 'rusak', 'error': 'rusak'
}

class TextPreprocessor:
    def __init__(self, use_stemmer=True):
        self.use_stemmer = use_stemmer
        if use_stemmer:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = None
        self.stem_cache = {}

    def clean_text(self, text):
        """
        Clean text from URLs, usernames, hashtags, emojis, punctuation, and extra whitespace.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)
        
        # Remove Usernames (@username) and Hashtags (#tag)
        text = re.sub(r'@\S+|#\S+', '', text)
        
        # Remove repeated characters (3 or more times, e.g., 'baguuus' -> 'baguus')
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove non-alphabet characters (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Clean double/multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def normalize_slang(self, text):
        """
        Replace common slang and abbreviations with standard Indonesian words.
        """
        words = text.split()
        normalized_words = [SLANG_DICT.get(word, word) for word in words]
        return " ".join(normalized_words)

    def remove_stopwords(self, text):
        """
        Remove common stopwords but keep critical negation words.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in INDONESIAN_STOPWORDS]
        return " ".join(filtered_words)

    def stem_word(self, word):
        """
        Stem single word with local caching to improve performance.
        """
        if not self.use_stemmer or not self.stemmer:
            return word
        if word not in self.stem_cache:
            self.stem_cache[word] = self.stemmer.stem(word)
        return self.stem_cache[word]

    def stem_text(self, text):
        """
        Stem the words in a text using local cache.
        """
        if not self.use_stemmer:
            return text
        words = text.split()
        stemmed_words = [self.stem_word(w) for w in words]
        return " ".join(stemmed_words)

    def preprocess(self, text):
        """
        Execute full pipeline: Clean -> Normalize Slang -> Remove Stopwords -> Stemming.
        """
        # Step 1: Clean
        cleaned = self.clean_text(text)
        # Step 2: Slang normalization
        normalized = self.normalize_slang(cleaned)
        # Step 3: Stopwords removal
        no_stopwords = self.remove_stopwords(normalized)
        # Step 4: Stemming
        final_text = self.stem_text(no_stopwords)
        return final_text

    def preprocess_parallel(self, texts, n_jobs=4):
        """
        Preprocess a list of texts by first extracting all unique words, 
        stemming them in parallel (or using multiprocessing), and mapping back.
        """
        cleaned_texts = [self.normalize_slang(self.clean_text(t)) for t in texts]
        
        if not self.use_stemmer:
            return [self.remove_stopwords(t) for t in cleaned_texts]

        # Extract unique words
        unique_words = set()
        for text in cleaned_texts:
            unique_words.update(text.split())
        
        unique_words = list(unique_words)
        print(f"Total unique words to stem: {len(unique_words)}")
        
        # Stem unique words (using multiprocessing if possible)
        from concurrent.futures import ProcessPoolExecutor
        import math
        
        # Chunk unique words for parallel processing
        chunk_size = math.ceil(len(unique_words) / n_jobs)
        chunks = [unique_words[i:i + chunk_size] for i in range(0, len(unique_words), chunk_size)]
        
        stemmed_dict = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(_stem_chunk_helper, chunks)
            for res in results:
                stemmed_dict.update(res)
        
        # Update self.stem_cache
        self.stem_cache.update(stemmed_dict)
        
        # Map back to texts
        preprocessed_texts = []
        for text in cleaned_texts:
            words = text.split()
            # Stem and remove stopwords
            processed_words = [stemmed_dict.get(w, w) for w in words if w not in INDONESIAN_STOPWORDS]
            preprocessed_texts.append(" ".join(processed_words))
            
        return preprocessed_texts

# Helper function to save and load preprocessor
def save_preprocessor(preprocessor, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)

def load_preprocessor(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def _stem_chunk_helper(word_chunk):
    factory = StemmerFactory()
    local_stemmer = factory.create_stemmer()
    return {word: local_stemmer.stem(word) for word in word_chunk}
