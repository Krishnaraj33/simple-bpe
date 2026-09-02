import regex as re
from typing import List
from helpers import get_stats , merge_pairs

class Tokenizer:
    
    def __init__(self,vocab_size:int,pattern:str):
        self.vocab_size = vocab_size
        self.pattern = re.compile(pattern)
        self.merges: dict[tuple[int,int],int] = {}
        self.vocab:dict[int:bytes] = {idx:bytes([idx]) for idx in (range(256))}
    
    
    def encode(self,text:str) -> List[int]:
        text_chunks = re.findall(self.pattern,text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            
            #Filtering out single chars and iterating 
            while len(chunk_ids) >=2:
                #breaking the word for lookup
                stats = get_stats(chunk_ids)
                
                # Find the pair that was merged earliest (lowest token ID = highest priority rule).
                # Using min ensures we apply older/earlier rules before newer ones.
                pair = min(stats,key= lambda p: self.merges.get(p,float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = merge_pairs(chunk_ids,pair,idx)
            ids.extend(chunk_ids)
        return ids
        
    def decode(self,ids:List[int]) -> str:
        text_bytes = b''.join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8")
        
    def train(self,text:str):
        assert self.vocab_size > 256
        num_merges = self.vocab_size - 256
        
        text_chunks = (re.findall(self.pattern,text))
        
        ids = []
        
        # for chunk in text_chunks:
        #     ids.append(list(chunk.encode("utf-8")))
        
        #we need to convert it to list to have each character encoded individually 
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = get_stats(chunk_ids,stats)
            if not stats:
                break
            top_pair = max(stats,key=stats.get)
            idx = 256 + i
            
            ids = [merge_pairs(chunk_ids,top_pair,idx) for chunk_ids in ids]
            
            #Update our tuple to int mapping
            self.merges[top_pair] = idx 
            
            #Adding the newly minted token to our vocab 
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            print(f"Merged pair {top_pair} into token {idx} (Bytes: {self.vocab[idx]})")
            # print(self.vocab)

        # print(f"\nTrained {len(self.merges)} total unique merges successfully.")
        
        # for idx in range(256, len(self.vocab)):
        #     print(f"ID {idx}: {self.vocab[idx]}")
        
if __name__ == "__main__":
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    
    tokenizer = Tokenizer(260, GPT4_SPLIT_PATTERN)
    
    text = "Hello world"
    print("=== 1. TRAINING ===")
    tokenizer.train(text)
    
    print("\n=== 2. ENCODING ===")
    sample = "Hello world"
    encoded_ids = tokenizer.encode(sample)
    print(f"Encoded '{sample}' -> IDs: {encoded_ids}")
    
    print("\n=== 3. DECODING ===")
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded IDs {encoded_ids} -> Text: {repr(decoded_text)}")