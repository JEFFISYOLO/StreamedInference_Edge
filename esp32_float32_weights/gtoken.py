#!/usr/bin/env python3
"""
Generate proper token IDs for ESP32 inference
Run this on your PC where the model files are located
"""
import sys
import os

def method1_transformers():
    """Method 1: Use transformers library (recommended)"""
    try:
        from transformers import AutoTokenizer
        
        # Try to load from current directory or specified path
        paths = [
            "esp32_float32_weights",
            ".",
            sys.argv[1] if len(sys.argv) > 1 else None
        ]
        
        tokenizer = None
        for path in paths:
            if path and os.path.exists(path):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path)
                    print(f"✓ Loaded tokenizer from: {path}", file=sys.stderr)
                    break
                except:
                    continue
        
        if not tokenizer:
            raise Exception("Could not load tokenizer")
        
        # Test prompts
        prompts = [
            "Hello world",
            "The quick brown fox",
            "Once upon a time",
        ]
        
        print("// Generated token arrays for ESP32")
        print("// Copy these into your app_main.c\n")
        
        for i, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            
            print(f'// Prompt: "{prompt}"')
            print(f'const uint16_t prompt_{i}_tokens[] = {{')
            
            for j, tok in enumerate(tokens):
                if j % 10 == 0:
                    print('    ', end='')
                print(f'{tok:5d}', end='')
                if j < len(tokens) - 1:
                    print(',', end='')
                if (j + 1) % 10 == 0 or j == len(tokens) - 1:
                    print()
                else:
                    print(' ', end='')
            
            print('};')
            print(f'const int prompt_{i}_len = {len(tokens)};\n')
        
        # Print special tokens
        print(f"// Special tokens:")
        print(f"// BOS: {tokenizer.bos_token_id}")
        print(f"// EOS: {tokenizer.eos_token_id}")
        print(f"// PAD: {tokenizer.pad_token_id}")
        print(f"// UNK: {tokenizer.unk_token_id}")
        print(f"// Vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except ImportError:
        print("transformers library not installed", file=sys.stderr)
        print("Install with: pip install transformers", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error with transformers: {e}", file=sys.stderr)
        return False

def method2_sentencepiece():
    """Method 2: Use sentencepiece directly"""
    try:
        import sentencepiece as spm
        
        # Look for tokenizer.model
        model_path = None
        search_paths = [
            "esp32_float32_weights/tokenizer.model",
            "tokenizer.model",
            sys.argv[1] + "/tokenizer.model" if len(sys.argv) > 1 else None
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise Exception("tokenizer.model not found")
        
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        
        print(f"✓ Loaded SentencePiece model from: {model_path}", file=sys.stderr)
        
        prompts = [
            "Hello world",
            "The quick brown fox",
            "Once upon a time",
        ]
        
        print("// Generated token arrays for ESP32")
        print("// Copy these into your app_main.c\n")
        
        for i, prompt in enumerate(prompts):
            tokens = sp.EncodeAsIds(prompt)
            # Add BOS token (usually 1)
            tokens = [1] + tokens
            
            print(f'// Prompt: "{prompt}"')
            print(f'const uint16_t prompt_{i}_tokens[] = {{')
            
            for j, tok in enumerate(tokens):
                if j % 10 == 0:
                    print('    ', end='')
                print(f'{tok:5d}', end='')
                if j < len(tokens) - 1:
                    print(',', end='')
                if (j + 1) % 10 == 0 or j == len(tokens) - 1:
                    print()
                else:
                    print(' ', end='')
            
            print('};')
            print(f'const int prompt_{i}_len = {len(tokens)};\n')
        
        print(f"// Vocab size: {sp.GetPieceSize()}")
        
        return True
        
    except ImportError:
        print("sentencepiece library not installed", file=sys.stderr)
        print("Install with: pip install sentencepiece", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error with sentencepiece: {e}", file=sys.stderr)
        return False

def main():
    print("=" * 60, file=sys.stderr)
    print("ESP32 Token Generator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    # Try transformers first, then sentencepiece
    if method1_transformers():
        return
    
    print("\nTrying sentencepiece...", file=sys.stderr)
    if method2_sentencepiece():
        return
    
    # If both fail, provide manual instructions
    print("\n" + "=" * 60, file=sys.stderr)
    print("ERROR: Could not load tokenizer!", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("\nPlease install one of:", file=sys.stderr)
    print("  pip install transformers", file=sys.stderr)
    print("  pip install sentencepiece", file=sys.stderr)
    print("\nOr manually tokenize your prompts using:", file=sys.stderr)
    print("  https://huggingface.co/spaces/Xenova/the-tokenizer-playground", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()