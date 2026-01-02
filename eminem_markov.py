"""
Eminem-Style Rap Lyrics Generator
Bigram Markov Chain with Rhyme Awareness

Inspired by HW 6 Assignment Autumn 2025 CMSC 14100 @ UChicago by Staff
Refined the bigram Markov chain and used unique data for the lyrics corpus
Added rhyme awareness to the generator with the help of the pronouncing library
Consulted Claude Code; Built with Cursor; Used Groq API

Example Usage:
    python eminem_markov.py eminem_lyrics.txt --bars 8 --rhyme-scheme AABB
    python eminem_markov.py eminem_lyrics.txt --bars 8 --refine --show-raw
"""

from __future__ import annotations

import random
import re
import argparse
import os
from collections import defaultdict
from typing import Optional

try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

START_TOKEN = "<START>"
END_TOKEN = "<END>"


class RhymeEngine:
    """Handles rhyme detection and finding rhyming words."""
    
    def __init__(self, vocabulary: set[str]):
        self.vocabulary = vocabulary
        self.rhyme_cache: dict[str, list[str]] = {}
        self._build_rhyme_index()
    
    def _get_rhyme_sound(self, word: str) -> Optional[str]:
        """Extract the rhyme sound (last stressed vowel + everything after) from a word."""
        word = word.lower().strip()
        
        if HAS_PRONOUNCING:
            phones_list = pronouncing.phones_for_word(word)
            if phones_list:
                phones = phones_list[0]
                # Get rhyming part
                rhyme_part = pronouncing.rhyming_part(phones)
                return rhyme_part if rhyme_part else None
        
        if len(word) >= 3:
            return word[-3:]
        elif len(word) >= 2:
            return word[-2:]
        return word
    
    def _build_rhyme_index(self):
        """Pre-compute rhyme groups for the vocabulary."""
        rhyme_groups: dict[str, list[str]] = defaultdict(list)
        
        for word in self.vocabulary:
            sound = self._get_rhyme_sound(word)
            if sound:
                rhyme_groups[sound].append(word)
        
        # Cache: for each word, store words that rhyme with it
        for sound, words in rhyme_groups.items():
            for word in words:
                self.rhyme_cache[word.lower()] = [w for w in words if w.lower() != word.lower()]
    
    def get_rhymes(self, word: str) -> list[str]:
        """Get words from vocabulary that rhyme with the given word."""
        return self.rhyme_cache.get(word.lower(), [])
    
    def words_rhyme(self, word1: str, word2: str) -> bool:
        """Check if two words rhyme."""
        sound1 = self._get_rhyme_sound(word1)
        sound2 = self._get_rhyme_sound(word2)
        
        if sound1 and sound2:
            return sound1 == sound2
        return False


class BigramMarkovChain:
    """Bigram Markov Chain for text generation."""
    
    def __init__(self):
        # transitions[(word1, word2)] = {next_word: count}
        self.transitions: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocabulary: set[str] = set()
        self.line_starters: list[tuple[str, str]] = []  # Bigrams that start lines
    
    def _preprocess_line(self, line: str) -> list[str]:
        """Clean and tokenize a line."""
        # Keep apostrophes in contractions, remove other punctuation
        line = line.lower().strip()
        line = re.sub(r"[^\w\s']", "", line)
        tokens = line.split()
        return tokens
    
    def train(self, corpus_path: str):
        """Train the model on a lyrics file."""
        print(f"Training on: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_count = 0
        for line in lines:
            tokens = self._preprocess_line(line)
            
            if len(tokens) < 2:
                continue
            
            line_count += 1
            
            # Add start and end tokens
            tokens = [START_TOKEN, START_TOKEN] + tokens + [END_TOKEN]
            
            # Record line starters (first real bigram)
            if len(tokens) >= 4:
                self.line_starters.append((tokens[2], tokens[3]))
            
            # Build vocabulary (exclude special tokens)
            for token in tokens:
                if token not in (START_TOKEN, END_TOKEN):
                    self.vocabulary.add(token)
            
            # Record bigram transitions
            for i in range(len(tokens) - 2):
                bigram = (tokens[i], tokens[i + 1])
                next_word = tokens[i + 2]
                self.transitions[bigram][next_word] += 1
        
        print(f"Trained on {line_count} lines")
        print(f"Vocabulary size: {len(self.vocabulary)} words")
        print(f"Unique bigrams: {len(self.transitions)}")
    
    def _sample_next(self, bigram: tuple[str, str], bias_words: Optional[set[str]] = None, 
                     bias_weight: float = 3.0) -> str:
        """Sample the next word given a bigram, optionally biasing toward certain words."""
        if bigram not in self.transitions:
            return END_TOKEN
        
        choices = self.transitions[bigram]
        words = list(choices.keys())
        weights = list(choices.values())
        
        # Apply bias for rhyming words
        if bias_words:
            weights = [
                w * bias_weight if words[i] in bias_words else w
                for i, w in enumerate(weights)
            ]
        
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return random.choices(words, weights=weights, k=1)[0]
    
    def generate_line(self, rhyme_engine: Optional[RhymeEngine] = None,
                      target_rhyme: Optional[str] = None,
                      max_words: int = 15,
                      min_words: int = 4) -> list[str]:
        """Generate a single line of lyrics."""
        # Start with a random line-starting bigram
        if not self.line_starters:
            return ["no", "training", "data"]
        
        current_bigram = random.choice(self.line_starters)
        line = list(current_bigram)
        
        # Get rhyming words to bias toward
        rhyme_bias: set[str] = set()
        if rhyme_engine and target_rhyme:
            rhyme_bias = set(rhyme_engine.get_rhymes(target_rhyme))
        
        for _ in range(max_words):
            # Increase rhyme bias as we approach line end
            bias_weight = 3.0 if len(line) < max_words - 3 else 8.0
            
            next_word = self._sample_next(
                current_bigram,
                bias_words=rhyme_bias if len(line) >= min_words else None,
                bias_weight=bias_weight
            )
            
            if next_word == END_TOKEN:
                if len(line) >= min_words:
                    break
                else:
                    # Line too short, try to continue
                    next_word = self._sample_next(current_bigram)
                    if next_word == END_TOKEN:
                        break
            
            line.append(next_word)
            current_bigram = (current_bigram[1], next_word)
        
        return line
    
    def get_line_ending(self, line: list[str]) -> Optional[str]:
        """Get the last real word from a line."""
        for word in reversed(line):
            if word not in (START_TOKEN, END_TOKEN) and word.isalpha():
                return word
        return None


class LLMRefiner:
    """Refines raw Markov output using Groq's LLM API."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not HAS_GROQ:
            raise ImportError("Groq library not found. Install with: pip install groq")
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. "
                "Get a free key at: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"
    
    def refine(self, raw_lyrics: str, rhyme_scheme: str = "AABB") -> str:
        """Refine raw Markov lyrics for coherence while preserving style."""
        
        prompt = f"""You are a rap lyrics editor. Your job is to take raw, rough draft rap lyrics and refine them for coherence and flow while preserving the original style, vocabulary, and attitude.

RULES:
1. Keep the same number of bars (lines)
2. Preserve the rhyme scheme: {rhyme_scheme}
3. Keep as much original vocabulary as possible - these words define the style
4. Fix grammar only where necessary for flow
5. Make each line make sense on its own AND connect to the verse theme
6. Keep the aggressive/introspective Eminem-style tone
7. Maintain similar line lengths (8-12 words per bar)
8. Do NOT censor or sanitize any language
9. Do NOT add explanations - output ONLY the refined lyrics

RAW LYRICS:
{raw_lyrics}

REFINED LYRICS:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert rap lyrics editor. Output only refined lyrics, nothing else."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            refined = response.choices[0].message.content.strip()
            return refined
            
        except Exception as e:
            print(f"LLM refinement failed: {e}")
            print("Returning raw lyrics instead.")
            return raw_lyrics


class RapLyricsGenerator:
    """Main generator combining Markov chain with rhyme awareness."""
    
    RHYME_SCHEMES = {
        "AABB": [0, 0, 1, 1],  # Couplets
        "ABAB": [0, 1, 0, 1],  # Alternate
        "ABBA": [0, 1, 1, 0],  # Enclosed
        "AAAA": [0, 0, 0, 0],  # Mono-rhyme
        "FREE": None,          # No enforced scheme
    }
    
    def __init__(self, corpus_path: str, groq_api_key: Optional[str] = None):
        self.markov = BigramMarkovChain()
        self.markov.train(corpus_path)
        self.rhyme_engine = RhymeEngine(self.markov.vocabulary)
        
        self.refiner: Optional[LLMRefiner] = None
        if HAS_GROQ:
            try:
                self.refiner = LLMRefiner(api_key=groq_api_key)
            except (ImportError, ValueError) as e:
                print(f"LLM refinement unavailable: {e}\n")
    
    def generate_verse(self, num_bars: int = 4, rhyme_scheme: str = "AABB") -> list[str]:
        """Generate a verse with the specified rhyme scheme."""
        scheme_pattern = self.RHYME_SCHEMES.get(rhyme_scheme.upper())
        
        lines: list[list[str]] = []
        rhyme_targets: dict[int, str] = {}  # rhyme_group -> target word
        
        for i in range(num_bars):
            target_rhyme = None
            
            if scheme_pattern:
                # Extend pattern if needed
                pattern_idx = i % len(scheme_pattern)
                rhyme_group = scheme_pattern[pattern_idx]
                
                # If we've seen this rhyme group before, try to rhyme with it
                if rhyme_group in rhyme_targets:
                    target_rhyme = rhyme_targets[rhyme_group]
            
            # Generate the line
            line = self.markov.generate_line(
                rhyme_engine=self.rhyme_engine,
                target_rhyme=target_rhyme
            )
            
            lines.append(line)
            
            # Record this line's ending for future rhyming
            if scheme_pattern:
                pattern_idx = i % len(scheme_pattern)
                rhyme_group = scheme_pattern[pattern_idx]
                ending = self.markov.get_line_ending(line)
                if ending and rhyme_group not in rhyme_targets:
                    rhyme_targets[rhyme_group] = ending
        
        # Format lines as strings
        return [" ".join(line) for line in lines]
    
    def generate(self, num_bars: int = 8, rhyme_scheme: str = "AABB", 
                 refine: bool = False, show_raw: bool = False) -> str:
        """Generate lyrics with optional LLM refinement."""
        verse = self.generate_verse(num_bars, rhyme_scheme)
        raw_lyrics = "\n".join(verse)
        
        if refine:
            if not self.refiner:
                print("Warning: LLM refinement requested but unavailable. Returning raw lyrics.\n")
                return raw_lyrics
            
            if show_raw:
                print("=== RAW MARKOV OUTPUT ===")
                print(raw_lyrics)
                print("\n=== LLM REFINED OUTPUT ===")
            
            return self.refiner.refine(raw_lyrics, rhyme_scheme)
        
        return raw_lyrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate rap lyrics using a Markov chain with rhyme awareness"
    )
    parser.add_argument("lyrics_file", help="Path to the lyrics corpus file")
    parser.add_argument("--bars", type=int, default=8, help="Number of bars to generate (default: 8)")
    parser.add_argument(
        "--rhyme-scheme", 
        type=str, 
        default="AABB",
        choices=["AABB", "ABAB", "ABBA", "AAAA", "FREE"],
        help="Rhyme scheme to use (default: AABB)"
    )
    parser.add_argument("--verses", type=int, default=1, help="Number of verses to generate (default: 1)")
    parser.add_argument("--refine", action="store_true", help="Use LLM to refine output for coherence")
    parser.add_argument("--show-raw", action="store_true", help="Show raw Markov output alongside refined (requires --refine)")
    parser.add_argument("--api-key", type=str, help="Groq API key (or set GROQ_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize generator
    print("-" * 50)
    generator = RapLyricsGenerator(args.lyrics_file, groq_api_key=args.api_key)
    print("-" * 50)
    print()
    
    # Generate verses
    for v in range(args.verses):
        if args.verses > 1:
            print(f"[Verse {v + 1}]")
        
        lyrics = generator.generate(
            num_bars=args.bars, 
            rhyme_scheme=args.rhyme_scheme,
            refine=args.refine,
            show_raw=args.show_raw
        )
        print(lyrics)
        
        if v < args.verses - 1:
            print()


if __name__ == "__main__":
    main()
