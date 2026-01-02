Eminem Lyrics Generator 🎤

A bigram Markov chain that generates rap lyrics in Eminem's style, with rhyme awareness and optional LLM refinement.

How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Lyrics Corpus  │ ──▶ │  Markov Chain   │ ──▶ │  LLM Refiner    │
│  (Eminem songs) │     │  (bigram model) │     │  (Groq/Llama)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. Markov Chain learns word patterns and transitions from real lyrics
2. Rhyme Engine biases generation toward rhyming line endings
3. LLM Refiner (optional) polishes output for coherence while preserving style

 Installation

```bash
git clone https://github.com/yourusername/eminem-lyrics-generator.git
cd eminem-lyrics-generator
pip install -r requirements.txt
```

 Usage

 Basic Generation (Markov only)
```bash
python eminem_markov.py lyrics.txt --bars 8 --rhyme-scheme AABB
```

 With LLM Refinement
```bash
export GROQ_API_KEY="your-key-here"  # Get free key at console.groq.com
python eminem_markov.py lyrics.txt --bars 8 --refine
```

 Compare Raw vs Refined
```bash
python eminem_markov.py lyrics.txt --bars 8 --refine --show-raw
```

 Options

| Flag | Description | Default |
|------|-------------|---------|
| `--bars N` | Number of bars to generate | 8 |
| `--rhyme-scheme` | AABB, ABAB, ABBA, AAAA, or FREE | AABB |
| `--verses N` | Number of verses | 1 |
| `--refine` | Enable LLM refinement | off |
| `--show-raw` | Show raw output alongside refined | off |
| `--api-key` | Groq API key (or use env var) | - |

```

 Project Structure

```
eminem-lyrics-generator/
├── eminem_markov.py    # Main script
├── README.md
└── sample_output.txt   # Example generations
```


 Credits

- Built with the [pronouncing](https://pronouncing.readthedocs.io/) library for phonetic rhyme detection
- LLM refinement powered by [Groq](https://groq.com/) (Llama 3.1)
- Inspired by UChicago CMSC 14100 coursework

 License

MIT License - see [LICENSE](LICENSE) for details.
