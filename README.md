# audmixmod - The Ultimate Audio Analysis & Transformation Tool
*designed and developed by Claude Sonnet 4 [https://claude.ai](https://claude.ai) and Jeremy Carter <jeremy@jeremycarter.ca>*

**audmixmod** is a comprehensive audio processing powerhouse that converts audio files to MusicXML format while providing extensive audio transformations, AI-powered hearing features, and professional-grade analysis capabilities. Think of it as the Swiss Army knife for audio analysis, transcription, and creative sound manipulation.

## 🚀 Features Overview

### Core Audio Processing
- **Universal Audio Support**: WAV, FLAC, OGG, MP3, MP4, M4A formats
- **Native Sample Rate Preservation**: No unnecessary resampling unless requested
- **True Stereo Processing**: Maintains stereo integrity throughout the pipeline
- **Musical Transcription**: High-quality conversion to MusicXML, MIDI, ABC, LilyPond
- **Intelligent Analysis**: Key detection, tempo analysis, chord progression recognition

### Audio Transformations
- **Time & Pitch**: Time stretching, pitch shifting with high-quality algorithms
- **Spectral Processing**: Harmonic/percussive separation, spectral manipulation
- **Creative Effects**: Granular synthesis, stutter effects, phase randomization
- **Frequency Control**: Precise frequency masking and spectral centroid shifting
- **Dynamic Processing**: Compression, normalization, fade in/out

### AI Hearing Features 🧠
- **Timbre Analysis**: Comprehensive spectral feature extraction (MFCCs, spectral centroid, bandwidth, flatness)
- **Rhythm Intelligence**: Advanced rhythm analysis with tempo stability metrics
- **Texture Recognition**: Zero-crossing rate analysis, spectral contrast, tonnetz features
- **Musical Intelligence**: Automated chord detection, key analysis, beat tracking
- **AI-Powered Analysis**: Claude/OpenAI integration for intelligent audio insights
- **Darunia Mode**: 🎵 Epic Goron King celebration mode! 🎵

### Visualizations & Analysis
- **Waveform Visualization**: High-resolution stereo/mono waveform plots
- **Spectrograms**: Standard, Mel-scale, and Constant-Q Transform visualizations
- **Piano Roll**: Beautiful MIDI-style piano roll diagrams
- **Frequency Analysis**: FFT spectrum analysis with CSV export
- **Musical Visualizations**: Chromagrams, tonnetz plots

### Professional Integration
- **DAW Project Creation**: Native project files for Ardour, Bitwig, Reaper, Ableton Live
- **Batch Processing**: Parallel processing of multiple files
- **Watch Folder**: Automatic processing of new files
- **Preset System**: Save and load transformation presets
- **Webhook Integration**: Automated notifications and cloud integration

## 📦 Installation

### Requirements
```bash
pip install librosa soundfile numpy scipy matplotlib
pip install mido  # Optional: for MIDI export
pip install anthropic openai  # Optional: for AI features
```

### Environment Variables (Optional)
For AI-powered features, configure your provider and credentials:

#### Basic Setup
```bash
# Choose your AI provider
export MODEL_PROVIDER="anthropic"  # or: openai, chatgpt, claude.ai, local, custom
export MODEL_NAME="claude-3-sonnet-20240229"  # or: gpt-4, gpt-4-turbo, etc.

# Anthropic/Claude API (recommended)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export ANTHROPIC_API_BASE="https://api.anthropic.com"  # Optional: custom endpoint

# OpenAI API (alternative)
export OPENAI_API_KEY="sk-your-key-here" 
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional: custom endpoint
```

#### Advanced Provider Configuration

**Local LLM (llama.cpp, Ollama, etc.):**
```bash
export MODEL_PROVIDER="local"
export MODEL_NAME="llama-3"
export OPENAI_API_BASE="http://localhost:1234/v1"  # Your local LLM endpoint
```

**Custom GPT (ChatGPT with custom instructions):**
```bash
export MODEL_PROVIDER="openai"
export MODEL_NAME="gpt-4o-gizmo-g-abc123def456"  # Your Custom GPT ID
export OPENAI_API_KEY="sk-your-key-here"
```

**Custom Provider (any OpenAI-compatible API):**
```bash
export MODEL_PROVIDER="custom"
export MODEL_NAME="my-custom-model"
export CUSTOM_AI_API_KEY="your-custom-api-key"
export CUSTOM_AI_API_BASE="https://my-ai-service.com/v1"
# Add any custom parameters:
export CUSTOM_AI_REQUEST_TEMPERATURE="0.7"
export CUSTOM_AI_REQUEST_TOP_P="0.9"
```

**Custom Provider via Config File:**
```bash
export CUSTOM_AI_CONFIG_FILE="/path/to/ai-config.json"
```

```json
{
  "api_key": "your-key",
  "api_base": "https://your-endpoint.com",
  "request_temperature": 0.7,
  "request_max_tokens": 2000
}
```

### Clone and Install
```bash
git clone https://github.com/yourusername/audmixmod.git
cd audmixmod
chmod +x audmixmod.py
```

## 🎵 Quick Start

### Basic Audio Transcription
```bash
# Simple transcription to MusicXML
./audmixmod.py song.wav

# Full analysis with all AI features
./audmixmod.py song.wav --output-all --analysis-report
```

### Creative Audio Transformation
```bash
# Chipmunk effect (higher pitch, faster)
./audmixmod.py song.wav --pitch-shift 7 --time-stretch 0.8

# Create ethereal ambience
./audmixmod.py song.wav --harmonic-only --reverb 0.3 --time-stretch 2.0
```

### Batch Processing
```bash
# Process entire music library
./audmixmod.py --batch-dir /music/library --output-dir /processed --parallel
```

## 📖 Complete Usage Guide

### Basic Syntax
```bash
audmixmod.py [INPUT_FILE] [OPTIONS]
```

## 🎛️ Command Line Options

### Basic Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `input_file` | Input audio file path | `song.wav` |
| `-o, --output-dir` | Output directory for all files | `-o /output` |
| `--filename-prefix` | Custom prefix for output files | `--filename-prefix processed_` |
| `-sr, --sample-rate` | Force specific sample rate | `-sr 44100` |

### Batch Processing & Automation
| Argument | Description | Example |
|----------|-------------|---------|
| `--batch-dir` | Process all files in directory | `--batch-dir /music` |
| `--file-pattern` | File pattern for batch processing | `--file-pattern "*.wav"` |
| `--watch-folder` | Auto-process new files in folder | `--watch-folder /dropbox/music` |
| `--parallel` | Use parallel processing for batch | `--parallel` |

### Output Formats
| Argument | Description | Default |
|----------|-------------|---------|
| `--output-midi` | Generate MIDI file | No |
| `--output-abc` | Generate ABC notation | No |
| `--output-lilypond` | Generate LilyPond notation | No |
| `--output-csv` | Generate CSV data export | No |
| `--output-all` | Generate ALL formats + AI features | No |

### DAW Integration
| Argument | Description | Supported DAWs |
|----------|-------------|----------------|
| `--send-to-daw` | Create DAW project file | `ardour`, `bitwig`, `reaper`, `ableton` |
| `--daw-project-name` | Custom project name | Default: filename |
| `--daw-session-path` | Custom session path | Default: output dir |

**Example:**
```bash
# Create Ableton Live project
./audmixmod.py track.wav --send-to-daw ableton --daw-project-name "Remix Session"
```

### Analysis Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--key-detection` | Enable musical key detection | No |
| `--chord-analysis` | Analyze chord progressions | No |
| `--confidence-threshold` | Note detection confidence (0.0-1.0) | 0.1 |
| `--pitch-detection-method` | Algorithm: `piptrack` or `crepe` | `piptrack` |
| `--tempo-detection-method` | Method: `standard` or `advanced` | `standard` |
| `--pitch-correction` | Snap pitches to semitones | No |

## 🎨 Audio Transformations

### Time & Pitch Manipulation
| Argument | Description | Range | Example |
|----------|-------------|-------|---------|
| `--time-stretch` | Time stretch factor | 0.1-10.0 | `--time-stretch 0.5` (2x faster) |
| `--pitch-shift` | Pitch shift in semitones | -48 to +48 | `--pitch-shift 12` (+1 octave) |

**Creative Examples:**
```bash
# Slow ambient stretch
./audmixmod.py song.wav --time-stretch 4.0 --pitch-shift -12

# Chipmunk vocals
./audmixmod.py vocals.wav --pitch-shift 8 --time-stretch 0.7

# Deep, slow monster voice
./audmixmod.py voice.wav --pitch-shift -24 --time-stretch 2.5
```

### Harmonic & Percussive Processing
| Argument | Description | Range |
|----------|-------------|-------|
| `--harmonic-only` | Extract only harmonic content | Boolean |
| `--percussive-only` | Extract only percussive content | Boolean |
| `--harmonic-percussive-ratio` | Blend ratio (0=percussive, 1=harmonic) | 0.0-1.0 |

**Examples:**
```bash
# Extract melody line only
./audmixmod.py song.wav --harmonic-only

# Extract drums/percussion only
./audmixmod.py song.wav --percussive-only

# 70% harmonic, 30% percussive blend
./audmixmod.py song.wav --harmonic-percussive-ratio 0.7
```

### Spectral Manipulations
| Argument | Description | Range |
|----------|-------------|-------|
| `--spectral-centroid-shift` | Shift brightness/darkness | -2.0 to +2.0 |
| `--spectral-bandwidth-stretch` | Stretch spectral bandwidth | 0.1-5.0 |
| `--frequency-mask-low` | Low frequency cutoff (Hz) | 20-20000 |
| `--frequency-mask-high` | High frequency cutoff (Hz) | 20-20000 |

**Examples:**
```bash
# Make brighter/airier
./audmixmod.py song.wav --spectral-centroid-shift 0.5

# Make darker/warmer
./audmixmod.py song.wav --spectral-centroid-shift -0.5

# Keep only mid frequencies (phone effect)
./audmixmod.py voice.wav --frequency-mask-low 300 --frequency-mask-high 3000
```

### Creative Effects
| Argument | Description | Range |
|----------|-------------|-------|
| `--stutter` | Stutter effect duration (seconds) | 0.01-1.0 |
| `--granular-synthesis` | Apply granular synthesis | Boolean |
| `--phase-randomize` | Randomize phase (0-1, creates weird textures) | 0.0-1.0 |

**Examples:**
```bash
# Glitchy stutter effect
./audmixmod.py beat.wav --stutter 0.1

# Granular texture transformation
./audmixmod.py pad.wav --granular-synthesis

# Weird phase-scrambled texture
./audmixmod.py drone.wav --phase-randomize 0.3
```

### Preprocessing Options
| Argument | Description | Range |
|----------|-------------|-------|
| `--denoise` | Apply noise reduction | Boolean |
| `--normalize` | Normalize audio levels | Boolean |
| `--trim-silence` | Remove silence from ends | Boolean |
| `--fade-in` | Fade in duration (seconds) | 0.1-10.0 |
| `--fade-out` | Fade out duration (seconds) | 0.1-10.0 |
| `--reverse` | Reverse the audio | Boolean |

## 🤖 AI-Powered Features

### Universal AI Provider Support
Supports multiple AI providers with seamless switching:
- **Anthropic Claude** - Official API with advanced reasoning
- **OpenAI GPT** - Including GPT-4, GPT-4 Turbo, and Custom GPTs
- **Local LLMs** - llama.cpp, Ollama, and OpenAI-compatible servers
- **Custom Providers** - Any OpenAI-compatible API endpoint

### AI Analysis
Get intelligent insights about your audio:

```bash
# Basic AI analysis (uses configured provider)
./audmixmod.py song.wav --ai-enabled --ai-analyze

# Custom AI prompt
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --ai-prompt "Describe this audio as if you're a jazz musician"

# Override provider and model for this session
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --model-provider openai --model-name gpt-4-turbo

# Use a Custom GPT trained for audio analysis
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --model-name "gpt-4o-gizmo-g-MusicAnalyzer123"
```

### Custom GPT Support 🤖
First-class support for ChatGPT Custom GPTs with conversation state:

```bash
# Use your specialized audio analysis Custom GPT
export MODEL_NAME="gpt-4o-gizmo-g-YourAudioGPT"
./audmixmod.py song.wav --ai-enabled --ai-analyze

# Or specify directly via command line
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --model-name "gpt-4o-gizmo-g-AudioMaster456"
```

**Custom GPT Features:**
- Automatic detection of Custom GPT naming pattern
- Conversation state management with `conversation_id`
- Special API handling for Custom GPT initialization
- Support for all Custom GPT capabilities

### Local LLM Integration 💻
Run audio analysis completely offline with local models:

```bash
# Using llama.cpp server
export MODEL_PROVIDER="local"
export MODEL_NAME="llama-3-8b-instruct"
export OPENAI_API_BASE="http://localhost:8080/v1"
./audmixmod.py song.wav --ai-enabled --ai-analyze

# Using Ollama
export OPENAI_API_BASE="http://localhost:11434/v1"
./audmixmod.py song.wav --ai-enabled --model-provider local \
  --model-name "llama3:latest" --ai-analyze

# One-shot command line usage
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --model-provider local --model-name "phi-3" \
  --custom-ai-api-base "http://localhost:1234/v1"
```

### Custom AI Providers 🔧
Integrate any OpenAI-compatible API:

```bash
# Anthropic via proxy
./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --model-provider custom \
  --custom-ai-api-base "https://my-proxy.com/anthropic" \
  --custom-ai-api-key "proxy-key-123"

# Commercial AI service
export MODEL_PROVIDER="custom"
export CUSTOM_AI_API_KEY="your-service-key"
export CUSTOM_AI_API_BASE="https://api.customai.com/v1"
export CUSTOM_AI_REQUEST_TEMPERATURE="0.3"  # Custom parameters
./audmixmod.py song.wav --ai-enabled --ai-analyze

# Using config file for complex setups
echo '{
  "api_key": "sk-custom-123",
  "api_base": "https://my-ai.com/v1",
  "request_temperature": 0.7,
  "request_top_p": 0.9,
  "request_max_tokens": 2000
}' > ai-config.json

./audmixmod.py song.wav --ai-enabled --ai-analyze \
  --custom-ai-config-file ai-config.json
```

### Darunia Mode 🎵
The ultimate audio celebration experience! The Goron King himself will dance to your audio:

```bash
# Basic Darunia Mode
./audmixmod.py song.wav --darunia-mode

# Enhanced bliss with transformations
./audmixmod.py song.wav --harmonic-only --granular-synthesis --darunia-mode

# Maximum bliss experience
./audmixmod.py song.wav --time-stretch 2.0 --pitch-shift 7 \
  --harmonic-only --granular-synthesis --darunia-mode
```

**What happens in Darunia Mode:**
- Animated ANSI art of the Goron King dancing
- Bliss intensity based on your transformations
- Contextual tips for your next audio adventure
- Sparkly exit sequence that'll make you question reality
- Pure terminal magic, Brother!

### AI Environment Setup
```bash
# Provider Selection
export MODEL_PROVIDER="anthropic"     # anthropic, openai, chatgpt, claude.ai, local, custom
export MODEL_NAME="claude-3-sonnet-20240229"  # Specific model name

# Official APIs
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Custom endpoints (proxy servers, local LLMs)
export ANTHROPIC_API_BASE="https://my-proxy.com/anthropic"
export OPENAI_API_BASE="https://my-local-llm.com/v1"

# Custom provider configuration
export CUSTOM_AI_API_KEY="your-custom-key"
export CUSTOM_AI_API_BASE="https://my-ai-service.com/v1"
export CUSTOM_AI_CONFIG_FILE="/path/to/config.json"

# Custom GPT examples
export MODEL_NAME="gpt-4o-gizmo-g-AudioAnalyzer123"  # Your Custom GPT
export MODEL_NAME="gpt-4-turbo"                     # Standard GPT-4

# Local LLM examples
export MODEL_PROVIDER="local"
export MODEL_NAME="llama-3-8b-instruct"
export OPENAI_API_BASE="http://localhost:8080/v1"  # llama.cpp server
```

## 🧠 AI Hearing Features

### Comprehensive Timbre Analysis
Extract detailed acoustic characteristics that describe the "color" and texture of sound:

```bash
# Extract full timbre profile
./audmixmod.py song.wav --timbre-features-json
```

**Generated Features:**
- **Spectral Centroid**: Brightness/darkness of sound
- **Spectral Bandwidth**: Width of frequency distribution
- **Spectral Rolloff**: High-frequency rolloff point
- **Spectral Flatness**: Noisiness vs. tonality measure
- **Zero Crossing Rate**: Roughness/smoothness indicator
- **MFCCs**: 13 Mel-frequency cepstral coefficients
- **Chroma Features**: Pitch class profiles
- **Tonnetz**: Harmonic network representation
- **Spectral Contrast**: Spectral valley/peak distinction

### Advanced Rhythm Analysis
Deep analysis of temporal characteristics:

```bash
# Detailed rhythm intelligence
./audmixmod.py song.wav --detailed-rhythm-analysis
```

**Analysis Output:**
- **Tempo Stability**: How consistent the tempo is
- **Beat Consistency**: Regularity of beat intervals
- **Onset Density**: Notes per second measurement
- **Rhythmic Complexity**: Irregularity measure
- **Downbeat Detection**: Strong beat identification

### Audio Thumbnails
Create preview clips from the most representative part:

```bash
# 5-second thumbnail from audio center
./audmixmod.py song.wav --audio-thumbnail --thumbnail-duration 5.0
```

## 📊 Visualizations

### Waveform Analysis
```bash
# High-resolution waveform plots
./audmixmod.py song.wav --waveform-png
```
- Stereo: Separate left/right channel plots
- Mono: Single waveform visualization
- High DPI (300 DPI) output for print quality

### Spectral Visualizations
```bash
# Standard spectrogram
./audmixmod.py song.wav --generate-spectrogram

# Mel-scale spectrogram (human auditory perception)
./audmixmod.py song.wav --generate-mel-spectrogram --save-mel-data

# Constant-Q Transform (musical pitch analysis)
./audmixmod.py song.wav --generate-cqt --save-cqt-data

# Chromagram (pitch class analysis)
./audmixmod.py song.wav --chromagram-image
```

### Musical Visualizations
```bash
# Piano roll diagram
./audmixmod.py song.wav --piano-roll-png

# FFT spectrum analysis with CSV export
./audmixmod.py song.wav --fft-spectrum-csv
```

### All Visualizations at Once
```bash
# Generate every possible visualization
./audmixmod.py song.wav --output-all
```

## 🎼 Musical Analysis

### Key Detection
```bash
# Detect musical key and mode
./audmixmod.py song.wav --key-detection --verbose
```
Output: Key (C, D, E, etc.) and mode (major/minor)

### Chord Analysis
```bash
# Analyze chord progressions
./audmixmod.py song.wav --chord-analysis
```
Detects common triads: C, Dm, Em, F, G, Am, etc.

### Tempo Detection
```bash
# Standard tempo detection
./audmixmod.py song.wav --tempo-detection-method standard

# Advanced multi-scale tempo analysis
./audmixmod.py song.wav --tempo-detection-method advanced
```

## 🎚️ Preset System

### Save Transformation Presets
```bash
# Create a preset for vintage vinyl effect
./audmixmod.py song.wav --harmonic-only --spectral-centroid-shift -0.3 \
  --denoise --normalize --save-preset "vintage_vinyl" --preset-file presets.json
```

### Load and Apply Presets
```bash
# Apply saved preset
./audmixmod.py newsong.wav --preset-name "vintage_vinyl" --preset-file presets.json
```

### Random Transformation Experimentation
```bash
# Apply random effects for creative discovery
./audmixmod.py song.wav --random-transform
```

## 🔄 Batch Processing & Automation

### Process Multiple Files
```bash
# Process all WAV files in directory
./audmixmod.py --batch-dir /music --output-dir /processed --file-pattern "*.wav"

# Parallel processing for speed
./audmixmod.py --batch-dir /music --output-dir /processed --parallel
```

### Watch Folder Automation
```bash
# Auto-process new files (great for dropboxes)
./audmixmod.py --watch-folder /dropbox/incoming --output-dir /processed
```

### Apply Transformations to Batch
```bash
# Apply same transformation to all files
./audmixmod.py --batch-dir /vocals --output-dir /processed \
  --pitch-shift 5 --harmonic-only --parallel
```

## 🎹 DAW Integration

### Supported DAWs
- **Ardour**: Creates `.ardour` session files
- **Bitwig Studio**: Creates `.bwproject` files  
- **Reaper**: Creates `.rpp` project files
- **Ableton Live**: Creates `.als` project files (gzipped XML)

### DAW Project Creation Examples
```bash
# Ableton Live project
./audmixmod.py track.wav --send-to-daw ableton

# Reaper project with processing
./audmixmod.py song.wav --harmonic-only --send-to-daw reaper

# Ardour session for mixing
./audmixmod.py multitrack.wav --send-to-daw ardour --daw-project-name "Mix Session"
```

## 📈 Performance & Benchmarking

### Performance Analysis
```bash
# Show detailed timing information
./audmixmod.py song.wav --benchmark --verbose
```

**Benchmark Output:**
- Total processing time
- Load time
- Audio duration
- Real-time factor (how much faster than real-time)

### Memory Optimization
- Native sample rate preservation (no unnecessary resampling)
- Efficient stereo processing
- Smart mono conversion only when needed
- Minimal memory footprint for large files

## 🔧 Advanced Features

### Webhook Integration
```bash
# Send notifications when processing completes
./audmixmod.py song.wav --webhook-notify "https://your-webhook-url.com/notify"
```

### Dry Run Mode
```bash
# See what would be processed without actually doing it
./audmixmod.py --batch-dir /music --dry-run
```

### Configuration Files
```bash
# Load settings from JSON config
./audmixmod.py song.wav --config-file settings.json
```

**Example config.json:**
```json
{
  "verbose": true,
  "output_all": true,
  "harmonic_only": true,
  "spectral_centroid_shift": -0.2,
  "analysis_report": true
}
```

## 📝 Analysis Reports

### Comprehensive Text Reports
```bash
# Generate detailed analysis report
./audmixmod.py song.wav --analysis-report --timbre-features-json --detailed-rhythm-analysis
```

**Report Includes:**
- Basic file information (duration, sample rate, channels)
- Musical analysis (tempo, key, note distribution)
- Timbre characteristics (brightness, texture, spectral features)
- Rhythm analysis (tempo stability, beat consistency)
- Processing recommendations
- Chord progression analysis

## 💡 Creative Usage Examples

### Vintage Tape Effect
```bash
./audmixmod.py modern_song.wav --harmonic-only --spectral-centroid-shift -0.4 \
  --denoise --fade-in 0.1 --fade-out 0.2
```

### Ethereal Ambient Transformation
```bash
./audmixmod.py piano.wav --time-stretch 3.0 --harmonic-only \
  --spectral-centroid-shift 0.2 --granular-synthesis
```

### Lo-Fi Hip Hop Processing
```bash
./audmixmod.py jazz_sample.wav --harmonic-percussive-ratio 0.8 \
  --spectral-centroid-shift -0.3 --dynamic-range-compress 0.6
```

### Vocal Processing Chain
```bash
./audmixmod.py vocals.wav --denoise --normalize --frequency-mask-low 80 \
  --frequency-mask-high 8000 --pitch-correction
```

### Experimental Glitch Processing
```bash
./audmixmod.py song.wav --stutter 0.05 --phase-randomize 0.2 \
  --granular-synthesis --random-transform
```

## 🎯 Output Files Reference

For input file `song.wav`, audmixmod can generate:

### Always Generated
- `song.musicxml` - MusicXML transcription
- `song.json` - Complete analysis data

### Optional Outputs (with flags)
- `song.mid` - MIDI file (`--output-midi`)
- `song.abc` - ABC notation (`--output-abc`)
- `song.ly` - LilyPond notation (`--output-lilypond`)
- `song.csv` - Analysis data CSV (`--output-csv`)
- `song_waveform.png` - Waveform plot (`--waveform-png`)
- `song_spectrogram.png` - Spectrogram (`--generate-spectrogram`)
- `song_piano_roll.png` - Piano roll (`--piano-roll-png`)
- `song_cqt.png` - Constant-Q Transform (`--generate-cqt`)
- `song_mel_spectrogram.png` - Mel spectrogram (`--generate-mel-spectrogram`)
- `song_chromagram.png` - Chromagram (`--chromagram-image`)
- `song_fft_spectrum.csv` - FFT data (`--fft-spectrum-csv`)
- `song_timbre_features.json` - Timbre analysis (`--timbre-features-json`)
- `song_rhythm_analysis.json` - Rhythm analysis (`--detailed-rhythm-analysis`)
- `song_report.txt` - Text analysis report (`--analysis-report`)
- `song_thumbnail.wav` - Audio preview (`--audio-thumbnail`)

### Processed Audio Files
Generated when transformations are applied:
- `song.44100.transformed.wav` - Transformed audio output

### DAW Project Files
- `song_ardour/` - Ardour session directory
- `song.bwproject` - Bitwig Studio project
- `song.rpp` - Reaper project
- `song.als` - Ableton Live project

## 🔍 Troubleshooting

### Common Issues

**"Error loading audio file"**
- Check file format (supported: WAV, FLAC, OGG, MP3, MP4, M4A)
- Verify file isn't corrupted
- Ensure file permissions allow reading

**"MIDI export requires 'mido' package"**
```bash
pip install mido
```

**"Memory issues with large files"**
- Use native sample rate (don't specify `--sample-rate`)
- Process in batches rather than all at once
- Consider using `--normalize` to reduce dynamic range

**"Slow processing"**
- Use `--parallel` for batch processing
- Avoid unnecessary transformations
- Use `--dry-run` to test settings first

### Performance Tips

1. **Use native sample rates** - Don't resample unless necessary
2. **Batch process similar files** together
3. **Use presets** for repeated transformations
4. **Enable parallel processing** for multiple files
5. **Disable unused features** to speed up processing

## 🤝 Contributing

audmixmod is open source and welcomes contributions! Areas where help is appreciated:

- Additional audio transformations
- New visualization types
- More DAW integrations
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with amazing open-source libraries:
- **librosa** - Audio analysis powerhouse
- **soundfile** - Audio I/O
- **numpy/scipy** - Numerical computing
- **matplotlib** - Visualizations
- **mido** - MIDI processing

---

**audmixmod** - Where audio meets intelligence. Transform, analyze, and create like never before! 🎵✨
