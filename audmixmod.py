#!/usr/bin/env python3
"""
audmixmod - The Ultimate Audio Analysis & Transformation Tool
Converts audio files to MusicXML format for LLM processing with extensive transformations
Supports: .wav, .flac, .ogg, .mp3, .mp4, .m4a
Features: Multi-format output, DAW integration, batch processing, advanced analysis, AI hearing features
"""

import os
import sys
import argparse
import librosa
import numpy as np
from scipy.signal import find_peaks
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import soundfile as sf
from typing import List, Tuple, Dict, Optional, Union
import json
import glob
import time
import random
import threading
import queue
import gzip
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pickle
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime
import requests
import warnings
import re
import os
from typing import Optional, Dict, Any
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import openai
except ImportError:
    openai = None
warnings.filterwarnings('ignore')

class AudMixMod:
    def __init__(self, sample_rate: Optional[int] = None, config: Optional[Dict] = None):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.frame_length = 2048
        self.config = config or {}
        self.processing_stats = {}
        self.original_channels = 1
        self.original_format = None
        
        # Initialize AI processor
        self.ai_processor = AIProcessor(config) if config.get('ai_enabled') else None
        
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        # Remove/replace invalid characters for Windows/Unix filesystems
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        # Trim whitespace and dots from ends
        sanitized = sanitized.strip(' .')
        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = 'audio_file'
        if len(sanitized) > 200:  # Leave room for extensions and suffixes
            sanitized = sanitized[:200]
        return sanitized
        
    def _ensure_json_serializable(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(np.nan_to_num(obj, nan=0.0, posinf=1e10, neginf=-1e10))
        elif isinstance(obj, np.ndarray):
            return [self._ensure_json_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._ensure_json_serializable(item) for item in obj)
        else:
            return obj
        
    def load_audio(self, file_path: str, user_sample_rate: Optional[int] = None, 
                   preserve_stereo: bool = True) -> Tuple[np.ndarray, bool]:
        """Load audio file preserving original characteristics unless transformations require mono"""
        start_time = time.time()
        try:
            # Get original file info
            info = sf.info(file_path)
            original_sr = info.samplerate
            self.original_channels = info.channels
            self.original_format = info.format
            self.original_subtype = info.subtype  # Preserve bit depth info
            
            # Determine if we need to force mono (only if specific transformations require it)
            force_mono = self._requires_mono_processing()
            
            # Only resample if user explicitly requests it
            if user_sample_rate and user_sample_rate != original_sr:
                target_sr = user_sample_rate
                resampling_occurred = True
                if self.config.get('verbose'):
                    print(f"User requested resampling: {original_sr} Hz → {target_sr} Hz")
            else:
                # Use native sample rate - no resampling unless explicitly requested
                target_sr = original_sr
                resampling_occurred = False
                if self.config.get('verbose'):
                    print(f"Using native sample rate: {target_sr} Hz (no resampling)")
            
            self.sample_rate = target_sr
            
            # Load audio preserving stereo unless forced to mono
            if preserve_stereo and not force_mono and self.original_channels > 1:
                audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
                if self.config.get('verbose'):
                    print(f"Loaded as {self.original_channels}-channel audio")
            else:
                audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
                self.original_channels = 1
                if self.config.get('verbose'):
                    print("Loaded as mono audio")
            
            # Store loading stats
            self.processing_stats['load_time'] = time.time() - start_time
            self.processing_stats['original_sr'] = original_sr
            self.processing_stats['target_sr'] = target_sr
            self.processing_stats['original_channels'] = info.channels
            self.processing_stats['processed_channels'] = self.original_channels
            self.processing_stats['original_subtype'] = info.subtype
            
            if audio.ndim == 1:
                self.processing_stats['duration'] = len(audio) / target_sr
            else:
                self.processing_stats['duration'] = audio.shape[1] / target_sr
            
            return audio, resampling_occurred
        except Exception as e:
            raise Exception(f"Error loading audio file: {e}")
    
    def _requires_mono_processing(self) -> bool:
        """Check if any current transformations require mono processing"""
        # Most transformations now work fine with stereo through channel iteration
        # Only very specific analysis-based transforms might require mono
        mono_requiring_transforms = [
            # Currently no transforms require mono - all handle stereo properly
            # If we add transforms that genuinely need mono, add them here
        ]
        
        # Check if any transforms that require mono are enabled and set to True
        for transform in mono_requiring_transforms:
            if self.config.get(transform) is True:
                return True
        return False
    
    def _convert_to_mono_for_analysis(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo to mono only for analysis purposes"""
        if audio.ndim == 1:
            return audio
        else:
            # Convert stereo to mono by averaging channels
            return np.mean(audio, axis=0)
    
    def apply_preprocessing(self, audio: np.ndarray, preprocess_config: Dict) -> np.ndarray:
        """Apply preprocessing steps to audio (handles both mono and stereo)"""
        processed = audio.copy()
        
        if preprocess_config.get('denoise'):
            if self.config.get('verbose'):
                print("  Applying noise reduction")
            # Handle stereo denoising
            if processed.ndim == 2:
                for channel in range(processed.shape[0]):
                    stft = librosa.stft(processed[channel])
                    magnitude = np.abs(stft)
                    phase = np.angle(stft)
                    # Estimate noise from first 0.5 seconds
                    noise_frames = int(0.5 * self.sample_rate / self.hop_length)
                    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
                    # Subtract noise profile
                    denoised_magnitude = magnitude - 0.5 * noise_profile
                    denoised_magnitude = np.maximum(denoised_magnitude, 0.1 * magnitude)
                    processed[channel] = librosa.istft(denoised_magnitude * np.exp(1j * phase))
            else:
                # Mono denoising (original code)
                stft = librosa.stft(processed)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                noise_frames = int(0.5 * self.sample_rate / self.hop_length)
                noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
                denoised_magnitude = magnitude - 0.5 * noise_profile
                denoised_magnitude = np.maximum(denoised_magnitude, 0.1 * magnitude)
                processed = librosa.istft(denoised_magnitude * np.exp(1j * phase))
        
        if preprocess_config.get('normalize'):
            if self.config.get('verbose'):
                print("  Normalizing audio levels")
            processed = librosa.util.normalize(processed)
        
        if preprocess_config.get('trim_silence'):
            if self.config.get('verbose'):
                print("  Trimming silence")
            if processed.ndim == 2:
                # For stereo, trim based on the sum of both channels
                mono_for_trim = np.sum(processed, axis=0)
                processed, trim_indices = librosa.effects.trim(processed, top_db=20)
            else:
                processed, _ = librosa.effects.trim(processed, top_db=20)
        
        if preprocess_config.get('fade_in'):
            fade_samples = int(preprocess_config['fade_in'] * self.sample_rate)
            fade_curve = np.linspace(0, 1, fade_samples)
            if processed.ndim == 2:
                for channel in range(processed.shape[0]):
                    processed[channel, :fade_samples] *= fade_curve
            else:
                processed[:fade_samples] *= fade_curve
        
        if preprocess_config.get('fade_out'):
            fade_samples = int(preprocess_config['fade_out'] * self.sample_rate)
            fade_curve = np.linspace(1, 0, fade_samples)
            if processed.ndim == 2:
                for channel in range(processed.shape[0]):
                    processed[channel, -fade_samples:] *= fade_curve
            else:
                processed[-fade_samples:] *= fade_curve
        
        if preprocess_config.get('reverse'):
            if self.config.get('verbose'):
                print("  Reversing audio")
            if processed.ndim == 2:
                processed = processed[:, ::-1]
            else:
                processed = processed[::-1]
        
        return processed
    
    def apply_audio_transformations(self, audio: np.ndarray, transforms: Dict) -> np.ndarray:
        """Apply all user-specified audio transformations (handles both mono and stereo)"""
        processed_audio = audio.copy()
        
        # Check if any actual transformations are requested (not just flags)
        actual_transforms = {}
        for key, value in transforms.items():
            if value is not None and value is not False:
                # For boolean transforms, only include if True
                if isinstance(value, bool):
                    if value:
                        actual_transforms[key] = value
                else:
                    # For numeric transforms, include any non-None value
                    actual_transforms[key] = value
        
        if not actual_transforms:
            if self.config.get('verbose'):
                print("No actual transformations specified - skipping transformation pipeline")
            return processed_audio
        
        if self.config.get('verbose'):
            print("Applying audio transformations...")
        
        # Time-domain transformations
        if actual_transforms.get('time_stretch'):
            rate = actual_transforms['time_stretch']
            if self.config.get('verbose'):
                print(f"  Time stretching by factor {rate}")
            if processed_audio.ndim == 2:
                # Store original shape for length matching
                original_length = processed_audio.shape[1]
                expected_length = int(original_length / rate)  # Expected output length
                
                stretched_channels = []
                for channel in range(processed_audio.shape[0]):
                    stretched = librosa.effects.time_stretch(processed_audio[channel], rate=rate)
                    
                    # Ensure consistent length across all channels
                    if len(stretched) != expected_length:
                        if len(stretched) > expected_length:
                            stretched = stretched[:expected_length]
                        else:
                            pad_length = expected_length - len(stretched)
                            stretched = np.pad(stretched, (0, pad_length), mode='constant')
                    
                    stretched_channels.append(stretched)
                
                # Reconstruct stereo array with consistent shape
                processed_audio = np.array(stretched_channels)
            else:
                processed_audio = librosa.effects.time_stretch(processed_audio, rate=rate)
        
        if actual_transforms.get('pitch_shift'):
            steps = actual_transforms['pitch_shift']
            if self.config.get('verbose'):
                print(f"  Pitch shifting by {steps} semitones")
            if processed_audio.ndim == 2:
                # Store original shape for consistent length handling
                original_length = processed_audio.shape[1]
                
                shifted_channels = []
                for channel in range(processed_audio.shape[0]):
                    shifted = librosa.effects.pitch_shift(
                        processed_audio[channel], sr=self.sample_rate, n_steps=steps)
                    
                    # Ensure consistent length (pitch_shift can also change length slightly)
                    if len(shifted) != original_length:
                        if len(shifted) > original_length:
                            shifted = shifted[:original_length]
                        else:
                            pad_length = original_length - len(shifted)
                            shifted = np.pad(shifted, (0, pad_length), mode='constant')
                    
                    shifted_channels.append(shifted)
                
                processed_audio = np.array(shifted_channels)
            else:
                processed_audio = librosa.effects.pitch_shift(processed_audio, sr=self.sample_rate, n_steps=steps)
        
        # Spectral transformations (need to convert to mono for some)
        if actual_transforms.get('harmonic_only'):
            if self.config.get('verbose'):
                print("  Extracting harmonic component only")
            if processed_audio.ndim == 2:
                # Apply to each channel separately
                for channel in range(processed_audio.shape[0]):
                    processed_audio[channel] = librosa.effects.harmonic(processed_audio[channel])
            else:
                processed_audio = librosa.effects.harmonic(processed_audio)
        
        if actual_transforms.get('percussive_only'):
            if self.config.get('verbose'):
                print("  Extracting percussive component only")
            if processed_audio.ndim == 2:
                for channel in range(processed_audio.shape[0]):
                    processed_audio[channel] = librosa.effects.percussive(processed_audio[channel])
            else:
                processed_audio = librosa.effects.percussive(processed_audio)
        
        if actual_transforms.get('harmonic_percussive_ratio'):
            ratio = actual_transforms['harmonic_percussive_ratio']
            if self.config.get('verbose'):
                print(f"  Adjusting harmonic/percussive balance (ratio: {ratio})")
            if processed_audio.ndim == 2:
                for channel in range(processed_audio.shape[0]):
                    harmonic = librosa.effects.harmonic(processed_audio[channel])
                    percussive = librosa.effects.percussive(processed_audio[channel])
                    processed_audio[channel] = ratio * harmonic + (1 - ratio) * percussive
            else:
                harmonic = librosa.effects.harmonic(processed_audio)
                percussive = librosa.effects.percussive(processed_audio)
                processed_audio = ratio * harmonic + (1 - ratio) * percussive
        
        # Advanced spectral manipulations
        if any(k in actual_transforms for k in ['spectral_centroid_shift', 'spectral_bandwidth_stretch', 
                                               'frequency_mask_low', 'frequency_mask_high']):
            
            if processed_audio.ndim == 2:
                # Process each channel separately with robust shape handling
                for channel in range(processed_audio.shape[0]):
                    transformed_channel = self._apply_spectral_transforms(
                        processed_audio[channel], actual_transforms)
                    # Handle potential length mismatch from ISTFT
                    if len(transformed_channel) != processed_audio.shape[1]:
                        if len(transformed_channel) > processed_audio.shape[1]:
                            # Trim if longer
                            transformed_channel = transformed_channel[:processed_audio.shape[1]]
                        else:
                            # Pad if shorter
                            pad_length = processed_audio.shape[1] - len(transformed_channel)
                            transformed_channel = np.pad(transformed_channel, (0, pad_length), mode='constant')
                    processed_audio[channel] = transformed_channel
            else:
                processed_audio = self._apply_spectral_transforms(processed_audio, actual_transforms)
        
        # Creative effects
        if actual_transforms.get('stutter'):
            stutter_time = actual_transforms['stutter']
            if self.config.get('verbose'):
                print(f"  Adding stutter effect ({stutter_time}s)")
            stutter_samples = int(stutter_time * self.sample_rate)
            # Create stuttering by repeating small chunks
            stuttered = []
            if processed_audio.ndim == 2:
                for i in range(0, processed_audio.shape[1], stutter_samples * 4):
                    chunk = processed_audio[:, i:i + stutter_samples]
                    stuttered.extend([chunk, chunk, chunk, chunk])
                processed_audio = np.concatenate(stuttered, axis=1)[:, :processed_audio.shape[1]]
            else:
                for i in range(0, len(processed_audio), stutter_samples * 4):
                    chunk = processed_audio[i:i + stutter_samples]
                    stuttered.extend([chunk, chunk, chunk, chunk])
                processed_audio = np.concatenate(stuttered)[:len(processed_audio)]
        
        if actual_transforms.get('granular_synthesis'):
            grain_size = int(0.05 * self.sample_rate)  # 50ms grains
            if self.config.get('verbose'):
                print("  Applying granular synthesis")
            
            if processed_audio.ndim == 2:
                # Process each stereo channel separately to maintain true stereo
                processed_channels = []
                for channel in range(processed_audio.shape[0]):
                    channel_audio = processed_audio[channel]
                    grains = []
                    for i in range(0, len(channel_audio) - grain_size, grain_size // 2):
                        grain = channel_audio[i:i + grain_size]
                        # Random pitch shift per grain
                        shift = random.uniform(-2, 2)
                        grain = librosa.effects.pitch_shift(grain, sr=self.sample_rate, n_steps=shift)
                        grains.append(grain)
                    processed_channel = np.concatenate(grains)[:len(channel_audio)]
                    processed_channels.append(processed_channel)
                processed_audio = np.array(processed_channels)
            else:
                grains = []
                for i in range(0, len(processed_audio) - grain_size, grain_size // 2):
                    grain = processed_audio[i:i + grain_size]
                    # Random pitch shift per grain
                    shift = random.uniform(-2, 2)
                    grain = librosa.effects.pitch_shift(grain, sr=self.sample_rate, n_steps=shift)
                    grains.append(grain)
                processed_audio = np.concatenate(grains)[:len(processed_audio)]
        
        return processed_audio
    
    def _apply_spectral_transforms(self, audio_channel: np.ndarray, transforms: Dict) -> np.ndarray:
        """Apply spectral transformations to a single audio channel"""
        original_length = len(audio_channel)
        stft = librosa.stft(audio_channel, hop_length=self.hop_length, n_fft=self.frame_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        if transforms.get('spectral_centroid_shift'):
            shift = transforms['spectral_centroid_shift']
            if self.config.get('verbose'):
                print(f"    Shifting spectral centroid by {shift}")
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            magnitude_sum = np.sum(magnitude, axis=0)
            # Robust centroid calculation with silence detection
            magnitude_sum = np.maximum(magnitude_sum, 1e-10)  # Prevent true zeros
            centroid = np.sum(freqs[:, np.newaxis] * magnitude, axis=0) / magnitude_sum
            # Clip centroid to reasonable bounds and handle NaN/inf
            centroid = np.clip(centroid, freqs[1], freqs[-1])  # Between min and max freq
            centroid = np.nan_to_num(centroid, nan=1000.0, posinf=freqs[-1], neginf=freqs[1])
            
            for i, freq in enumerate(freqs):
                scaling = 1 + shift * (freq / (centroid + 1e-6))
                magnitude[i] *= np.clip(scaling, 0.1, 10.0)
        
        if transforms.get('frequency_mask_low') or transforms.get('frequency_mask_high'):
            low_freq = transforms.get('frequency_mask_low', 0)
            high_freq = transforms.get('frequency_mask_high', self.sample_rate // 2)
            if self.config.get('verbose'):
                print(f"    Applying frequency mask: {low_freq}Hz - {high_freq}Hz")
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            magnitude[~mask] *= 0.01
        
        stft_modified = magnitude * np.exp(1j * phase)
        reconstructed = librosa.istft(stft_modified, hop_length=self.hop_length)
        
        # Ensure output length matches input length exactly
        if len(reconstructed) != original_length:
            if len(reconstructed) > original_length:
                reconstructed = reconstructed[:original_length]
            else:
                # Pad with zeros if shorter
                pad_length = original_length - len(reconstructed)
                reconstructed = np.pad(reconstructed, (0, pad_length), mode='constant')
        
        return reconstructed
    
    def extract_onset_times(self, audio: np.ndarray) -> np.ndarray:
        """Extract note onset times from audio (handles stereo by converting to mono for analysis)"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        onset_frames = librosa.onset.onset_detect(
            y=analysis_audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            units='time'
        )
        return onset_frames
    
    def extract_pitches(self, audio: np.ndarray, method: str = 'piptrack') -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch information using various methods (uses mono conversion for analysis)"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        
        if method == 'piptrack':
            pitches, magnitudes = librosa.piptrack(
                y=analysis_audio, sr=self.sample_rate, hop_length=self.hop_length,
                fmin=80, fmax=2000
            )
            pitch_track = []
            confidence_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                confidence = magnitudes[index, t]
                pitch_track.append(pitch if confidence > 0.1 else 0)
                confidence_track.append(confidence)
            return np.array(pitch_track), np.array(confidence_track)
        
        elif method == 'crepe':
            # Simplified CREPE-like pitch tracking
            f0 = librosa.yin(analysis_audio, fmin=80, fmax=2000, sr=self.sample_rate)
            confidence = np.ones_like(f0) * 0.8  # Placeholder confidence
            return f0, confidence
        
        return self.extract_pitches(audio, 'piptrack')
    
    def extract_tempo_and_beats(self, audio: np.ndarray, method: str = 'standard') -> Tuple[float, np.ndarray]:
        """Extract tempo and beat positions using various methods"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        
        if method == 'advanced':
            # Multi-scale tempo estimation
            tempos = []
            for hop in [256, 512, 1024]:
                tempo, _ = librosa.beat.beat_track(y=analysis_audio, sr=self.sample_rate, hop_length=hop)
                tempos.append(tempo)
            tempo = np.median(tempos)
            _, beats = librosa.beat.beat_track(y=analysis_audio, sr=self.sample_rate, units='time')
            return tempo, beats
        else:
            tempo, beats = librosa.beat.beat_track(y=analysis_audio, sr=self.sample_rate, units='time')
            return tempo, beats
    
    def detect_key(self, audio: np.ndarray) -> Tuple[str, str]:
        """Detect musical key of the audio"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        chroma = librosa.feature.chroma_stft(y=analysis_audio, sr=self.sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Simple key detection using chroma centroid
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)
        
        # Determine major/minor (simplified)
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        major_score = np.dot(chroma_mean, np.roll(major_profile, key_idx))
        minor_score = np.dot(chroma_mean, np.roll(minor_profile, key_idx))
        
        mode = 'major' if major_score > minor_score else 'minor'
        return key_names[key_idx], mode
    
    def analyze_chords(self, audio: np.ndarray) -> List[Dict]:
        """Analyze chord progressions"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        chroma = librosa.feature.chroma_stft(y=analysis_audio, sr=self.sample_rate, hop_length=self.hop_length)
        times = librosa.frames_to_time(range(chroma.shape[1]), sr=self.sample_rate, hop_length=self.hop_length)
        
        # Simple chord templates (major and minor triads)
        chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            'Am': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        }
        
        chords = []
        for i, frame in enumerate(chroma.T):
            best_chord = max(chord_templates.keys(), 
                           key=lambda c: np.dot(frame, chord_templates[c]))
            chords.append({
                'time': times[i],
                'chord': best_chord,
                'confidence': np.dot(frame, chord_templates[best_chord])
            })
        
        return chords
    
    # NEW FEATURE: Extract detailed timbre and texture features
    def extract_timbre_features(self, audio: np.ndarray) -> Dict:
        """Extract comprehensive timbre and texture analysis features"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        
        features = {
            'spectral_centroid': librosa.feature.spectral_centroid(y=analysis_audio, sr=self.sample_rate).tolist(),
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=analysis_audio, sr=self.sample_rate).tolist(),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=analysis_audio, sr=self.sample_rate).tolist(),
            'spectral_flatness': librosa.feature.spectral_flatness(y=analysis_audio).tolist(),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(analysis_audio).tolist(),
            'mfcc': librosa.feature.mfcc(y=analysis_audio, sr=self.sample_rate, n_mfcc=13).tolist(),
            'chroma': librosa.feature.chroma_stft(y=analysis_audio, sr=self.sample_rate).tolist(),
            'tonnetz': librosa.feature.tonnetz(y=analysis_audio, sr=self.sample_rate).tolist(),
            'contrast': librosa.feature.spectral_contrast(y=analysis_audio, sr=self.sample_rate).tolist(),
        }
        
        # Add summary statistics - create a copy of the keys to avoid modification during iteration
        feature_names = list(features.keys())
        for feature_name in feature_names:
            feature_values = features[feature_name]
            if isinstance(feature_values[0], list):  # Multi-dimensional features
                feature_array = np.array(feature_values)
                # Ensure proper JSON serialization with explicit type conversion
                features[f'{feature_name}_mean'] = [float(x) for x in np.mean(feature_array, axis=1)]
                features[f'{feature_name}_std'] = [float(x) for x in np.std(feature_array, axis=1)]
            else:  # Single-dimensional features
                feature_array = np.array(feature_values[0])  # Take first row for single-dim features
                # Robust type conversion with NaN handling
                mean_val = np.mean(feature_array)
                std_val = np.std(feature_array)
                features[f'{feature_name}_mean'] = float(np.nan_to_num(mean_val, nan=0.0))
                features[f'{feature_name}_std'] = float(np.nan_to_num(std_val, nan=0.0))
        
        return features
    
    # NEW FEATURE: Extract detailed onset and beat annotations
    def extract_detailed_rhythm_analysis(self, audio: np.ndarray, tempo: float, beats: np.ndarray) -> Dict:
        """Extract comprehensive rhythm and timing analysis"""
        analysis_audio = self._convert_to_mono_for_analysis(audio)
        
        # Enhanced onset detection with multiple methods
        onset_envelope = librosa.onset.onset_strength(y=analysis_audio, sr=self.sample_rate)
        onsets_complex = librosa.onset.onset_detect(y=analysis_audio, sr=self.sample_rate, 
                                                   onset_envelope=onset_envelope, units='time')
        
        # Detect downbeats (if possible)
        try:
            downbeats = librosa.beat.beat_track(y=analysis_audio, sr=self.sample_rate, units='time')[1]
        except:
            downbeats = beats[::4] if len(beats) >= 4 else beats  # Estimate every 4th beat
        
        # Calculate tempo stability and variations
        tempo_track = librosa.beat.tempo(onset_envelope=onset_envelope, sr=self.sample_rate)
        
        # Calculate rhythm patterns
        beat_intervals = np.diff(beats) if len(beats) > 1 else np.array([60/tempo])
        beat_consistency = np.std(beat_intervals) / np.mean(beat_intervals) if len(beat_intervals) > 0 else 0
        
        rhythm_analysis = {
            'tempo': float(tempo),
            'tempo_stability': float(1 / (1 + np.std(tempo_track))),  # Higher = more stable
            'beats': beats.tolist(),
            'beat_intervals': beat_intervals.tolist(),
            'beat_consistency': float(beat_consistency),
            'onsets': onsets_complex.tolist(),
            'downbeats': downbeats.tolist(),
            'onset_density': float(len(onsets_complex) / (len(analysis_audio) / self.sample_rate)),
            'rhythmic_complexity': float(np.std(np.diff(onsets_complex)) if len(onsets_complex) > 1 else 0)
        }
        
        return rhythm_analysis
    
    # NEW FEATURE: Create audio thumbnail/preview
    def create_audio_thumbnail(self, audio: np.ndarray, duration: float = 5.0, 
                              output_path: str = None) -> str:
        """Create a short preview clip of the audio"""
        if output_path is None:
            output_path = "thumbnail.wav"
        
        # Calculate samples for the desired duration
        thumbnail_samples = int(duration * self.sample_rate)
        
        # Take from the middle of the track for best representation
        if audio.ndim == 1:
            total_samples = len(audio)
            start_sample = max(0, (total_samples - thumbnail_samples) // 2)
            thumbnail = audio[start_sample:start_sample + thumbnail_samples]
        else:
            total_samples = audio.shape[1]
            start_sample = max(0, (total_samples - thumbnail_samples) // 2)
            thumbnail = audio[:, start_sample:start_sample + thumbnail_samples]
        
        # Save the thumbnail
        sf.write(output_path, thumbnail.T if thumbnail.ndim == 2 else thumbnail, self.sample_rate)
        
        if self.config.get('verbose'):
            print(f"Audio thumbnail saved: {output_path}")
        
        return output_path
    
    def hz_to_midi(self, frequency: float) -> int:
        """Convert frequency to MIDI note number"""
        if frequency <= 0:
            return 0
        return int(round(69 + 12 * np.log2(frequency / 440.0)))
    
    def midi_to_note_name(self, midi_num: int) -> str:
        """Convert MIDI number to note name"""
        if midi_num == 0:
            return "rest"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_num // 12) - 1
        note = note_names[midi_num % 12]
        return f"{note}{octave}"
    
    def quantize_duration(self, duration: float, tempo: float) -> Tuple[str, int]:
        """Quantize duration to musical note values with proper divisions"""
        beat_duration = 60.0 / float(tempo)
        duration_in_beats = float(duration) / beat_duration
        
        # Use 480 divisions per quarter note (standard MIDI resolution)
        divisions_per_quarter = 480
        duration_in_divisions = duration_in_beats * divisions_per_quarter
        
        # Common note durations in divisions
        note_types = [
            (divisions_per_quarter * 4, "whole"),     # 1920 divisions
            (divisions_per_quarter * 2, "half"),      # 960 divisions  
            (divisions_per_quarter, "quarter"),       # 480 divisions
            (divisions_per_quarter // 2, "eighth"),   # 240 divisions
            (divisions_per_quarter // 4, "16th"),     # 120 divisions
            (divisions_per_quarter // 8, "32nd"),     # 60 divisions
        ]
        
        # Find closest standard duration
        closest_divisions = min(note_types, key=lambda x: abs(x[0] - duration_in_divisions))
        
        # Return type and actual divisions (clamped to reasonable values)
        actual_divisions = max(60, min(1920, int(round(duration_in_divisions))))
        return closest_divisions[1], actual_divisions
    
    def group_notes_into_measures(self, notes: List[Dict], beats: np.ndarray, tempo: float) -> List[List[Dict]]:
        """Group notes into measures based on time intervals"""
        if len(notes) == 0:
            return []
        
        # Simple approach: group notes by time intervals (4 seconds per measure as baseline)
        measure_duration = 4.0  # seconds per measure
        measures = []
        current_measure = []
        current_measure_start = 0.0
        
        for note in notes:
            note_time = note['onset_time']
            
            # If note is beyond current measure duration, start new measure
            if note_time >= current_measure_start + measure_duration:
                if current_measure:
                    measures.append(current_measure)
                current_measure = []
                current_measure_start = note_time
            
            current_measure.append(note)
        
        # Add the final measure
        if current_measure:
            measures.append(current_measure)
        
        # If no measures were created, put all notes in one measure
        if not measures:
            measures = [notes]
        
        return measures
    
    def calculate_time_signature(self, measure_notes: List[Dict]) -> Tuple[int, int]:
        """Calculate the appropriate time signature for a measure based on its notes"""
        if not measure_notes:
            return (4, 4)  # Default fallback
        
        # Sum up all note durations in 480-division units
        total_divisions = sum(note.get('duration_divisions', 480) for note in measure_notes)
        
        # Convert to quarter notes (480 divisions = 1 quarter note)
        quarter_notes = total_divisions / 480.0
        
        # Always use simple time signatures that match common patterns
        # Round to nearest quarter note and use that as numerator with /4 denominator
        numerator = max(1, round(quarter_notes))
        
        # Cap at reasonable values
        if numerator > 8:
            numerator = 8
            
        return (numerator, 4)
    
    def normalize_measure_durations(self, measure_notes: List[Dict], time_sig: Tuple[int, int]) -> List[Dict]:
        """Adjust note durations to fit exactly in the given time signature"""
        if not measure_notes:
            return measure_notes
        
        # Calculate expected total divisions for this time signature
        expected_divisions = time_sig[0] * (480 * 4 // time_sig[1])  # e.g., 4/4 = 4 * 480 = 1920
        
        # Calculate actual total divisions
        actual_divisions = sum(note.get('duration_divisions', 480) for note in measure_notes)
        
        if actual_divisions == 0:
            return measure_notes
        
        # Scale all durations proportionally to fit exactly
        scale_factor = expected_divisions / actual_divisions
        
        normalized_notes = []
        total_so_far = 0
        
        for i, note in enumerate(measure_notes):
            note_copy = note.copy()
            
            if i == len(measure_notes) - 1:
                # Last note: use whatever divisions are left to ensure exact fit
                note_copy['duration_divisions'] = expected_divisions - total_so_far
            else:
                # Scale proportionally and round to integer
                scaled_duration = int(round(note['duration_divisions'] * scale_factor))
                # Ensure minimum duration
                scaled_duration = max(60, scaled_duration)  # At least 32nd note
                note_copy['duration_divisions'] = scaled_duration
                total_so_far += scaled_duration
            
            # Update note type based on new duration
            if note_copy['duration_divisions'] >= 1440:  # >= dotted half
                note_copy['type'] = 'half'
            elif note_copy['duration_divisions'] >= 720:  # >= dotted quarter
                note_copy['type'] = 'quarter' 
            elif note_copy['duration_divisions'] >= 360:  # >= dotted eighth
                note_copy['type'] = 'eighth'
            elif note_copy['duration_divisions'] >= 180:  # >= dotted 16th
                note_copy['type'] = '16th'
            else:
                note_copy['type'] = '32nd'
            
            normalized_notes.append(note_copy)
        
        return normalized_notes
    
    def create_musicxml(self, notes: List[Dict], tempo: float, beats: np.ndarray) -> str:
        """Create MusicXML with dynamic time signatures based on actual musical content"""
        
        if self.config.get('verbose'):
            print(f"DEBUG: Creating MusicXML with {len(notes)} notes, tempo {tempo}")
        
        # Add proper duration info to each note
        for note in notes:
            note_type, duration_divisions = self.quantize_duration(note.get('duration', 0.5), tempo)
            note['duration_divisions'] = duration_divisions
            note['type'] = note_type
        
        # Group notes into natural measures
        measure_groups = self.group_notes_into_measures(notes, beats, tempo)
        
        if self.config.get('verbose'):
            print(f"DEBUG: Created {len(measure_groups)} measures")
        
        # Create MusicXML structure
        score_partwise = Element('score-partwise', version='3.1')
        
        work = SubElement(score_partwise, 'work')
        work_title = SubElement(work, 'work-title')
        work_title.text = 'Transcribed Audio'
        
        part_list = SubElement(score_partwise, 'part-list')
        score_part = SubElement(part_list, 'score-part', id='P1')
        part_name = SubElement(score_part, 'part-name')
        part_name.text = 'Audio Track'
        
        part = SubElement(score_partwise, 'part', id='P1')
        
        # Create measures with dynamic time signatures
        for measure_num, measure_notes in enumerate(measure_groups, 1):
            if not measure_notes:
                continue
                
            # Calculate time signature for this measure
            time_sig_num, time_sig_den = self.calculate_time_signature(measure_notes)
            
            # Normalize note durations to fit exactly in this time signature
            normalized_notes = self.normalize_measure_durations(measure_notes, (time_sig_num, time_sig_den))
                
            measure = SubElement(part, 'measure', number=str(measure_num))
            
            # Add attributes to first measure or when time signature changes
            prev_time_sig = (4, 4) if measure_num == 1 else self.calculate_time_signature(measure_groups[measure_num-2])
            
            if measure_num == 1 or (time_sig_num, time_sig_den) != prev_time_sig:
                attributes = SubElement(measure, 'attributes')
                
                # Use 480 divisions per quarter note
                divisions = SubElement(attributes, 'divisions')
                divisions.text = '480'
                
                if measure_num == 1:
                    # Key signature (only in first measure)
                    key = SubElement(attributes, 'key')
                    fifths = SubElement(key, 'fifths')
                    fifths.text = '0'
                
                # Time signature
                time = SubElement(attributes, 'time')
                beats_elem = SubElement(time, 'beats')
                beats_elem.text = str(time_sig_num)
                beat_type = SubElement(time, 'beat-type')
                beat_type.text = str(time_sig_den)
            
            # Add notes to measure (using normalized durations)
            for note_info in normalized_notes:
                note = SubElement(measure, 'note')
                
                if note_info['pitch'] == 'rest':
                    rest = SubElement(note, 'rest')
                else:
                    # Parse note name safely
                    pitch_str = str(note_info['pitch'])
                    if len(pitch_str) >= 2:
                        pitch = SubElement(note, 'pitch')
                        step = SubElement(pitch, 'step')
                        step.text = pitch_str[0].upper()
                        
                        if '#' in pitch_str:
                            alter = SubElement(pitch, 'alter')
                            alter.text = '1'
                        elif 'b' in pitch_str:
                            alter = SubElement(pitch, 'alter')
                            alter.text = '-1'
                        
                        # Extract octave safely
                        octave_str = ''.join(filter(str.isdigit, pitch_str))
                        if octave_str:
                            octave = SubElement(pitch, 'octave')
                            octave.text = octave_str
                        else:
                            octave = SubElement(pitch, 'octave')
                            octave.text = '4'
                    else:
                        # Invalid pitch, make it a rest
                        rest = SubElement(note, 'rest')
                
                # Duration in divisions
                duration_elem = SubElement(note, 'duration')
                duration_elem.text = str(note_info['duration_divisions'])
                
                # Note type
                note_type = SubElement(note, 'type')
                note_type.text = note_info['type']
        
        if self.config.get('verbose'):
            print(f"DEBUG: Created {len(measure_groups)} measures with dynamic time signatures")
        
        rough_string = tostring(score_partwise, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent='  ')
    
    def process_audio_file(self, file_path: str, output_dir: Optional[str] = None, 
                      filename_prefix: Optional[str] = None, user_sample_rate: Optional[int] = None, 
                      transforms: Optional[Dict] = None, preprocess_config: Optional[Dict] = None) -> str:
        """Main processing function"""
        start_time = time.time()
        
        if self.config.get('verbose'):
            print(f"Loading audio file: {file_path}")
        
        # Dry run check
        if self.config.get('dry_run'):
            print(f"[DRY RUN] Would process: {file_path}")
            return file_path
        
        try:
            # Extract filename for output files
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            filename = filename_prefix if filename_prefix else base_filename
            filename_base = filename  # This is the missing variable definition!
            
            # Set default output directory if not provided
            if output_dir is None:
                output_dir = os.path.dirname(file_path) or os.getcwd()
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if any transformations are specified
            transforms_applied = transforms and any(v is not None and v is not False for v in transforms.values())
            preprocess_applied = preprocess_config and any(v for v in preprocess_config.values())
            
            audio, resampling_occurred = self.load_audio(file_path, user_sample_rate, 
                                                    preserve_stereo=True)
            
            if self.config.get('verbose'):
                print(f"Processing at sample rate: {self.sample_rate} Hz")
                print(f"Channels: {self.original_channels}")
            
            # Apply preprocessing
            if preprocess_config:
                audio = self.apply_preprocessing(audio, preprocess_config)
            
            # Apply transformations if specified
            transform_applied = False
            transform_suffix = None
            if transforms and any(v is not None for v in transforms.values()):
                audio = self.apply_audio_transformations(audio, transforms)
                transform_applied = True
                transform_suffix = "transformed"
            
            # Save processed audio ONLY if resampling occurred, transformations applied, or preprocessing applied
            if resampling_occurred or transform_applied or preprocess_applied:
                if self.config.get('verbose'):
                    print(f"Saving processed audio due to: resampling={resampling_occurred}, transforms={transform_applied}, preprocessing={preprocess_applied}")
                self.save_processed_audio(audio, file_path, output_dir, transform_suffix)
            elif self.config.get('verbose'):
                print("No transformations applied - using original file for analysis")
            
            if self.config.get('verbose'):
                print("Extracting musical features...")
            
            # Musical analysis (convert to mono for analysis if needed)
            analysis_audio = self._convert_to_mono_for_analysis(audio)
            
            if self.config.get('verbose'):
                print("DEBUG: Starting tempo detection")
            tempo, beats = self.extract_tempo_and_beats(audio, 
                                                    self.config.get('tempo_detection_method', 'standard'))
            if self.config.get('verbose'):
                print(f"DEBUG: Tempo type: {type(tempo)}, value: {tempo}")
            tempo = float(tempo)  # Ensure it's a Python float
            
            if self.config.get('verbose'):
                print(f"Detected tempo: {tempo:.1f} BPM")
            
            if self.config.get('verbose'):
                print("DEBUG: Starting onset detection")
            onsets = self.extract_onset_times(audio)
            if self.config.get('verbose'):
                print(f"DEBUG: Onsets type: {type(onsets)}, shape: {onsets.shape if hasattr(onsets, 'shape') else 'no shape'}")
            onsets = np.asarray(onsets, dtype=np.float64)  # Ensure proper numpy array
            
            if self.config.get('verbose'):
                print(f"Found {len(onsets)} note onsets")
            
            if self.config.get('verbose'):
                print("DEBUG: Starting pitch detection")
            pitches, confidences = self.extract_pitches(audio, 
                                                    self.config.get('pitch_detection_method', 'piptrack'))
            if self.config.get('verbose'):
                print(f"DEBUG: Pitches type: {type(pitches)}, shape: {pitches.shape if hasattr(pitches, 'shape') else 'no shape'}")
            pitches = np.asarray(pitches, dtype=np.float64)
            confidences = np.asarray(confidences, dtype=np.float64)
            
            # Key detection
            if self.config.get('verbose'):
                print("DEBUG: Starting key detection")
            key_sig = self.detect_key(audio)
            if self.config.get('verbose'):
                print(f"Detected key: {key_sig[0]} {key_sig[1]}")
            
            # Chord analysis
            chords = None
            if self.config.get('chord_analysis'):
                if self.config.get('verbose'):
                    print("DEBUG: Starting chord analysis")
                chords = self.analyze_chords(audio)
                if self.config.get('verbose'):
                    print(f"Analyzed {len(chords)} chord frames")
            
            # NEW: Timbre analysis
            timbre_features = None
            if self.config.get('timbre_features_json'):
                if self.config.get('verbose'):
                    print("DEBUG: Starting timbre analysis")
                timbre_features = self.extract_timbre_features(audio)
                if self.config.get('verbose'):
                    print("Timbre features extracted")
            
            # NEW: Detailed rhythm analysis
            rhythm_analysis = None
            if self.config.get('detailed_rhythm_analysis'):
                if self.config.get('verbose'):
                    print("DEBUG: Starting detailed rhythm analysis")
                rhythm_analysis = self.extract_detailed_rhythm_analysis(audio, tempo, beats)
                if self.config.get('verbose'):
                    print("Detailed rhythm analysis complete")
            
            # Convert to time-based representation
            if self.config.get('verbose'):
                print("DEBUG: Converting to time-based representation")
            times = librosa.frames_to_time(range(len(pitches)), 
                                        sr=self.sample_rate, 
                                        hop_length=self.hop_length)
            
            # Create note events
            if self.config.get('verbose'):
                print("DEBUG: Creating note events")
            notes = []
            confidence_threshold = self.config.get('confidence_threshold', 0.1)
            
            for i, onset_time in enumerate(onsets):
                try:
                    if self.config.get('verbose') and i < 10:  # Only show first 10 for brevity
                        print(f"DEBUG: Processing onset {i}, onset_time type: {type(onset_time)}, value: {onset_time}")
                    onset_time = float(onset_time)  # Convert to Python float
                    
                    onset_frame = int(onset_time * self.sample_rate / self.hop_length)
                    if onset_frame < len(pitches):
                        pitch_hz = float(pitches[onset_frame])
                        confidence = float(confidences[onset_frame])
                        
                        if self.config.get('verbose') and i < 10:
                            print(f"DEBUG: pitch_hz type: {type(pitch_hz)}, confidence type: {type(confidence)}")
                        
                        # Apply confidence threshold
                        if confidence < confidence_threshold:
                            continue
                        
                        if i < len(onsets) - 1:
                            duration = float(onsets[i + 1]) - onset_time
                        else:
                            if audio.ndim == 1:
                                duration = float(len(audio)) / float(self.sample_rate) - onset_time
                            else:
                                duration = float(audio.shape[1]) / float(self.sample_rate) - onset_time
                        
                        if self.config.get('verbose') and i < 10:
                            print(f"DEBUG: duration type: {type(duration)}, value: {duration}")
                        
                        if pitch_hz > 0:
                            if self.config.get('verbose') and i < 10:
                                print(f"DEBUG: Converting pitch_hz {pitch_hz} to MIDI")
                            midi_note = self.hz_to_midi(pitch_hz)
                            if self.config.get('verbose') and i < 10:
                                print(f"DEBUG: MIDI note: {midi_note}")
                            note_name = self.midi_to_note_name(midi_note)
                            
                            # Pitch correction to nearest semitone
                            if self.config.get('pitch_correction'):
                                corrected_hz = librosa.midi_to_hz(midi_note)
                                note_name = self.midi_to_note_name(midi_note)
                            
                            notes.append({
                                'pitch': note_name,
                                'onset_time': onset_time,
                                'duration': duration,  # Store raw duration for later quantization
                                'confidence': confidence
                            })
                            
                except Exception as e:
                    if self.config.get('verbose'):
                        print(f"DEBUG: Error processing onset {i}: {e}")
                        print(f"DEBUG: onset_time: {onset_time}, type: {type(onset_time)}")
                    raise e
            
            if self.config.get('verbose'):
                print(f"Extracted {len(notes)} notes")
            
            # Generate visualizations and new features
            if any(self.config.get(viz) for viz in ['waveform_png', 'generate_spectrogram', 'chromagram_image',
                                                'generate_cqt', 'generate_mel_spectrogram', 'fft_spectrum_csv']):
                self.generate_visualizations(audio, output_dir, filename_base)
            
            # NEW: Create audio thumbnail
            if self.config.get('audio_thumbnail'):
                thumbnail_path = os.path.join(output_dir, f"{filename}_thumbnail.wav")
                self.create_audio_thumbnail(audio, duration=self.config.get('thumbnail_duration', 5.0), 
                                        output_path=thumbnail_path)
            
            # NEW: Generate piano roll visualization
            if self.config.get('piano_roll_png'):
                self.generate_piano_roll(notes, output_dir, filename, tempo)
            
            # Save in multiple formats
            self.save_multiple_formats(notes, tempo, output_dir, filename, key_sig, chords, beats,
                                    timbre_features, rhythm_analysis)
            
            # Generate analysis report
            if self.config.get('analysis_report'):
                report_path = os.path.join(output_dir, f"{filename}_report.txt")
                self.generate_analysis_report(notes, tempo, key_sig, chords, file_path, report_path,
                                            timbre_features, rhythm_analysis)
            
            # AI Analysis and Processing
            if self.ai_processor and self.ai_processor.is_available():
                if self.config.get('ai_analyze'):
                    if self.config.get('verbose'):
                        print("Running AI analysis...")
                    analysis_data = {
                        'tempo': tempo,
                        'key': f"{key_sig[0]} {key_sig[1]}",
                        'duration': self.processing_stats.get('duration', 0),
                        'notes': notes,
                        'timbre_features': timbre_features,
                        'rhythm_analysis': rhythm_analysis
                    }
                    ai_analysis = self.ai_processor.analyze_audio_with_ai(analysis_data, 
                                                                         self.config.get('ai_prompt'))
                    print(f"\n\033[1;36m🤖 AI Analysis:\033[0m\n{ai_analysis}\n")
                    
                    # Save AI analysis
                    with open(os.path.join(output_dir, f"{filename}_ai_analysis.txt"), 'w') as f:
                        f.write(ai_analysis)
            
            # 🎵 DARUNIA MODE ACTIVATION 🎵
            if self.config.get('darunia_mode'):
                if self.config.get('verbose'):
                    print("Brother... prepare yourself for PURE BLISS...")
                # Gather all transformation data for Darunia's assessment
                applied_transforms = transforms or {}
                analysis_for_darunia = {
                    'tempo': tempo,
                    'key': f"{key_sig[0]} {key_sig[1]}",
                    'duration': self.processing_stats.get('duration', 0),
                    'notes_count': len(notes)
                }
                if not self.ai_processor:
                    self.ai_processor = AIProcessor(self.config)  # Create minimal instance for Darunia
                self.ai_processor.darunia_mode(analysis_for_darunia, applied_transforms)
            
            # Set output path for return value
            output_path = os.path.join(output_dir, f"{filename}.musicxml")
            
            # DAW project creation
            if self.config.get('send_to_daw'):
                daw = self.config['send_to_daw']
                if transform_applied or preprocess_applied or resampling_occurred:
                    # Use the processed audio if it exists
                    processed_audio_path = self.save_processed_audio(audio, file_path, output_dir, "for_daw")
                else:
                    # Use original file
                    processed_audio_path = file_path
                self.create_daw_project(notes, processed_audio_path, tempo, daw, output_dir)
            
            # Webhook notification
            if self.config.get('webhook_notify'):
                self._send_webhook_notification(file_path, output_path, self.config['webhook_notify'])
            
            # Benchmark reporting
            if self.config.get('benchmark'):
                total_time = time.time() - start_time
                print(f"\nBenchmark Results:")
                print(f"Total processing time: {total_time:.3f} seconds")
                print(f"Load time: {self.processing_stats.get('load_time', 0):.3f} seconds")
                print(f"Audio duration: {self.processing_stats.get('duration', 0):.3f} seconds")
                print(f"Real-time factor: {self.processing_stats.get('duration', 0) / total_time:.2f}x")
            
            if self.config.get('verbose'):
                print(f"Processing completed successfully!")
                print(f"Primary output: {output_path}")
            
            return output_path
            
        except Exception as e:
            if self.config.get('verbose'):
                print(f"DEBUG: Full error details: {e}")
                import traceback
                traceback.print_exc()
            raise e
    
    def save_processed_audio(self, audio: np.ndarray, original_file_path: str, 
                           output_dir: str, suffix: str = None) -> str:
        """Save the processed audio to specified output directory with appropriate filename, preserving format characteristics"""
        file_name = os.path.basename(original_file_path)
        name_without_ext, original_ext = os.path.splitext(file_name)
        # Sanitize filename for cross-platform compatibility
        name_without_ext = self._sanitize_filename(name_without_ext)
        
        if suffix:
            suffix = self._sanitize_filename(suffix)
            processed_filename = f"{name_without_ext}.{self.sample_rate}.{suffix}.wav"
        else:
            processed_filename = f"{name_without_ext}.{self.sample_rate}.wav"
        processed_path = os.path.join(output_dir, processed_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare audio for saving (handle both mono and stereo)
        if audio.ndim == 2:
            # Stereo - transpose for soundfile (channels last)
            audio_to_save = audio.T
        else:
            # Mono
            audio_to_save = audio
        
        # Preserve original bit depth if possible
        subtype = getattr(self, 'original_subtype', 'PCM_16')
        if 'FLOAT' in subtype:
            subtype = 'FLOAT'  # Use 32-bit float
        elif 'PCM_32' in subtype:
            subtype = 'PCM_32'  # Use 32-bit PCM
        elif 'PCM_24' in subtype:
            subtype = 'PCM_24'  # Use 24-bit PCM
        else:
            subtype = 'PCM_16'  # Default to 16-bit
        
        try:
            sf.write(processed_path, audio_to_save, self.sample_rate, subtype=subtype)
        except:
            # Fallback to 16-bit if the subtype isn't supported
            sf.write(processed_path, audio_to_save, self.sample_rate, subtype='PCM_16')
            
        if self.config.get('verbose'):
            print(f"Processed audio saved to: {processed_path}")
        
        return processed_path
    
    def generate_visualizations(self, audio: np.ndarray, output_dir: str, filename: str):
        """Generate various visualizations including new AI hearing features"""
        if self.config.get('verbose'):
            print("Generating visualizations...")
        
        fig_size = (12, 8)
        
        # Convert to mono for some visualizations if needed
        if audio.ndim == 2:
            mono_audio = np.mean(audio, axis=0)
        else:
            mono_audio = audio
        
        # Waveform
        if self.config.get('waveform_png'):
            plt.figure(figsize=fig_size)
            if audio.ndim == 2:
                # Plot stereo
                times = np.linspace(0, audio.shape[1] / self.sample_rate, audio.shape[1])
                plt.subplot(2, 1, 1)
                plt.plot(times, audio[0])
                plt.title('Waveform - Left Channel')
                plt.ylabel('Amplitude')
                plt.subplot(2, 1, 2)
                plt.plot(times, audio[1])
                plt.title('Waveform - Right Channel')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
            else:
                # Plot mono
                times = np.linspace(0, len(audio) / self.sample_rate, len(audio))
                plt.plot(times, audio)
                plt.title('Waveform')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_waveform.png"), dpi=300)
            plt.close()
        
        # Spectrogram
        if self.config.get('generate_spectrogram'):
            plt.figure(figsize=fig_size)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(mono_audio)), ref=np.max)
            librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_spectrogram.png"), dpi=300)
            plt.close()
        
        # Chromagram
        if self.config.get('chromagram_image'):
            plt.figure(figsize=fig_size)
            chroma = librosa.feature.chroma_stft(y=mono_audio, sr=self.sample_rate)
            librosa.display.specshow(chroma, sr=self.sample_rate, x_axis='time', y_axis='chroma')
            plt.colorbar()
            plt.title('Chromagram')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_chromagram.png"), dpi=300)
            plt.close()
        
        # NEW: Constant-Q Transform (CQT)
        if self.config.get('generate_cqt'):
            plt.figure(figsize=fig_size)
            cqt = np.abs(librosa.cqt(mono_audio, sr=self.sample_rate))
            librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                                   sr=self.sample_rate, x_axis='time', y_axis='cqt_note')
            plt.title('Constant-Q Transform (CQT)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_cqt.png"), dpi=300)
            plt.close()
            
            # Save CQT data as numpy array
            if self.config.get('save_cqt_data'):
                np.save(os.path.join(output_dir, f"{filename}_cqt.npy"), cqt)
        
        # NEW: Mel Spectrogram
        if self.config.get('generate_mel_spectrogram'):
            plt.figure(figsize=fig_size)
            mel_spec = librosa.feature.melspectrogram(y=mono_audio, sr=self.sample_rate)
            mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
            librosa.display.specshow(mel_spec_db, sr=self.sample_rate, x_axis='time', y_axis='mel')
            plt.title('Mel Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_mel_spectrogram.png"), dpi=300)
            plt.close()
            
            # Save mel spectrogram data
            if self.config.get('save_mel_data'):
                np.save(os.path.join(output_dir, f"{filename}_mel_spec.npy"), mel_spec)
        
        # NEW: FFT Spectrum Snapshot
        if self.config.get('fft_spectrum_csv'):
            D = np.abs(librosa.stft(mono_audio))
            # Average spectrum across all time frames
            avg_spectrum = np.mean(D, axis=1)
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            
            # Save as CSV
            spectrum_data = np.column_stack([freqs, avg_spectrum])
            np.savetxt(os.path.join(output_dir, f"{filename}_fft_spectrum.csv"), 
                      spectrum_data, delimiter=",", header="frequency_hz,magnitude", comments="")
            
            # Also create a plot
            plt.figure(figsize=fig_size)
            plt.plot(freqs, avg_spectrum)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('Average FFT Spectrum')
            plt.xlim(0, self.sample_rate // 2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{filename}_fft_spectrum.png"), dpi=300)
            plt.close()
    
    # NEW: Generate piano roll visualization
    def generate_piano_roll(self, notes: List[Dict], output_dir: str, filename: str, tempo: float):
        """Generate a piano roll visualization from detected notes"""
        if not notes:
            return
        
        if self.config.get('verbose'):
            print("Generating piano roll visualization...")
        
        # Prepare data for piano roll
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Extract note data
        note_times = []
        note_pitches = []
        note_durations = []
        
        for note in notes:
            if note['pitch'] != 'rest':
                try:
                    # Convert note name to MIDI number for plotting
                    midi_num = self.hz_to_midi(librosa.note_to_hz(note['pitch']))
                    note_times.append(note['onset_time'])
                    note_pitches.append(midi_num)
                    note_durations.append(note.get('duration', 0.5))
                except:
                    continue  # Skip invalid notes
        
        if not note_times:
            return
        
        # Create piano roll bars
        for i, (time, pitch, duration) in enumerate(zip(note_times, note_pitches, note_durations)):
            # Color notes based on pitch (higher = brighter)
            color_intensity = (pitch - min(note_pitches)) / (max(note_pitches) - min(note_pitches) + 1)
            color = plt.cm.viridis(color_intensity)
            
            # Draw note rectangle
            rect = plt.Rectangle((time, pitch - 0.4), duration, 0.8, 
                               facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        # Set up the plot
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('MIDI Note Number')
        ax.set_title(f'Piano Roll Visualization (Tempo: {tempo:.1f} BPM)')
        
        if note_pitches:
            ax.set_ylim(min(note_pitches) - 2, max(note_pitches) + 2)
            ax.set_xlim(0, max(note_times) + max(note_durations))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add note names on y-axis
        if note_pitches:
            midi_range = range(int(min(note_pitches)), int(max(note_pitches)) + 1)
            note_names = [self.midi_to_note_name(midi) for midi in midi_range]
            ax.set_yticks(midi_range)
            ax.set_yticklabels(note_names, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}_piano_roll.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_daw_project(self, notes: List[Dict], audio_file: str, tempo: float, daw: str, output_dir: str):
        """Create DAW project files"""
        if self.config.get('verbose'):
            print(f"Creating {daw.upper()} project...")
        
        project_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        if daw == 'ardour':
            self._create_ardour_project(notes, audio_file, tempo, output_dir, project_name)
        elif daw == 'bitwig':
            self._create_bitwig_project(notes, audio_file, tempo, output_dir, project_name)
        elif daw == 'reaper':
            self._create_reaper_project(notes, audio_file, tempo, output_dir, project_name)
        elif daw == 'ableton':
            self._create_ableton_project(notes, audio_file, tempo, output_dir, project_name)
    
    def _create_ardour_project(self, notes: List[Dict], audio_file: str, tempo: float, output_dir: str, project_name: str):
        """Create Ardour session"""
        session_dir = os.path.join(output_dir, f"{project_name}_ardour")
        os.makedirs(session_dir, exist_ok=True)
        
        # Calculate audio length for session
        if notes:
            audio_length = max(note['onset_time'] + note.get('duration', 0.5) for note in notes)
        else:
            audio_length = 60  # Default 1 minute
        
        audio_length_samples = int(audio_length * self.sample_rate)
        
        # Basic Ardour session XML
        session_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Session version="3001" name="{project_name}" sample-rate="{self.sample_rate}" end-is-free="yes" session-range-location="yes" id-counter="1000" name-counter="1" event-counter="0" vca-counter="1" order-keys-counter="0">
  <Config>
    <Option name="native-file-data-format" value="FormatFloat"/>
    <Option name="native-file-header-format" value="HeaderWAV"/>
  </Config>
  <Metadata/>
  <Sources>
    <Source name="{os.path.basename(audio_file)}" type="audio" flags="" id="1001" level="0" channel="0" origin="" take-id=""/>
  </Sources>
  <Regions>
    <Region name="{os.path.basename(audio_file)}" muted="0" opaque="1" locked="0" video-locked="0" automatic="0" whole-file="1" import="0" external="1" sync-marked="0" left-of-split="0" right-of-split="0" hidden="0" position-locked="0" valid-transients="0" start="0" length="{audio_length_samples}" position="0" sync-position="0" ancestral-start="0" ancestral-length="0" stretch="1" shift="1" positional-lock-style="AudioTime" layering-index="0" envelope-active="0" default-fade-in="0" default-fade-out="0" fade-in-active="1" fade-out-active="1" scale-amplitude="1" id="2001" type="audio" first-edit="nothing" source-0="1001" master-source-0="1001" channels="{self.original_channels}"/>
  </Regions>
  <Locations>
    <Location id="87" name="session" start="0" end="{audio_length_samples}" flags="IsSessionRange" locked="no" position-lock-style="AudioTime"/>
  </Locations>
  <Routes>
    <Route id="1" name="Audio 1" default-type="audio" strict-io="1" active="1" denormal-protection="0" meter-point="MeterPostFader" disk-io-point="DiskIOPreFader" meter-type="MeterPeak" audio-playlist="Audio 1.1" saved-meter-point="MeterPostFader" alignment-choice="Automatic" playback-channel-mode="AllChannels" capture-channel-mode="AllChannels" playback-channel-mask="0xffff" capture-channel-mask="0xffff" note-mode="Sustained" step-editing="0" input-active="1" monitoring="" record-safe="0" monitoring="0" solo-safe="0" solo-isolated="0" phase-invert="0" denormal-protection="0" soloed-by-upstream="0" soloed-by-downstream="0">
      <IO name="Audio 1" id="26" direction="Input" default-type="audio" user-latency="0">
        <Port type="audio" name="Audio 1/audio_in 1">
          <Connection other="system:capture_1"/>
        </Port>
      </IO>
      <IO name="Audio 1" id="27" direction="Output" default-type="audio" user-latency="0">
        <Port type="audio" name="Audio 1/audio_out 1">
          <Connection other="master/audio_in 1"/>
        </Port>
      </IO>
      <Controllable name="solo" id="19" flags="" value="0"/>
      <Controllable name="mute" id="21" flags="" value="0"/>
      <MuteMaster mute-point="PreFader,PostFader,Listen,Main" muted="0"/>
      <Diskstream flags="" playlist="Audio 1.1" name="Audio 1" id="18" speed="1" capture-alignment="Automatic" channels="{self.original_channels}"/>
    </Route>
  </Routes>
</Session>'''
        
        with open(os.path.join(session_dir, f"{project_name}.ardour"), 'w') as f:
            f.write(session_xml)
        
        # Copy audio file
        shutil.copy2(audio_file, session_dir)
    
    def _create_bitwig_project(self, notes: List[Dict], audio_file: str, tempo: float, output_dir: str, project_name: str):
        """Create Bitwig Studio project"""
        project_file = os.path.join(output_dir, f"{project_name}.bwproject")
        
        # Calculate audio length
        if notes:
            audio_length = max(note['onset_time'] + note.get('duration', 0.5) for note in notes)
        else:
            audio_length = 60
        
        # Basic Bitwig project XML
        project_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<bitwig-project version="3.3.11">
  <application version="3.3.11" />
  <project>
    <transport>
      <tempo>{tempo}</tempo>
      <timeSignature numerator="4" denominator="4" />
    </transport>
    <tracks>
      <track type="audio" name="Audio Track">
        <clips>
          <clip name="{os.path.basename(audio_file)}" start="0.0" length="{audio_length}">
            <source file="{audio_file}" />
          </clip>
        </clips>
      </track>
    </tracks>
  </project>
</bitwig-project>'''
        
        with open(project_file, 'w') as f:
            f.write(project_xml)
    
    def _create_reaper_project(self, notes: List[Dict], audio_file: str, tempo: float, output_dir: str, project_name: str):
        """Create Reaper project"""
        project_file = os.path.join(output_dir, f"{project_name}.rpp")
        
        # Calculate audio length
        if notes:
            audio_length = max(note['onset_time'] + note.get('duration', 0.5) for note in notes)
        else:
            audio_length = 60
        
        # Basic Reaper project
        project_content = f'''<REAPER_PROJECT 0.1 "6.0/linux64" 1234567890
  RIPPLE 0
  GROUPOVERRIDE 0 0 0
  AUTOXFADE 1
  ENVATTACH 1
  POOLEDENVATTACH 0
  MIXERUIFLAGS 11 48
  PEAKGAIN 1
  FEEDBACK 0
  PANLAW 1
  PROJOFFS 0 0 0
  MAXPROJLEN 0 600
  GRID 3199 8 1 8 1 0 0 0
  TIMEMODE 1 5 -1 30 0 0 -1
  VIDEO_CONFIG 0 0 256
  PANMODE 3
  CURSOR 0
  ZOOM 100 0 0
  VZOOMEX 6 0
  USE_REC_CFG 0
  RECMODE 1
  SMPTESYNC 0 30 100 40 1000 300 0 0 1 0 0
  LOOP 0
  LOOPGRAN 0 4
  RECORD_PATH "" ""
  <RECORD_CFG
  >
  <APPLYFX_CFG
  >
  RENDER_FILE ""
  RENDER_PATTERN ""
  RENDER_FMT 0 2 0
  RENDER_1X 0
  RENDER_RANGE 1 0 0 18 1000
  RENDER_RESAMPLE 3 0 1
  RENDER_ADDTOPROJ 0
  RENDER_STEMS 0
  RENDER_DITHER 0
  TIMELOCKMODE 1
  TEMPOENVLOCKMODE 1
  ITEMMIX 0
  DEFPITCHMODE 589824 0
  TAKELANE 1
  SAMPLERATE {self.sample_rate} 0 0
  <RENDER_CFG
  >
  <METRONOME 6 2
    VOL 0.25 0.125
    FREQ 800 1600 1
    BEATLEN 4
    SAMPLES "" ""
    PATTERN 2863311530 2863311529
  >
  <GLOBAL_AUTOMATION_OVERRIDE
  >
  <MASTERHW 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 256 0 2 0 1 3 3
  >
  <TRACK {{"11223344-5566-7788-9900-AABBCCDDEEFF"}}
    NAME "Audio Track"
    PEAKCOL 16576
    BEAT -1
    AUTOMODE 0
    VOLPAN 1 0 -1 -1 1
    MUTESOLO 0 0 0
    IPHASE 0
    PLAYOFFS 0 1
    ISBUS 0 0
    BUSCOMP 0 0 0 0 0
    SHOWINMIX 1 0.6667 0.5 1 0.5 0 0 0
    FREEMODE 0
    SEL 0
    REC 0 0 1 0 0 0 0
    VU 2
    TRACKHEIGHT 0 0 0 0 0 0
    INQ 0 0 0 0.5 100 0 0 100
    NCHAN {self.original_channels}
    FX 1
    TRACKID {{"11223344-5566-7788-9900-AABBCCDDEEFF"}}
    PERF 0
    MIDIOUT -1
    MAINSEND 1 0
    <ITEM
      POSITION 0
      SNAPOFFS 0
      LENGTH {audio_length}
      LOOP 1
      ALLTAKES 0
      FADEIN 1 0 0 1 0 0 0
      FADEOUT 1 0 0 1 0 0 0
      MUTE 0 0
      SEL 0
      IGUID {{"22334455-6677-8899-AABB-CCDDEEFF0011"}}
      IID 1
      NAME "{os.path.basename(audio_file)}"
      VOLPAN 1 0 1 -1
      SOFFS 0
      PLAYRATE 1 1 0 -1 0 0.0025
      CHANMODE 0
      GUID {{"33445566-7788-99AA-BBCC-DDEEFF001122"}}
      <SOURCE WAVE
        FILE "{audio_file}"
      >
    >
  >
>'''
        
        with open(project_file, 'w') as f:
            f.write(project_content)
    
    def _create_ableton_project(self, notes: List[Dict], audio_file: str, tempo: float, output_dir: str, project_name: str):
        """Create Ableton Live project"""
        project_file = os.path.join(output_dir, f"{project_name}.als")
        
        # Calculate audio length
        if notes:
            audio_length = max(note['onset_time'] + note.get('duration', 0.5) for note in notes)
        else:
            audio_length = 60
        
        # Basic Ableton Live project XML (will be gzipped)
        project_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="10.0_377" SchemaChangeCount="3" Creator="Ableton Live 11.1.1" Revision="c3fce9bb43c3d4aae5e2adb31e04160ca9e326c3">
    <LiveSet>
        <MasterTrack Id="0">
            <n>
                <EffectiveName Value="Master"/>
            </n>
            <Tempo>
                <Manual Value="{tempo}"/>
            </Tempo>
        </MasterTrack>
        <Tracks>
            <AudioTrack Id="1">
                <n>
                    <EffectiveName Value="Audio"/>
                </n>
                <DeviceChain>
                    <MainSequencer>
                        <ClipSlotList>
                            <ClipSlot Id="0">
                                <ClipSlot>
                                    <Value>
                                        <AudioClip Id="0">
                                            <Name Value="{os.path.basename(audio_file)}"/>
                                            <AudioFile>
                                                <Name Value="{os.path.basename(audio_file)}"/>
                                                <FileName Value="{audio_file}"/>
                                                <Length Value="{audio_length}"/>
                                                <SampleRate Value="{self.sample_rate}"/>
                                                <Channels Value="{self.original_channels}"/>
                                            </AudioFile>
                                            <Loop>
                                                <LoopStart Value="0"/>
                                                <LoopEnd Value="{audio_length}"/>
                                                <StartRelative Value="0"/>
                                                <LoopOn Value="false"/>
                                                <OutMarker Value="{audio_length}"/>
                                            </Loop>
                                        </AudioClip>
                                    </Value>
                                </ClipSlot>
                            </ClipSlot>
                        </ClipSlotList>
                    </MainSequencer>
                </DeviceChain>
            </AudioTrack>
        </Tracks>
    </LiveSet>
</Ableton>'''
        
        # Compress the XML as Ableton expects
        with gzip.open(project_file, 'wt', encoding='utf-8') as f:
            f.write(project_xml)
    
    def save_multiple_formats(self, notes: List[Dict], tempo: float, output_dir: str, filename: str, 
                            key_sig: Tuple[str, str], chords: List[Dict] = None, beats: np.ndarray = None,
                            timbre_features: Dict = None, rhythm_analysis: Dict = None):
        """Save transcription in multiple formats including new AI hearing features"""
        base_path = os.path.join(output_dir, filename)
        
        # MusicXML (original format)
        musicxml = self.create_musicxml(notes, tempo, beats if beats is not None else np.array([]))
        with open(f"{base_path}.musicxml", 'w', encoding='utf-8') as f:
            f.write(musicxml)
        
        # MIDI export
        if self.config.get('output_midi'):
            self._save_midi(notes, tempo, f"{base_path}.mid")
        
        # ABC notation
        if self.config.get('output_abc'):
            abc_content = self._create_abc_notation(notes, tempo, key_sig)
            with open(f"{base_path}.abc", 'w') as f:
                f.write(abc_content)
        
        # LilyPond
        if self.config.get('output_lilypond'):
            lily_content = self._create_lilypond(notes, tempo, key_sig)
            with open(f"{base_path}.ly", 'w') as f:
                f.write(lily_content)
        
        # CSV data export
        if self.config.get('output_csv'):
            self._save_csv(notes, chords, f"{base_path}.csv")
        
        # NEW: Save timbre features as JSON
        if self.config.get('timbre_features_json') and timbre_features:
            with open(f"{base_path}_timbre_features.json", 'w') as f:
                json.dump(timbre_features, f, indent=2)
        
        # NEW: Save detailed rhythm analysis
        if self.config.get('detailed_rhythm_analysis') and rhythm_analysis:
            with open(f"{base_path}_rhythm_analysis.json", 'w') as f:
                json.dump(rhythm_analysis, f, indent=2)
        
        # Enhanced JSON with all analysis data
        json_data = {
            'tempo': tempo,
            'key_signature': {'key': key_sig[0], 'mode': key_sig[1]},
            'notes': notes,
            'chords': chords or [],
            'total_duration': notes[-1]['onset_time'] if notes else 0,
            'analysis_metadata': {
                'sample_rate': self.sample_rate,
                'hop_length': self.hop_length,
                'frame_length': self.frame_length,
                'original_channels': self.original_channels,
                'processing_stats': self.processing_stats
            }
        }
        
        # Add new analysis data to JSON
        if timbre_features:
            json_data['timbre_features'] = timbre_features
        if rhythm_analysis:
            json_data['rhythm_analysis'] = rhythm_analysis
        
        # Ensure all data is JSON serializable
        json_data = self._ensure_json_serializable(json_data)
        
        with open(f"{base_path}.json", 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_midi(self, notes: List[Dict], tempo: float, output_path: str):
        """Save as MIDI file (requires python-midi or mido)"""
        try:
            import mido
            from mido import MidiFile, MidiTrack, Message
            
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Set tempo
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
            
            # Convert notes to MIDI
            for note in notes:
                if note['pitch'] != 'rest':
                    try:
                        midi_note = self.hz_to_midi(librosa.note_to_hz(note['pitch']))
                        velocity = 64
                        duration_ticks = int(note.get('duration', 0.5) * 480)  # Convert to ticks
                        
                        track.append(Message('note_on', channel=0, note=midi_note, velocity=velocity, time=0))
                        track.append(Message('note_off', channel=0, note=midi_note, velocity=velocity, time=duration_ticks))
                    except:
                        continue  # Skip invalid notes
            
            mid.save(output_path)
            if self.config.get('verbose'):
                print(f"MIDI saved to: {output_path}")
        except ImportError:
            if self.config.get('verbose'):
                print("MIDI export requires 'mido' package: pip install mido")
    
    def _create_abc_notation(self, notes: List[Dict], tempo: float, key_sig: Tuple[str, str]) -> str:
        """Create ABC notation"""
        abc_content = f"""X:1
T:Transcribed Audio
M:4/4
L:1/4
Q:{int(tempo)}
K:{key_sig[0]}{'' if key_sig[1] == 'major' else 'm'}
"""
        
        for note in notes:
            if note['pitch'] == 'rest':
                abc_content += "z"
            else:
                # Convert to ABC notation (simplified)
                note_name = note['pitch'][0].upper()
                if '#' in note['pitch']:
                    note_name += '^'
                elif 'b' in note['pitch']:
                    note_name += '_'
                
                try:
                    octave = int(note['pitch'][-1])
                    if octave >= 5:
                        note_name = note_name.lower()
                except:
                    pass
                
                abc_content += note_name
            
            # Add duration modifier based on note type
            note_type = note.get('type', 'quarter')
            if note_type == 'half':
                abc_content += "2"
            elif note_type == 'eighth':
                abc_content += "/2"
            elif note_type == '16th':
                abc_content += "/4"
            
            abc_content += " "
        
        return abc_content
    
    def _create_lilypond(self, notes: List[Dict], tempo: float, key_sig: Tuple[str, str]) -> str:
        """Create LilyPond notation"""
        lily_content = f"""\\version "2.20.0"

\\header {{
  title = "Transcribed Audio"
  tagline = ""
}}

\\score {{
  \\new Staff {{
    \\tempo 4 = {int(tempo)}
    \\key {key_sig[0].lower()} \\{key_sig[1]}
    \\time 4/4
    
"""
        
        for note in notes:
            if note['pitch'] == 'rest':
                note_type = note.get('type', 'quarter')
                lily_content += f"r{note_type[0] if note_type != 'quarter' else '4'} "
            else:
                # Convert to LilyPond notation
                note_name = note['pitch'][0].lower()
                if '#' in note['pitch']:
                    note_name += "is"
                elif 'b' in note['pitch']:
                    note_name += "es"
                
                try:
                    octave = int(note['pitch'][-1])
                    if octave >= 4:
                        note_name += "'" * (octave - 3)
                    elif octave < 3:
                        note_name += "," * (3 - octave)
                except:
                    pass
                
                note_type = note.get('type', 'quarter')
                duration = note_type[0] if note_type != 'quarter' else '4'
                lily_content += f"{note_name}{duration} "
        
        lily_content += """
  }
  \\layout { }
  \\midi { }
}"""
        
        return lily_content
    
    def _save_csv(self, notes: List[Dict], chords: List[Dict], output_path: str):
        """Save analysis data as CSV"""
        with open(output_path, 'w', newline='') as csvfile:
            # Notes CSV
            notes_writer = csv.writer(csvfile)
            notes_writer.writerow(['onset_time', 'pitch', 'duration', 'type', 'confidence'])
            
            for note in notes:
                notes_writer.writerow([
                    note['onset_time'],
                    note['pitch'],
                    note.get('duration', 0.5),
                    note.get('type', 'quarter'),
                    note.get('confidence', 0.8)
                ])
            
            # Chords CSV (if available)
            if chords:
                csvfile.write('\n\nChords:\n')
                chord_writer = csv.writer(csvfile)
                chord_writer.writerow(['time', 'chord', 'confidence'])
                for chord in chords:
                    chord_writer.writerow([chord['time'], chord['chord'], chord['confidence']])
    
    def generate_analysis_report(self, notes: List[Dict], tempo: float, key_sig: Tuple[str, str], 
                               chords: List[Dict], audio_file: str, output_path: str,
                               timbre_features: Dict = None, rhythm_analysis: Dict = None):
        """Generate detailed analysis report including new AI hearing features"""
        report = f"""
AUDIO ANALYSIS REPORT
=====================

File: {audio_file}
Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC INFORMATION
-----------------
Duration: {self.processing_stats.get('duration', 0):.2f} seconds
Sample Rate: {self.sample_rate} Hz
Channels: {self.original_channels} ({'Stereo' if self.original_channels == 2 else 'Mono'})
Tempo: {tempo:.1f} BPM
Key: {key_sig[0]} {key_sig[1]}
Total Notes: {len(notes)}

PROCESSING STATISTICS
--------------------
Load Time: {self.processing_stats.get('load_time', 0):.3f} seconds
Original Sample Rate: {self.processing_stats.get('original_sr', 0)} Hz
Resampling: {'Yes' if self.processing_stats.get('original_sr') != self.sample_rate else 'No'}

NOTE DISTRIBUTION
-----------------
"""
        
        # Note statistics
        pitch_counts = {}
        duration_counts = {}
        
        for note in notes:
            pitch = note['pitch']
            duration = note.get('type', 'quarter')
            pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
        
        report += "Most common pitches:\n"
        for pitch, count in sorted(pitch_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"  {pitch}: {count} times\n"
        
        report += "\nNote durations:\n"
        for duration, count in sorted(duration_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"  {duration}: {count} notes\n"
        
        # NEW: Timbre analysis section
        if timbre_features:
            report += "\nTIMBRE ANALYSIS\n"
            report += "---------------\n"
            
            # Safely get numeric values, handling both single values and lists
            def get_numeric_value(features_dict, key, default=0):
                value = features_dict.get(key, default)
                if isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], list):
                            # Multi-dimensional - take mean of first dimension
                            return float(np.mean(value[0]))
                        else:
                            # Single dimensional - take mean
                            return float(np.mean(value))
                    else:
                        return float(default)
                return float(value)
            
            centroid_mean = get_numeric_value(timbre_features, 'spectral_centroid_mean')
            bandwidth_mean = get_numeric_value(timbre_features, 'spectral_bandwidth_mean')
            zcr_mean = get_numeric_value(timbre_features, 'zero_crossing_rate_mean')
            flatness_mean = get_numeric_value(timbre_features, 'spectral_flatness_mean')
            
            report += f"Average Spectral Centroid: {centroid_mean:.2f} Hz\n"
            report += f"Average Spectral Bandwidth: {bandwidth_mean:.2f} Hz\n"
            report += f"Average Zero Crossing Rate: {zcr_mean:.4f}\n"
            report += f"Spectral Flatness: {flatness_mean:.4f}\n"
            
            # Describe brightness and roughness
            if centroid_mean > 3000:
                report += "Brightness: High (bright, airy sound)\n"
            elif centroid_mean > 1500:
                report += "Brightness: Medium (balanced sound)\n"
            else:
                report += "Brightness: Low (dark, warm sound)\n"
            
            if zcr_mean > 0.1:
                report += "Texture: High energy, noisy content\n"
            elif zcr_mean > 0.05:
                report += "Texture: Medium energy, mixed content\n"
            else:
                report += "Texture: Low energy, smooth content\n"
        
        # NEW: Rhythm analysis section
        if rhythm_analysis:
            report += "\nRHYTHM ANALYSIS\n"
            report += "---------------\n"
            report += f"Tempo Stability: {rhythm_analysis.get('tempo_stability', 0):.3f} (higher = more stable)\n"
            report += f"Beat Consistency: {rhythm_analysis.get('beat_consistency', 0):.3f} (lower = more consistent)\n"
            report += f"Onset Density: {rhythm_analysis.get('onset_density', 0):.2f} onsets/second\n"
            report += f"Rhythmic Complexity: {rhythm_analysis.get('rhythmic_complexity', 0):.3f}\n"
            
            # Describe rhythm characteristics
            onset_density = rhythm_analysis.get('onset_density', 0)
            if onset_density > 5:
                report += "Rhythm Character: Very dense, many notes\n"
            elif onset_density > 2:
                report += "Rhythm Character: Moderately dense\n"
            else:
                report += "Rhythm Character: Sparse, few notes\n"
            
            complexity = rhythm_analysis.get('rhythmic_complexity', 0)
            if complexity > 0.5:
                report += "Timing: Irregular, complex rhythms\n"
            elif complexity > 0.2:
                report += "Timing: Moderately complex rhythms\n"
            else:
                report += "Timing: Regular, simple rhythms\n"
        
        # Chord analysis
        if chords:
            chord_counts = {}
            for chord in chords:
                chord_name = chord['chord']
                chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1
            
            report += "\nCHORD PROGRESSION\n"
            report += "-----------------\n"
            report += "Most common chords:\n"
            for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report += f"  {chord}: {count} times\n"
        
        # Processing recommendations
        report += "\nPROCESSING INSIGHTS\n"
        report += "-------------------\n"
        
        if timbre_features:
            # Use the same safe getter function
            def get_numeric_value(features_dict, key, default=0):
                value = features_dict.get(key, default)
                if isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], list):
                            return float(np.mean(value[0]))
                        else:
                            return float(np.mean(value))
                    else:
                        return float(default)
                return float(value)
            
            centroid = get_numeric_value(timbre_features, 'spectral_centroid_mean')
            flatness = get_numeric_value(timbre_features, 'spectral_flatness_mean')
            
            if centroid > 4000:
                report += "• Audio has bright character - may benefit from low-pass filtering\n"
            elif centroid < 1000:
                report += "• Audio has dark character - may benefit from high-frequency enhancement\n"
            
            if flatness > 0.5:
                report += "• High spectral flatness indicates noise-like content\n"
            elif flatness < 0.1:
                report += "• Low spectral flatness indicates tonal content\n"
        
        if rhythm_analysis:
            stability = rhythm_analysis.get('tempo_stability', 0)
            if stability < 0.5:
                report += "• Tempo appears unstable - consider tempo correction\n"
            
            consistency = rhythm_analysis.get('beat_consistency', 0)
            if consistency > 0.3:
                report += "• Beat timing is inconsistent - may benefit from quantization\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
    
    def process_batch(self, input_dir: str, output_dir: str, file_pattern: str = "*", 
                     transforms: Dict = None, parallel: bool = False):
        """Process multiple files in batch"""
        supported_exts = ('.wav', '.flac', '.ogg', '.mp3', '.mp4', '.m4a')
        pattern_path = os.path.join(input_dir, file_pattern)
        files = [f for f in glob.glob(pattern_path) if f.lower().endswith(supported_exts)]
        
        if not files:
            print(f"No audio files found matching pattern: {pattern_path}")
            return
        
        print(f"Found {len(files)} files to process")
        os.makedirs(output_dir, exist_ok=True)
        
        if parallel and len(files) > 1:
            max_workers = min(mp.cpu_count(), len(files))
            print(f"Processing in parallel using {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                # Prepare arguments for static method to avoid state contamination
                for file_path in files:
                    args_tuple = (file_path, output_dir, transforms, self.config.copy(), self.sample_rate)
                    future = executor.submit(AudMixMod._process_single_file_static, args_tuple)
                    futures.append((file_path, future))
                
                for file_path, future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per file
                        print(f"✓ Completed: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"✗ Failed: {os.path.basename(file_path)} - {e}")
        else:
            for file_path in files:
                try:
                    self._process_single_file(file_path, output_dir, transforms)
                    print(f"✓ Completed: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"✗ Failed: {os.path.basename(file_path)} - {e}")
    
    def _process_single_file(self, file_path: str, output_dir: str, transforms: Dict = None):
        """Process a single file (for batch processing)"""
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        return self.process_audio_file(
            file_path=file_path,
            output_dir=output_dir,
            filename_prefix=filename,
            transforms=transforms
        )
    
    @staticmethod
    def _process_single_file_static(args_tuple):
        """Static method for multiprocessing to avoid state contamination"""
        file_path, output_dir, transforms, config, sample_rate = args_tuple
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create a fresh instance for this process
        processor = AudMixMod(sample_rate=sample_rate, config=config)
        return processor.process_audio_file(
            file_path=file_path,
            output_dir=output_dir,
            filename_prefix=filename,
            transforms=transforms
        )
    
    def create_preset(self, preset_name: str, transforms: Dict, output_path: str):
        """Save transformation preset"""
        preset_data = {
            'name': preset_name,
            'created': datetime.now().isoformat(),
            'transforms': transforms
        }
        
        presets_file = output_path
        if os.path.exists(presets_file):
            with open(presets_file, 'r') as f:
                presets = json.load(f)
        else:
            presets = {}
        
        presets[preset_name] = preset_data
        
        with open(presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        
        print(f"Preset '{preset_name}' saved to {presets_file}")
    
    def load_preset(self, preset_name: str, presets_file: str) -> Dict:
        """Load transformation preset"""
        with open(presets_file, 'r') as f:
            presets = json.load(f)
        
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        return presets[preset_name]['transforms']
    
    def watch_folder(self, watch_dir: str, output_dir: str, transforms: Dict = None):
        """Watch folder for new files and auto-process"""
        print(f"Watching folder: {watch_dir}")
        print(f"Output folder: {output_dir}")
        print("Press Ctrl+C to stop watching...")
        
        processed_files = set()
        
        try:
            while True:
                pattern_path = os.path.join(watch_dir, "*")
                current_files = set(glob.glob(pattern_path))
                new_files = current_files - processed_files
                
                for file_path in new_files:
                    if file_path.lower().endswith(('.wav', '.flac', '.ogg', '.mp3', '.mp4', '.m4a')):
                        try:
                            print(f"Processing new file: {os.path.basename(file_path)}")
                            self._process_single_file(file_path, output_dir, transforms)
                            processed_files.add(file_path)
                            print(f"✓ Completed: {os.path.basename(file_path)}")
                        except Exception as e:
                            print(f"✗ Failed: {os.path.basename(file_path)} - {e}")
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\nStopped watching folder")
    
    def _send_webhook_notification(self, input_file: str, output_file: str, webhook_url: str):
        """Send webhook notification when processing completes"""
        payload = {
            'input_file': input_file,
            'output_file': output_file,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if self.config.get('verbose'):
                print(f"Webhook notification sent: {response.status_code}")
        except Exception as e:
            if self.config.get('verbose'):
                print(f"Webhook notification failed: {e}")

class AIProvider:
    """Base class for AI providers"""
    def __init__(self, config: Dict):
        self.config = config
        self.verbose = config.get('verbose', False)
        self.conversation_id = None  # For stateful conversations
        
    def is_available(self) -> bool:
        """Check if this provider is available"""
        raise NotImplementedError
        
    def analyze_audio(self, prompt: str) -> str:
        """Analyze audio with AI"""
        raise NotImplementedError
        
    def _extract_conversation_id(self, response_data: Dict) -> str:
        """Extract conversation_id from response if present"""
        return response_data.get('conversation_id')
        
    def _add_conversation_id(self, request_data: Dict) -> Dict:
        """Add conversation_id to request if we have one"""
        if self.conversation_id:
            request_data['conversation_id'] = self.conversation_id
        return request_data

class AnthropicProvider(AIProvider):
    """Anthropic/Claude provider"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.model_name = config.get('model_name', 'claude-3-sonnet-20240229')
        self._initialize()
        
    def _initialize(self):
        if not anthropic:
            return
            
        api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        api_base = os.getenv('ANTHROPIC_API_BASE')
        
        if api_key:
            if api_base:
                self.client = anthropic.Anthropic(api_key=api_key, base_url=api_base)
                if self.verbose:
                    print(f"✓ Anthropic client initialized with custom base: {api_base}")
            else:
                self.client = anthropic.Anthropic(api_key=api_key)
                if self.verbose:
                    print("✓ Anthropic client initialized (official API)")
                    
    def is_available(self) -> bool:
        return self.client is not None
        
    def analyze_audio(self, prompt: str) -> str:
        if not self.is_available():
            return "Anthropic provider not available"
            
        try:
            # Prepare request with optional conversation_id
            request_data = {
                "model": self.model_name,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            request_data = self._add_conversation_id(request_data)
            
            response = self.client.messages.create(**request_data)
            
            # Extract conversation_id if present
            if hasattr(response, '__dict__'):
                self.conversation_id = self._extract_conversation_id(response.__dict__)
                
            return response.content[0].text
        except Exception as e:
            return f"Anthropic analysis error: {e}"

class OpenAIProvider(AIProvider):
    """OpenAI provider with Custom GPT support"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.model_name = config.get('model_name', 'gpt-4')
        self.is_custom_gpt = self._detect_custom_gpt(self.model_name)
        self._initialize()
        
    def _detect_custom_gpt(self, model_name: str) -> bool:
        """Detect if model is a Custom GPT based on naming pattern"""
        import re
        pattern = r'^gpt-4[a-zA-Z0-9-]*-gizmo-g-[a-zA-Z0-9]+'
        return bool(re.match(pattern, model_name))
        
    def _initialize(self):
        if not openai:
            return
            
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = os.getenv('OPENAI_API_BASE')
        
        if api_key:
            if api_base:
                self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
                if self.verbose:
                    print(f"✓ OpenAI client initialized with custom base: {api_base}")
            else:
                self.client = openai.OpenAI(api_key=api_key)
                if self.verbose:
                    print("✓ OpenAI client initialized (official API)")
                    
            if self.is_custom_gpt and self.verbose:
                print(f"✓ Detected Custom GPT: {self.model_name}")
                
    def is_available(self) -> bool:
        return self.client is not None
        
    def _initialize_custom_gpt(self) -> bool:
        """Initialize Custom GPT if needed"""
        if not self.is_custom_gpt:
            return True
            
        try:
            # Custom GPTs might need initialization - this is a placeholder
            # for any special initialization logic
            if self.verbose:
                print(f"Initializing Custom GPT: {self.model_name}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"Custom GPT initialization failed: {e}")
            return False
            
    def analyze_audio(self, prompt: str) -> str:
        if not self.is_available():
            return "OpenAI provider not available"
            
        try:
            # Initialize Custom GPT if needed
            if self.is_custom_gpt and not self._initialize_custom_gpt():
                return "Custom GPT initialization failed"
                
            # Prepare request with optional conversation_id
            request_data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            request_data = self._add_conversation_id(request_data)
            
            response = self.client.chat.completions.create(**request_data)
            
            # Extract conversation_id if present
            if hasattr(response, '__dict__'):
                self.conversation_id = self._extract_conversation_id(response.__dict__)
                
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI analysis error: {e}"

class LocalProvider(AIProvider):
    """Local LLM provider (llama.cpp compatible)"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.model_name = config.get('model_name', 'llama')
        self.api_base = os.getenv('OPENAI_API_BASE', 'http://localhost:8080')
        self._initialize()
        
    def _initialize(self):
        if not openai:
            return
            
        try:
            # Use OpenAI client with local endpoint
            self.client = openai.OpenAI(
                api_key="local",  # llama.cpp doesn't need real key
                base_url=self.api_base
            )
            if self.verbose:
                print(f"✓ Local LLM client initialized: {self.api_base}")
        except Exception as e:
            if self.verbose:
                print(f"Local LLM initialization failed: {e}")
                
    def is_available(self) -> bool:
        return self.client is not None
        
    def analyze_audio(self, prompt: str) -> str:
        if not self.is_available():
            return "Local LLM provider not available"
            
        try:
            request_data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            request_data = self._add_conversation_id(request_data)
            
            response = self.client.chat.completions.create(**request_data)
            
            # Extract conversation_id if present
            if hasattr(response, '__dict__'):
                self.conversation_id = self._extract_conversation_id(response.__dict__)
                
            return response.choices[0].message.content
        except Exception as e:
            return f"Local LLM analysis error: {e}"

class CustomProvider(AIProvider):
    """Custom provider with full configurability"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = None
        self.model_name = config.get('model_name', 'custom-model')
        self.provider_config = self._load_custom_config()
        self._initialize()
        
    def _load_custom_config(self) -> Dict:
        """Load custom provider configuration from env vars or files"""
        config = {}
        
        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith('CUSTOM_AI_'):
                config_key = key[10:].lower()  # Remove CUSTOM_AI_ prefix
                config[config_key] = value
                
        # Load from config file if specified
        config_file = os.getenv('CUSTOM_AI_CONFIG_FILE')
        if config_file and os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load custom config file: {e}")
                    
        return config
        
    def _initialize(self):
        api_key = self.provider_config.get('api_key') or os.getenv('CUSTOM_AI_API_KEY')
        api_base = self.provider_config.get('api_base') or os.getenv('CUSTOM_AI_API_BASE')
        
        if not api_key or not api_base:
            if self.verbose:
                print("Custom provider requires CUSTOM_AI_API_KEY and CUSTOM_AI_API_BASE")
            return
            
        try:
            # Use OpenAI-compatible client for custom endpoints
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
            if self.verbose:
                print(f"✓ Custom provider initialized: {api_base}")
        except Exception as e:
            if self.verbose:
                print(f"Custom provider initialization failed: {e}")
                
    def is_available(self) -> bool:
        return self.client is not None
        
    def analyze_audio(self, prompt: str) -> str:
        if not self.is_available():
            return "Custom provider not available"
            
        try:
            request_data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            
            # Add any custom request parameters
            for key, value in self.provider_config.items():
                if key.startswith('request_'):
                    param_name = key[8:]  # Remove request_ prefix
                    request_data[param_name] = value
                    
            request_data = self._add_conversation_id(request_data)
            
            response = self.client.chat.completions.create(**request_data)
            
            # Extract conversation_id if present
            if hasattr(response, '__dict__'):
                self.conversation_id = self._extract_conversation_id(response.__dict__)
                
            return response.choices[0].message.content
        except Exception as e:
            return f"Custom provider analysis error: {e}"

class AIProcessor:
    """Enhanced AI processor with flexible provider support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.verbose = config.get('verbose', False)
        self.provider = None
        self.ai_enabled = config.get('ai_enabled', False)
        
        if self.ai_enabled:
            self._initialize_provider()
    
    def _get_provider_type(self) -> str:
        """Determine which provider to use"""
        # Command line flag takes precedence
        if self.config.get('ai_model'):
            model_map = {
                'claude': 'anthropic',
                'openai': 'openai',
                'chatgpt': 'openai'
            }
            return model_map.get(self.config['ai_model'], self.config['ai_model'])
            
        # Environment variable
        provider = os.getenv('MODEL_PROVIDER', '').lower()
        if provider in ['chatgpt', 'openai']:
            return 'openai'
        elif provider in ['claude.ai', 'anthropic']:
            return 'anthropic'
        elif provider == 'local':
            return 'local'
        elif provider == 'custom':
            return 'custom'
            
        # Auto-detect based on available keys
        if os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY'):
            return 'anthropic'
        elif os.getenv('OPENAI_API_KEY'):
            return 'openai'
        elif os.getenv('CUSTOM_AI_API_KEY'):
            return 'custom'
            
        return 'anthropic'  # Default fallback
    
    def _initialize_provider(self):
        """Initialize the appropriate AI provider"""
        provider_type = self._get_provider_type()
        
        # Get model name from environment or config
        model_name = os.getenv('MODEL_NAME') or self.config.get('model_name')
        provider_config = self.config.copy()
        if model_name:
            provider_config['model_name'] = model_name
            
        try:
            if provider_type == 'anthropic':
                self.provider = AnthropicProvider(provider_config)
            elif provider_type == 'openai':
                self.provider = OpenAIProvider(provider_config)
            elif provider_type == 'local':
                self.provider = LocalProvider(provider_config)
            elif provider_type == 'custom':
                self.provider = CustomProvider(provider_config)
            else:
                if self.verbose:
                    print(f"Unknown provider type: {provider_type}")
                    
            if self.provider and self.verbose:
                provider_name = provider_config.get('model_name', 'default')
                print(f"✓ AI Provider initialized: {provider_type} ({provider_name})")
                
        except Exception as e:
            if self.verbose:
                print(f"AI provider initialization error: {e}")
    
    def is_available(self, provider: str = 'any') -> bool:
        """Check if AI services are available"""
        if not self.provider:
            return False
            
        if provider == 'any':
            return self.provider.is_available()
        else:
            # Check if current provider matches requested type
            provider_type = self._get_provider_type()
            if provider in ['claude', 'anthropic'] and provider_type == 'anthropic':
                return self.provider.is_available()
            elif provider in ['openai', 'chatgpt'] and provider_type == 'openai':
                return self.provider.is_available()
            elif provider == 'local' and provider_type == 'local':
                return self.provider.is_available()
            elif provider == 'custom' and provider_type == 'custom':
                return self.provider.is_available()
            else:
                return False
    
    def analyze_audio_with_ai(self, audio_analysis: Dict, prompt: str = None) -> str:
        """Get AI analysis of audio characteristics"""
        if not self.is_available():
            return "AI analysis not available (no providers configured or available)"
        
        default_prompt = f"""
        Analyze this audio data and provide insights about the musical content:
        
        Tempo: {audio_analysis.get('tempo', 'unknown')} BPM
        Key: {audio_analysis.get('key', 'unknown')}
        Duration: {audio_analysis.get('duration', 'unknown')} seconds
        Notes detected: {len(audio_analysis.get('notes', []))}
        
        Describe what you hear and suggest creative processing ideas.
        """
        
        analysis_prompt = prompt or default_prompt
        return self.provider.analyze_audio(analysis_prompt)

    def darunia_mode(self, audio_analysis: Dict, transformations_applied: Dict, ai_mode=False, ai_model=None, audio_bytes=None) -> None:
        """🎵 DARUNIA MODE: The Goron King grooves to your audio! 🎵"""
        import time
        import random
        import sys
        # Darunia's dancing phases
        dance_frames = [
            "\033[32m\n    🎵  ♪ ♫ ♪ ♫  🎵\n   ╔═══════════════╗\n🔥 ║  DARUNIA THE  ║ 🔥\n   ║  GORON KING   ║\n   ╚═══════════════╝\n     ╔═══════════╗\n     ║ (◔ ◡ ◔)   ║  Brother! This\n     ║  \\\\  o  //   ║  sound makes my\n     ║   \\\\ _ //    ║  soul DANCE!\n     ╚═══════════╝\n      /|\\   /|\\\n     🏔️     🏔️\n\033[0m",
            "\033[33m\n    🎶  ♪ ♫ ♪ ♫  🎶\n   ╔═══════════════╗\n🔥 ║  DARUNIA THE  ║ 🔥\n   ║  GORON KING   ║\n   ╚═══════════════╝\n     ╔═══════════╗\n     ║ (◕ ‿ ◕)   ║  The rhythm\n     ║    |o|     ║  flows through\n     ║   // \\\\\\\\    ║  my rocky heart!\n     ╚═══════════╝\n      🏔️\\\\   //🏔️\n     ~~\\\\     //~~\n\033[0m",
            "\033[35m\n    🎵  ♪ ♫ ♪ ♫  🎵\n   ╔═══════════════╗\n🔥 ║  DARUNIA THE  ║ 🔥\n   ║  GORON KING   ║\n   ╚═══════════════╝\n     ╔═══════════╗\n     ║ (★ ω ★)   ║  PURE BLISS,\n     ║   \\\\|o|//   ║  Brother! This\n     ║    \\\\o//    ║  is NIRVANA!\n     ╚═══════════╝\n        |   |\n     🏔️ ~~|~~🏔️\n\033[0m",
            "\033[36m\n    🎶  ♪ ♫ ♪ ♫  🎶\n   ╔═══════════════╗\n🔥 ║  DARUNIA THE  ║ 🔥\n   ║  GORON KING   ║\n   ╚═══════════════╝\n     ╔═══════════╗\n     ║ (◉ ◡ ◉)   ║  My socks have\n     ║    |o|     ║  ASCENDED to\n     ║   /| |\\\\    ║  Saturn! 🪐\n     ╚═══════════╝\n      //   \\\\\\\\\n     🏔️     🏔️\n\033[0m"
        ]
        proclamations = [
            "Brother! This groove has awakened my GORON SOUL!",
            "By the sacred stones! My very essence VIBRATES with joy!",
            "This sound... it's like liquid fire flowing through Death Mountain!",
            "Brother, you've created PURE AUDITORY BLISS!",
            "My rocky heart is MELTING with euphoria!",
            "This is better than a thousand rolling competitions!",
            "The spirits of the ancestors are headbanging in their caverns!"
        ]
        n_frames = len(dance_frames)
        print("\n" * (n_frames+2), end="")
        print(f"\033[{n_frames+2}A", end="")  # Move up to reserved region
        sys.stdout.flush()
        for i, frame in enumerate(dance_frames):
            print(f"\033[{n_frames+2}A", end="")  # Up to top of reserved
            print("\n" * i, end="")              # Down to current line
            print(frame)
            time.sleep(0.45)
        print(f"\033[{n_frames+2}B", end="")  # Down out of reserved lines
        sys.stdout.flush()
        if ai_mode and ai_model:
            darunia_prompt = (
                "You are Darunia, the Goron King from Zelda. Analyze the audio (stats below) and the audio file. "
                "Describe in passionate, over-the-top style what you 'hear.' Is this song or just sound/noise? Then, "
                "list 20+ personalized flag/option recommendations to improve the audio to your own Goron taste! "
                "Recommend only flags the program supports and be very creative."
            )
            input_data = {"stats": audio_analysis, "transforms": transformations_applied, "audio_bytes": audio_bytes}
            tirade = ai_model.ask(darunia_prompt, input_data)  # AI model must implement .ask()
            print(tirade)
        else:
            print(random.choice(proclamations) + "\n")
        print("🪨 Darunia's Groove Report 🪨")
        print("-" * 32)
        for key, value in audio_analysis.items():
            print(f"{key.capitalize()}: {value}")
        print("Applied transformations:", ", ".join(transformations_applied.keys()))
        print("="*50)
        print("🎵 Darunia mode complete. Return to normal workflow! 🎵\n")

def create_random_transforms() -> Dict:
    """Generate random transformation parameters for experimentation"""
    transforms = {}
    
    # Randomly select which effects to apply
    effects = [
        ('time_stretch', lambda: random.uniform(0.5, 2.0)),
        ('pitch_shift', lambda: random.randint(-12, 12)),
        ('harmonic_percussive_ratio', lambda: random.uniform(0.0, 1.0)),
        ('spectral_centroid_shift', lambda: random.uniform(-0.5, 0.5)),
        ('phase_randomize', lambda: random.uniform(0.0, 0.3)),
        ('stutter', lambda: random.uniform(0.05, 0.2)),
        ('dynamic_range_compress', lambda: random.uniform(0.3, 0.8)),
    ]
    
    # Apply 1-3 random effects
    num_effects = random.randint(1, 3)
    selected_effects = random.sample(effects, num_effects)
    
    for effect_name, value_generator in selected_effects:
        transforms[effect_name] = value_generator()
    
    return transforms

def get_ai_advice_pre(model, audio_path, user_prompt, proposed_flags):
    prompt = (
        "You are an audio AI advisor. Given this user purpose and proposed flag chain, do these flags make sense for this kind of audio? "
        "What would the output sound like if used? Be detailed, and recommend alternatives if needed."
    )
    with open(audio_path, "rb") as f:
        audio_bytes = f.read(80000)
    advice = model.ask(prompt, {"audio_bytes": audio_bytes, "user_prompt": user_prompt, "flags": proposed_flags})
    return advice

def get_ai_advice_post(model, audio_path, user_prompt, proposed_flags, analysis_data):
    prompt = (
        "You are an audio AI advisor. Here is an audio analysis and user purpose. Given the proposed flags/chain, do these make sense? "
        "Describe how the audio will change, suggest improvements, and rate the flag set."
    )
    with open(audio_path, "rb") as f:
        audio_bytes = f.read(80000)
    advice = model.ask(prompt, {"audio_bytes": audio_bytes, "user_prompt": user_prompt, "flags": proposed_flags, "analysis": analysis_data})
    return advice

def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description='audmixmod - The Ultimate Audio Analysis & Transformation Tool with AI Hearing Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s song.wav                                    # Basic transcription
  %(prog)s song.mp3 --pitch-shift 7 --time-stretch 0.8 # Chipmunk effect
  %(prog)s song.flac --output-all --analysis-report    # Full analysis with AI features
  %(prog)s --batch-dir /music --output-dir /output     # Batch processing
  %(prog)s song.wav --send-to-daw ableton              # Create Ableton project
  %(prog)s --watch-folder /dropbox/music /output       # Auto-process new files
  %(prog)s song.wav --random-transform                 # Random effects
  %(prog)s song.wav --timbre-features-json --generate-cqt --piano-roll-png  # AI hearing features
  %(prog)s song.wav --ai-enabled --ai-analyze          # AI analysis (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
  %(prog)s song.wav --darunia-mode                     # 🎵 PURE BLISS EXPERIENCE 🎵
  
  # AI Provider Examples:
  # export MODEL_PROVIDER="openai" MODEL_NAME="gpt-4-turbo"
  # export MODEL_PROVIDER="custom" CUSTOM_AI_API_BASE="https://my-llm.com/v1"
  %(prog)s song.wav --ai-enabled --ai-model custom --custom-ai-api-base "http://localhost:1234/v1"
  %(prog)s song.wav --ai-enabled --model-name "gpt-4o-gizmo-g-my-audio-gpt" --ai-analyze  # Custom GPT
  %(prog)s song.wav --ai-enabled --model-provider local --model-name "llama-3" --ai-analyze  # Local LLM
        '''
    )
    
    # Basic arguments
    parser.add_argument('input_file', nargs='?', help='Input audio file')
    parser.add_argument('-o', '--output-dir', help='Output directory for all generated files (default: current directory)')
    parser.add_argument('--filename-prefix', help='Prefix for output filenames')
    parser.add_argument('-sr', '--sample-rate', type=int, default=None, 
                       help='Sample rate for analysis (default: use native sample rate, no resampling)')
    
    # Batch processing
    parser.add_argument('--batch-dir', help='Process all audio files in directory')
    parser.add_argument('--file-pattern', default='*', help='File pattern for batch processing')
    parser.add_argument('--watch-folder', help='Watch folder for new files')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing for batch')
    
    # Configuration and presets
    parser.add_argument('--config-file', help='Load settings from JSON config file')
    parser.add_argument('--preset-file', help='Preset file path')
    parser.add_argument('--preset-name', help='Preset name to load/save')
    parser.add_argument('--save-preset', help='Save current settings as preset (without processing)')
    parser.add_argument('--random-transform', action='store_true', help='Apply random transformations')
    
    # Output formats
    parser.add_argument('--output-midi', action='store_true', help='Generate MIDI file')
    parser.add_argument('--output-abc', action='store_true', help='Generate ABC notation')
    parser.add_argument('--output-lilypond', action='store_true', help='Generate LilyPond notation')
    parser.add_argument('--output-csv', action='store_true', help='Generate CSV data')
    parser.add_argument('--output-all', action='store_true', help='Generate all output formats')
    
    # DAW integration
    parser.add_argument('--send-to-daw', choices=['ardour', 'bitwig', 'reaper', 'ableton'],
                       help='Create DAW project file')
    parser.add_argument('--daw-project-name', help='DAW project name')
    parser.add_argument('--daw-session-path', help='DAW session path')
    
    # Analysis options
    parser.add_argument('--key-detection', action='store_true', help='Enable key detection')
    parser.add_argument('--chord-analysis', action='store_true', help='Enable chord analysis')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence threshold for note detection')
    parser.add_argument('--pitch-detection-method', choices=['piptrack', 'crepe'], default='piptrack',
                       help='Pitch detection algorithm')
    parser.add_argument('--tempo-detection-method', choices=['standard', 'advanced'], default='standard',
                       help='Tempo detection method')
    parser.add_argument('--pitch-correction', action='store_true', 
                       help='Snap pitches to nearest semitones')
    parser.add_argument('--rhythm-quantize', choices=['none', 'smart'], default='none',
                       help='Rhythm quantization method')
    
    # NEW: AI Hearing Features
    parser.add_argument('--timbre-features-json', action='store_true',
                       help='Extract comprehensive timbre and texture features')
    parser.add_argument('--detailed-rhythm-analysis', action='store_true',
                       help='Perform detailed rhythm and timing analysis')
    parser.add_argument('--fft-spectrum-csv', action='store_true',
                       help='Export FFT spectrum data as CSV')
    parser.add_argument('--generate-cqt', action='store_true',
                       help='Generate Constant-Q Transform visualization')
    parser.add_argument('--save-cqt-data', action='store_true',
                       help='Save CQT data as numpy array')
    parser.add_argument('--generate-mel-spectrogram', action='store_true',
                       help='Generate Mel spectrogram visualization')
    parser.add_argument('--save-mel-data', action='store_true',
                       help='Save Mel spectrogram data as numpy array')
    parser.add_argument('--piano-roll-png', action='store_true',
                       help='Generate piano roll visualization')
    parser.add_argument('--audio-thumbnail', action='store_true',
                       help='Create audio thumbnail/preview clip')
    parser.add_argument('--thumbnail-duration', type=float, default=5.0,
                       help='Duration of audio thumbnail in seconds (default: 5.0)')
    
    # Preprocessing
    parser.add_argument('--denoise', action='store_true', help='Apply noise reduction')
    parser.add_argument('--normalize', action='store_true', help='Normalize audio levels')
    parser.add_argument('--trim-silence', action='store_true', help='Trim silence from ends')
    parser.add_argument('--fade-in', type=float, help='Fade in duration (seconds)')
    parser.add_argument('--fade-out', type=float, help='Fade out duration (seconds)')
    parser.add_argument('--reverse', action='store_true', help='Reverse the audio')
    
    # Basic transformations
    parser.add_argument('--time-stretch', type=float, default=None,
                       help='Time stretch factor (1.0=normal, >1=slower, <1=faster)')
    parser.add_argument('--pitch-shift', type=float, default=None,
                       help='Pitch shift in semitones (positive=up, negative=down)')
    
    # Harmonic/percussive separation
    parser.add_argument('--harmonic-only', action='store_true',
                       help='Extract only harmonic components')
    parser.add_argument('--percussive-only', action='store_true',
                       help='Extract only percussive components')
    parser.add_argument('--harmonic-percussive-ratio', type=float, default=None,
                       help='Harmonic/percussive balance (0.0=percussive only, 1.0=harmonic only)')
    
    # Spectral manipulations
    parser.add_argument('--spectral-centroid-shift', type=float, default=None,
                       help='Shift spectral centroid (brightness) by factor')
    parser.add_argument('--spectral-bandwidth-stretch', type=float, default=None,
                       help='Stretch spectral bandwidth by factor')
    parser.add_argument('--frequency-mask-low', type=float, default=None,
                       help='Low frequency cutoff in Hz')
    parser.add_argument('--frequency-mask-high', type=float, default=None,
                       help='High frequency cutoff in Hz')
    
    # Advanced spectral transformations
    parser.add_argument('--formant-shift', type=float, default=None,
                       help='Shift formants by semitones (vocal tract simulation)')
    parser.add_argument('--mel-scale-warp', type=float, default=None,
                       help='Warp mel-scale perception by factor')
    
    # Phase manipulations
    parser.add_argument('--phase-randomize', type=float, default=None,
                       help='Randomize phase (0.0-1.0, creates weird textures)')
    parser.add_argument('--phase-vocoder-stretch', type=float, default=None,
                       help='Phase vocoder time stretch (high quality)')
    
    # Temporal manipulations
    parser.add_argument('--tempo-stretch-independent', type=float, default=None,
                       help='Tempo-aware time stretching')
    parser.add_argument('--chroma-shift', type=int, default=None,
                       help='Shift chroma (pitch classes) by steps')
    
    # Creative effects
    parser.add_argument('--stutter', type=float, default=None,
                       help='Stutter effect duration in seconds')
    parser.add_argument('--granular-synthesis', action='store_true',
                       help='Apply granular synthesis effects')
    
    # Dynamic range
    parser.add_argument('--dynamic-range-compress', type=float, default=None,
                       help='Dynamic range compression ratio (0.0-1.0)')
    
    # Stereo effects (future)
    parser.add_argument('--stereo-width', type=float, default=None,
                       help='Stereo width adjustment')
    
    # Visualizations
    parser.add_argument('--generate-spectrogram', action='store_true',
                       help='Generate spectrogram image')
    parser.add_argument('--waveform-png', action='store_true',
                       help='Generate waveform visualization')
    parser.add_argument('--chromagram-image', action='store_true',
                       help='Generate chromagram visualization')
    
    # Analysis and reporting
    parser.add_argument('--analysis-report', action='store_true',
                       help='Generate detailed text analysis report')
    
    # Performance and debugging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without doing it')
    parser.add_argument('--benchmark', action='store_true',
                       help='Show detailed timing information')
    
    # Integration
    parser.add_argument('--webhook-notify', help='Webhook URL for completion notifications')
    parser.add_argument('--upload-to-cloud', help='Cloud storage URL for auto-upload')
    
    # AI Features 🤖
    parser.add_argument('--ai-enabled', action='store_true',
                       help='Enable AI-powered features (requires API keys in env vars)')
    parser.add_argument('--ai-analyze', action='store_true',
                       help='Get AI analysis of the audio content')
    parser.add_argument('--ai-prompt', 
                       help='Custom prompt for AI analysis')
    parser.add_argument('--ai-model', choices=['claude', 'openai', 'chatgpt', 'anthropic', 'local', 'custom'], 
                       default='claude', help='AI model provider (default: claude)')
    parser.add_argument('--model-name', dest='model_name',
                       help='Specific model name (e.g., gpt-4, claude-3-sonnet-20240229, or Custom GPT name)')
    parser.add_argument('--model-provider', dest='model_provider',
                       choices=['chatgpt', 'openai', 'claude.ai', 'anthropic', 'local', 'custom'],
                       help='Override MODEL_PROVIDER env var')
    parser.add_argument('--darunia-mode', action='store_true',
                       help='🎵 DARUNIA MODE: The Goron King celebrates your audio! 🎵')
    
    # Custom AI Provider Configuration
    parser.add_argument('--custom-ai-api-key', dest='custom_ai_api_key', help='Custom AI provider API key')
    parser.add_argument('--custom-ai-api-base', dest='custom_ai_api_base', help='Custom AI provider base URL')
    parser.add_argument('--custom-ai-config-file', dest='custom_ai_config_file', help='Custom AI provider config file (JSON)')
    
    # Environment Variables for AI:
    # MODEL_PROVIDER - Provider type: chatgpt, openai, claude.ai, anthropic, local, custom
    # MODEL_NAME - Specific model name (supports Custom GPTs: gpt-4*-gizmo-g-*)
    # ANTHROPIC_API_KEY or CLAUDE_API_KEY - Your Anthropic API key
    # ANTHROPIC_API_BASE - Custom API base URL (optional, defaults to official Anthropic API)
    # OPENAI_API_KEY - Your OpenAI API key  
    # OPENAI_API_BASE - Custom API base URL (optional, defaults to https://api.openai.com/v1)
    # CUSTOM_AI_API_KEY - Custom provider API key
    # CUSTOM_AI_API_BASE - Custom provider base URL
    # CUSTOM_AI_CONFIG_FILE - Path to custom provider config JSON file
    # CUSTOM_AI_* - Any env var starting with CUSTOM_AI_ becomes a config parameter
    
    args = parser.parse_args()
    
    # Load configuration file if specified
    config = {}
    if args.config_file:
        config = load_config_file(args.config_file)
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Handle custom AI provider command line overrides
    if args.model_provider:
        os.environ['MODEL_PROVIDER'] = args.model_provider
    if args.model_name:
        config['model_name'] = args.model_name
    if args.custom_ai_api_key:
        os.environ['CUSTOM_AI_API_KEY'] = args.custom_ai_api_key
    if args.custom_ai_api_base:
        os.environ['CUSTOM_AI_API_BASE'] = args.custom_ai_api_base
    if args.custom_ai_config_file:
        os.environ['CUSTOM_AI_CONFIG_FILE'] = args.custom_ai_config_file
    
    # Handle special modes
    if args.watch_folder:
        if not args.output_dir:
            print("Error: -o/--output-dir required for watch mode")
            sys.exit(1)
        
        transforms = collect_transforms(args)
        preprocess_config = collect_preprocessing(args)
        
        converter = AudMixMod(config=config)
        converter.watch_folder(args.watch_folder, args.output_dir, transforms)
        return
    
    if args.batch_dir:
        if not args.output_dir:
            print("Error: -o/--output-dir required for batch processing")
            sys.exit(1)
        
        transforms = collect_transforms(args)
        preprocess_config = collect_preprocessing(args)
        
        converter = AudMixMod(config=config)
        converter.process_batch(args.batch_dir, args.output_dir, args.file_pattern, 
                              transforms, args.parallel)
        return
    
    # Regular single file processing
    if not args.input_file:
        print("Error: Input file required (or use --batch-dir/--watch-folder)")
        parser.print_help()
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Check file extension
    supported_extensions = {'.wav', '.flac', '.ogg', '.mp3', '.mp4', '.m4a'}
    file_ext = os.path.splitext(args.input_file)[1].lower()
    if file_ext not in supported_extensions:
        print(f"Error: Unsupported file format '{file_ext}'")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        sys.exit(1)
    
    # Handle presets
    if args.preset_name and args.preset_file:
        if args.save_preset:
            # Save current settings as preset
            transforms = collect_transforms(args)
            preprocess_config = collect_preprocessing(args)
            # Merge transforms and preprocessing for preset saving
            all_settings = {**transforms, **preprocess_config}
            converter = AudMixMod(config=config)
            converter.create_preset(args.preset_name, all_settings, args.preset_file)
            return
        else:
            # Load preset
            converter = AudMixMod(config=config)
            try:
                preset_transforms = converter.load_preset(args.preset_name, args.preset_file)
                transforms = preset_transforms
                print(f"Loaded preset: {args.preset_name}")
            except Exception as e:
                print(f"Error loading preset: {e}")
                sys.exit(1)
    else:
        # Collect transformations from command line
        transforms = collect_transforms(args)
        if args.random_transform:
            random_transforms = create_random_transforms()
            transforms.update(random_transforms)
            print(f"Applied random transforms: {list(random_transforms.keys())}")
    
    preprocess_config = collect_preprocessing(args)
    
    # Set output-all flag effects
    if args.output_all:
        config.update({
            'output_midi': True,
            'output_abc': True,
            'output_lilypond': True,
            'output_csv': True,
            'generate_spectrogram': True,
            'waveform_png': True,
            'chromagram_image': True,
            'analysis_report': True,
            'chord_analysis': True,
            'key_detection': True,
            # NEW: Enable all AI hearing features
            'timbre_features_json': True,
            'detailed_rhythm_analysis': True,
            'fft_spectrum_csv': True,
            'generate_cqt': True,
            'generate_mel_spectrogram': True,
            'piano_roll_png': True,
            'audio_thumbnail': True,
            # AI features (if enabled)
            'ai_analyze': args.ai_enabled,
        })
    
    try:
        converter = AudMixMod(sample_rate=args.sample_rate, config=config)
        output_file = converter.process_audio_file(
            args.input_file,
            args.output_dir,
            args.filename_prefix,
            args.sample_rate,
            transforms,
            preprocess_config
        )
        
        if not args.dry_run:
            print(f"Conversion completed successfully!")
            print(f"Primary output: {output_file}")
            output_dir = args.output_dir or os.getcwd()
            print(f"All files saved to: {output_dir}")
            
            if config.get('output_all'):
                print("Generated all available output formats and AI hearing features")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def collect_transforms(args) -> Dict:
    """Collect transformation parameters from arguments"""
    return {
        'time_stretch': args.time_stretch,
        'pitch_shift': args.pitch_shift,
        'harmonic_only': args.harmonic_only,
        'percussive_only': args.percussive_only,
        'harmonic_percussive_ratio': args.harmonic_percussive_ratio,
        'spectral_centroid_shift': args.spectral_centroid_shift,
        'spectral_bandwidth_stretch': args.spectral_bandwidth_stretch,
        'frequency_mask_low': args.frequency_mask_low,
        'frequency_mask_high': args.frequency_mask_high,
        'formant_shift': args.formant_shift,
        'mel_scale_warp': args.mel_scale_warp,
        'phase_randomize': args.phase_randomize,
        'phase_vocoder_stretch': args.phase_vocoder_stretch,
        'tempo_stretch_independent': args.tempo_stretch_independent,
        'chroma_shift': args.chroma_shift,
        'dynamic_range_compress': args.dynamic_range_compress,
        'stereo_width': args.stereo_width,
        'stutter': args.stutter,
        'granular_synthesis': args.granular_synthesis,
    }

def collect_preprocessing(args) -> Dict:
    """Collect preprocessing parameters from arguments"""
    return {
        'denoise': args.denoise,
        'normalize': args.normalize,
        'trim_silence': args.trim_silence,
        'fade_in': args.fade_in,
        'fade_out': args.fade_out,
        'reverse': args.reverse,
    }

if __name__ == '__main__':
    main()
