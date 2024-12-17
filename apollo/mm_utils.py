from PIL import Image
from io import BytesIO
import base64
import numpy as np
import os, math, cv2, re

import torch
from transformers import StoppingCriteria
from apollo.constants import *

import tempfile
from io import BytesIO
from decord import VideoReader, cpu
from num2words import num2words
import datetime


def read_video_cv2(video_path, all_indices):
    vidcap = cv2.VideoCapture(video_path)
    frames_dict = {}
    max_index = max(all_indices)  # Find the maximum index to avoid unnecessary reading
    count = 0
    success = True
    while success and count <= max_index:
        success, frame = vidcap.read()
        if success and count in all_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            frames_dict[count] = im_pil
        count += 1
    # Now retrieve frames according to all_indices, allowing duplicates
    images = [frames_dict[idx] for idx in all_indices if idx in frames_dict]
    return np.stack([np.array(img) for img in images])

def read_video_decord(video_file, all_indices):
    vr = VideoReader(video_file, num_threads=1, ctx=cpu(0))
    return vr.get_batch(all_indices).asnumpy()

def read_video_decord_eval(video_file, all_indices):
    vr = VideoReader(video_file)
    return vr.get_batch(all_indices).asnumpy()

def load_frames_from_video(video_file, all_indices, video_decode_backend="decord", eval_=False):
    video_ending = os.path.splitext(video_file)[1]
    if video_ending in ['.gif', '.webm'] or video_decode_backend=="opencv":
        buffer = read_video_cv2(video_file, all_indices)
    else:
        # Use decord for other video formats
        if eval_:
            buffer = read_video_decord_eval(video_file, all_indices)
        else:
            buffer = read_video_decord(video_file, all_indices)
    return buffer # (T, H, W, C)

def pad_to_center_square(frames, mean_values):
    """
    Pad the given frame or frames numpy array to square dimensions using the mean values as the padding color.
    Handles both single frames (H, W, C) and batches of frames (N, H, W, C).

    Args:
        frames (np.array): The input frame array of shape (H, W, C) or (N, H, W, C).
        mean_values (tuple): Mean values for each channel, typically derived from dataset normalization parameters.

    Returns:
        np.array: The padded frame array with square dimensions.
    """
    if frames.ndim == 3:  # Single frame
        frames = frames[np.newaxis, :]  # Add a batch dimension
    elif frames.ndim != 4:
        raise ValueError("Input array must be either of shape (H, W, C) or (N, H, W, C)")

    N, height, width, channels = frames.shape
    size = max(width, height)
    background_color = np.array(mean_values, dtype=frames.dtype)
    
    # Create a background array with the size and fill it with the mean values
    padded_frames = np.full((N, size, size, channels), background_color, dtype=frames.dtype)

    # Calculate padding offsets
    top, left = (size - height) // 2, (size - width) // 2

    # Place the original frames in the center of the square canvas
    padded_frames[:, top:top + height, left:left + width, :] = frames
    return padded_frames

def calculate_sample_indices(clip_duration, frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=1):
    sample_video_fps = frames_per_clip / clip_duration
    num_clips = math.ceil((video_duration / clip_duration) * clip_sampling_ratio)
    frame_step = original_fps / sample_video_fps
    partition_len = total_frames // num_clips
    all_indices, clip_indices, timestamps = [], [], []
    if frame_step > 0.5:
        frame_step = max(1, int(original_fps / sample_video_fps)) #was int/floor
        clip_len = int(frames_per_clip * frame_step) #was int/floor
        sample_len = min(clip_len, total_frames)
        clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
        for i in range(num_clips):
            if partition_len > clip_len:
                start_idx = (partition_len - clip_len) // 2
                end_idx = start_idx + clip_len
                indices = np.arange(start_idx, end_idx, frame_step)
                indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                indices = indices+ i * partition_len

            else:
                
                indices = np.arange(0, sample_len, frame_step)
                if len(indices) < frames_per_clip:
                    padding = np.full(frames_per_clip - len(indices), sample_len)
                    indices = np.concatenate((indices, padding))
                    
                indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

            # Calculate timestamps
            start_time = (indices[0] / original_fps)
            end_time = (indices[-1] / original_fps)
            timestamps.append((start_time, end_time))

    else:
        ## original video FPS too low, we need to sample the same frame multiple times. 
        ##  Generally should not happen.
        # Calculate the number of times each frame should be sampled
        num_sample = int(np.ceil(1 / frame_step))
    
        # Compute the effective clip length considering the frame step
        clip_len = int(frames_per_clip * frame_step)
    
        # Create an expanded list of indices with each frame repeated num_sample times
        indices = np.repeat(np.arange(clip_len), num_sample)

        # Ensure the clip length does not exceed the total number of frames
        clip_len = min(clip_len, len(indices))
        clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
        
        sample_len = min(clip_len, total_frames)
        if len(indices) < frames_per_clip:
            padding = np.full(frames_per_clip - len(indices), sample_len)
            indices = np.concatenate((indices, padding))
    
        # Distribute the indices into clips
        for i in range(num_clips):
            current_clip_indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
            current_clip_indices = current_clip_indices + i * clip_step

            # Append the current clip indices to the list of all clips
            clip_indices.append(current_clip_indices)
            all_indices.extend(current_clip_indices)

            # Calculate timestamps
            start_time = (current_clip_indices[0] / original_fps)
            end_time = (current_clip_indices[-1] / original_fps)
            timestamps.append((start_time, end_time))

    return clip_indices, all_indices, timestamps

def get_video_details(fname):
    """ Load video content using Decord """
    assert os.path.exists(fname), f'video path not found {fname}'
    _fsize = os.path.getsize(fname)
    assert _fsize >= 1 * 1024, f"video too short {fname}"
    vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))            
    # Get the total number of frames and the original fps of the video
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    video_duration = total_frames / original_fps
    return total_frames, original_fps, video_duration
    
def split_into_clips(video, frames_per_clip):
    """ Split video into a list of clips """
    fpc = frames_per_clip
    nc = len(video) // frames_per_clip
    return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

def process_video(vision_processors, frames_per_clip, buffer):
    mm_data=[]
    for vision_processor in vision_processors:
        centered_buffer = pad_to_center_square(buffer, tuple(int(x * 255) for x in vision_processor.image_mean))
        processed_clips = []
        for clip in split_into_clips(centered_buffer, frames_per_clip):
            clip = vision_processor.preprocess(clip, return_tensors='pt')['pixel_values']
            if type(clip) is list:
                assert len(clip)==1, "LazyVideoDataset: error, vision processor returned clip that is list of len>1 ."
                clip = clip[0]
            processed_clips.append(clip)
        mm_data.append(torch.stack(processed_clips))
    return mm_data

def load_video(video_file, vision_processors, clip_duration, frames_per_clip, clip_sampling_ratio=1, video_decode_backend='decord', eval_=False):
    total_frames, original_fps, video_duration = get_video_details(video_file)
    _, all_indices, timestamps = calculate_sample_indices(clip_duration, frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=clip_sampling_ratio)
    buffer = load_frames_from_video(video_file, all_indices, video_decode_backend, eval_)
    mm_data = process_video(vision_processors, frames_per_clip, buffer)
    return mm_data, timestamps

class ApolloMMLoader:
    def __init__(self, vision_processors, clip_duration, frames_per_clip, num_repeat_token, device, model_max_length = 32768, clip_sampling_ratio=1, video_decode_backend="decord"):
        self.vision_processors=vision_processors
        self.clip_duration=clip_duration
        self.device=device
        self.frames_per_clip=frames_per_clip
        self.num_repeat_token = num_repeat_token
        self.clip_sampling_ratio=clip_sampling_ratio
        self.model_max_length=model_max_length
        self.video_decode_backend=video_decode_backend
        self.vidprompt = lambda num_clips, video_duration : f"You are provided the following series of {num2words(num_clips)}, {self.clip_duration} second clips from a {datetime.timedelta(seconds=video_duration)} [H:MM:SS] video.\n"
    
    def load_video(self, video_file):
        total_frames, original_fps, video_duration = get_video_details(video_file)
        clip_sampling_ratio = min(1, (self.model_max_length * self.clip_sampling_ratio) / (video_duration  * self.num_repeat_token / self.clip_duration))
        
        _, all_indices, timestamps = calculate_sample_indices(self.clip_duration, self.frames_per_clip, total_frames, original_fps, video_duration, clip_sampling_ratio=clip_sampling_ratio)
        video, timestamps = load_video(video_file, self.vision_processors, self.clip_duration, self.frames_per_clip, clip_sampling_ratio=clip_sampling_ratio, eval_=True)
        
        num_clips =  len(video[0]) 
        num_tokens = num_clips * self.num_repeat_token
        video = [v.to(device=self.device, dtype=torch.bfloat16) for v in video]
        replace_string = self.vidprompt(num_clips, video_duration)

        temporal_prompt = [f"{round(clip[0], 1)}-{round(clip[1], 1)} seconds: {X_TOKEN['video'] * self.num_repeat_token}" for clip in timestamps]
        temporal_prompt = ',\n'.join(temporal_prompt)
        replace_string = replace_string + temporal_prompt
        
        return video, replace_string
    
    def load_image(self, image_file):
        print('implement image loading')
        return None


def tokenizer_mm_token(prompt, tokenizer, return_tensors=None):
    tokens_regex = re.compile('|'.join(re.escape(token) for token in X_TOKEN.values()))
    input_ids, last_pos, start_id = [], 0, 0
    for match in tokens_regex.finditer(prompt):
        if match.start() > last_pos:
            input_ids.extend(tokenizer(prompt[last_pos:match.start()]).input_ids)
        elif match.start() == 0:
            input_ids = tokenizer('').input_ids
            start_id = 1
        input_ids.append(X_TOKEN_INDEX)
        last_pos = match.end()
    if last_pos < len(prompt):
        input_ids.extend(tokenizer(prompt[last_pos:]).input_ids[start_id:])
    return torch.tensor(input_ids, dtype=torch.long) if return_tensors == 'pt' else input_ids

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
