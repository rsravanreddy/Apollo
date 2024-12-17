# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    MISTRAL = auto()
    CHATML = auto()
    QWEN = auto()
    QWEN_2 = auto()
    GEMMA = auto()


conv_qwen = Conversation(
    system="""<|im_start|>system\n\nYou are a helpful vision-language assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_qwen_2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="qwen_2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.QWEN_2,
    sep=" ",
    sep2="<|endoftext|>",
)


default_conversation = conv_plain
conv_templates = {
    "qwen_1_5": conv_qwen,
    "qwen_2": conv_qwen
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
