# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for Objectives."""

from __future__ import absolute_import, division, print_function
import torch

def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(acts, h=None, w=None):
    shape = acts.shape
    h = shape[3] // 2 if h is None else h
    w = shape[4] // 2 if w is None else w
    return acts[: ,:, :, h:h+1, w:w+1]


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[:,batch:batch+1]
        else:
            return t
        
    return T2


def linconv_reshape(x,ref):
    #reshapes x to shape of y where x is shape (c) or (b,c) and y is either shape (b,c) or (b,c,h,w)
    if x.dim() == 1: x = x.unsqueeze(0)
    while x.dim() < ref.dim():
        x = x.unsqueeze(dim=-1)
    return x


def orthogonal_proj(x,direction):
    if isinstance(direction,int):
        v = torch.zeros(x.shape[1])
        v[direction] = 1.
    else:
        v = torch.tensor(direction)/torch.norm(direction)
    v = v.to(x.device)
    dot = (x * linconv_reshape(v,x)).sum(dim=1).unsqueeze(1)
    proj = dot * linconv_reshape(v,dot)
    orth_proj = x - linconv_reshape(proj,x)
    return orth_proj
    
    
