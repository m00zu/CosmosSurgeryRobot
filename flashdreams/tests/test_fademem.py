# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks for FadeMem consolidation and bounded positional layout."""

import math

import pytest
import torch
from flashdreams.core.attention.fademem import FadeMemKVCache

pytestmark = pytest.mark.ci_cpu


def test_consolidation_matches_list_reference_across_layers():
    generator = torch.Generator().manual_seed(7)
    frames = torch.randn(2, 2, 80, 3, 4, generator=generator)
    seed = [
        (frames[layer, 0, :1].unsqueeze(0), frames[layer, 1, :1].unsqueeze(0))
        for layer in range(2)
    ]
    memory = FadeMemKVCache(seed, capacity=7, seq_dim=1)
    pointers = [tensor.data_ptr() for pair in memory.layers() for tensor in pair]
    reference = [(0, 1, frames[:, :, 0].clone())]
    for frame in range(1, 80):
        reference.append((frame, 1, frames[:, :, frame].clone()))
        if len(reference) > 7:
            # Independent list implementation of the released age-gap rule.
            age = [(frame + 1 - center) ** 0.3 for center, _, _ in reference]
            i = min(range(len(reference) - 2), key=lambda i: abs(age[i + 1] - age[i]))
            left, right = reference[i : i + 2]
            span = left[1] + right[1]
            center = round(left[1] / span * left[0] + right[1] / span * right[0])
            a, b = math.sqrt(left[1]), math.sqrt(right[1])
            merged = left[2] * (a / (a + b)) + right[2] * (b / (a + b))
            if left[0] == 0:
                center, merged = 0, left[2]
            reference[i : i + 2] = [(center, span, merged)]
        memory.append(
            frame,
            [
                (
                    frames[layer, 0, frame : frame + 1].unsqueeze(0),
                    frames[layer, 1, frame : frame + 1].unsqueeze(0),
                )
                for layer in range(2)
            ],
        )
        assert [(e.center_frame, e.span) for e in memory.entries] == [
            (c, s) for c, s, _ in reference
        ]
        expected = torch.stack([entry[2] for entry in reference], dim=2)
        for layer, (key, value) in enumerate(memory.layers()):
            torch.testing.assert_close(key[0], expected[layer, 0])
            torch.testing.assert_close(value[0], expected[layer, 1])
        assert memory.entries[-1].center_frame == frame
        assert sum(e.span for e in memory.entries) == frame + 1
        positions, current = memory.attention_positions(frame + 1, 3)
        assert len(positions) == memory.count
        assert list(positions) == sorted(set(positions))
        assert max(positions) < current
        if frame >= 18:
            assert current == 18
            assert positions[-3:] == (15, 16, 17)
    assert pointers == [
        tensor.data_ptr() for pair in memory.layers() for tensor in pair
    ]
    memory.reset(seed)
    assert memory.count == 1 and memory.next_frame == 1
    for actual, original in zip(memory.layers(), seed, strict=True):
        for a, b in zip(actual, original, strict=True):
            torch.testing.assert_close(a, b)


def test_bf16_consolidation_rounds_like_out_of_place_merge_and_shift():
    # The merge runs in place and the shift copies frame by frame; both must
    # round exactly like ``left * w + right * w'`` on the storage dtype.
    generator = torch.Generator().manual_seed(3)
    frames = torch.randn(3, 2, 40, 5, 8, generator=generator).to(torch.bfloat16)
    seed = [(frames[layer, 0, :1][None], frames[layer, 1, :1][None]) for layer in range(3)]
    memory = FadeMemKVCache(seed, capacity=6, seq_dim=1, preserve_anchor=False)
    reference = [(0, 1, frames[:, :, 0].clone())]
    for frame in range(1, 40):
        reference.append((frame, 1, frames[:, :, frame].clone()))
        if len(reference) > 6:
            age = [(frame + 1 - center) ** 0.3 for center, _, _ in reference]
            i = min(range(len(reference) - 2), key=lambda i: abs(age[i + 1] - age[i]))
            left, right = reference[i : i + 2]
            span = left[1] + right[1]
            center = round(left[1] / span * left[0] + right[1] / span * right[0])
            a, b = math.sqrt(left[1]), math.sqrt(right[1])
            merged = left[2] * (a / (a + b)) + right[2] * (b / (a + b))
            reference[i : i + 2] = [(center, span, merged)]
        memory.append(
            frame,
            [
                (frames[layer, 0, frame : frame + 1][None], frames[layer, 1, frame : frame + 1][None])
                for layer in range(3)
            ],
        )
        expected = torch.stack([entry[2] for entry in reference], dim=2)
        for layer, (key, value) in enumerate(memory.layers()):
            assert torch.equal(key[0], expected[layer, 0])
            assert torch.equal(value[0], expected[layer, 1])
        # The insertion scratch entry is cleared after every merge.
        for key, value in memory._storage:
            assert not key[:, memory.count * 5 :].any()
            assert not value[:, memory.count * 5 :].any()


@pytest.mark.parametrize("capacity", [0, 1, True, 2.5])
def test_reject_invalid_capacity(capacity):
    x = torch.ones(1, 1, 2, 4)
    with pytest.raises(ValueError):
        FadeMemKVCache([(x, x)], capacity=capacity)


def test_invalid_append_and_reset_preserve_memory():
    x = torch.ones(1, 1, 2, 4)
    memory = FadeMemKVCache([(x, x), (x, x)], capacity=3)
    before = [(k.clone(), v.clone()) for k, v in memory.layers()]
    with pytest.raises(ValueError):
        memory.append(1, [(x * 2, x * 2), (x.double(), x.double())])
    with pytest.raises(ValueError):
        memory.reset([(x.repeat(1, 1, 2, 1), x.repeat(1, 1, 2, 1))] * 2)
    assert memory.next_frame == 1
    for original, actual in zip(before, memory.layers(), strict=True):
        for a, b in zip(original, actual, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("preserve_anchor", [True, False])
def test_large_budget_has_enough_unique_positions(preserve_anchor):
    x = torch.ones(1, 1, 1, 4)
    memory = FadeMemKVCache([(x, x)], capacity=30, preserve_anchor=preserve_anchor)
    for frame in range(1, 35):
        memory.append(frame, [(x * frame, x)])
    positions, current = memory.attention_positions(35, 3)
    assert len(set(positions)) == 30
    assert list(positions) == sorted(positions)
    assert positions[-1] < current


def test_four_frame_commits_match_pinned_upstream_snapshot():
    # Golden result from ContinuousFadeMem at 15b70ce9704d45ccbefd3a4a2991a966bfa4feb7,
    # with logging disabled, seven slots, beta=.3 and a shared chunk-end age.
    x = torch.zeros(1, 1, 1, 1)
    memory = FadeMemKVCache([(x, x)], capacity=7)
    for start in range(1, 81, 4):
        chunk = torch.arange(start, start + 4, dtype=torch.float32).reshape(1, 1, 4, 1)
        memory.append(start, [(chunk, chunk)])
    assert [(e.center_frame, e.span) for e in memory.entries] == [
        (0, 12),
        (23, 24),
        (46, 21),
        (63, 13),
        (74, 9),
        (79, 1),
        (80, 1),
    ]
    expected = torch.tensor(
        [
            0,
            24.470563888549805,
            46.838844299316406,
            63.780540466308594,
            74.55516815185547,
            79,
            80,
        ]
    )
    for tensor in memory.layers()[0]:
        torch.testing.assert_close(tensor.flatten(), expected)
    assert memory.attention_positions(81, 3) == ((0, 1, 2, 3, 15, 16, 17), 18)


def test_cosine_merges_adjacent_similar_descriptors_and_protects_newest():
    x = torch.zeros(1, 1, 1, 1)
    descriptor = torch.tensor([[-1.0, 0.0]])
    memory = FadeMemKVCache(
        [(x, x)], capacity=3, merge_selection="cosine", descriptors=descriptor
    )
    pointer = memory._descriptors.data_ptr()
    # The newest pair is identical, but excluded. The next most similar pair
    # differs from the oldest pair selected by the age policy.
    memory.append(
        1,
        [(torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1),) * 2],
        descriptors=torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]),
    )
    assert [(e.center_frame, e.span) for e in memory.entries] == [
        (0, 1),
        (2, 2),
        (3, 1),
    ]
    torch.testing.assert_close(
        memory.layers()[0][0].flatten(), torch.tensor([0.0, 1.5, 3.0])
    )
    torch.testing.assert_close(
        memory._descriptors[:3], torch.tensor([[-1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    )
    memory.append(4, [(x + 4, x + 4)], descriptors=torch.tensor([[1.0, 1.0]]))
    weight = math.sqrt(2) / (math.sqrt(2) + 1)
    assert [(e.center_frame, e.span) for e in memory.entries] == [
        (0, 1),
        (2, 3),
        (4, 1),
    ]
    torch.testing.assert_close(
        memory._descriptors[1], torch.tensor([1.0, 0.5 * weight + (1 - weight)])
    )
    positions, current = memory.attention_positions(5, 4)
    assert list(positions) == sorted(set(positions)) and max(positions) < current
    before = memory._descriptors.clone()
    for invalid in (None, torch.ones(2, 2), torch.tensor([[float("nan"), 0.0]])):
        with pytest.raises(ValueError):
            memory.append(5, [(x, x)], descriptors=invalid)
        assert memory.next_frame == 5 and torch.equal(memory._descriptors, before)
    memory.reset([(x, x)], descriptors=descriptor)
    assert memory.count == 1 and memory._descriptors.data_ptr() == pointer
    torch.testing.assert_close(memory._descriptors[0], descriptor[0])


def test_cosine_anchor_keeps_original_descriptor_and_zero_ties_are_finite():
    x = torch.ones(1, 1, 1, 1)
    descriptor = torch.zeros(1, 2)
    memory = FadeMemKVCache(
        [(x, x)], capacity=2, merge_selection="cosine", descriptors=descriptor
    )
    for start in range(1, 5):
        memory.append(start, [(x * start, x * start)], descriptors=descriptor)
    assert memory.entries[0].span == 4 and memory.entries[0].center_frame == 0
    assert torch.equal(memory._descriptors[0], descriptor[0])
    assert torch.equal(memory.layers()[0][0][:, :, :1], x)
    assert torch.isfinite(memory._descriptors).all()


@pytest.mark.parametrize(
    "options",
    [
        {"merge_selection": "bad"},
        {"merge_selection": "cosine"},
    ],
)
def test_reject_invalid_cosine_options(options):
    x = torch.ones(1, 1, 1, 1)
    with pytest.raises(ValueError):
        FadeMemKVCache([(x, x)], capacity=3, **options)


def test_cosine_selection_stream_runs_descriptor_work_without_changing_merges(
    monkeypatch,
):
    """Descriptor reads move to ``selection_stream``; slots and K/V stay identical."""
    generator = torch.Generator().manual_seed(3)
    frames = torch.randn(40, 1, 1, 2, 3, generator=generator)
    descriptors = torch.randn(40, 5, generator=generator)
    descriptors[17] = float("inf")
    sentinel, entered = object(), []

    class Recorder:
        def __init__(self, stream):
            assert stream is sentinel
            entered.append(stream)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def run(stream):
        memory = FadeMemKVCache(
            [(frames[0], frames[0])],
            capacity=6,
            merge_selection="cosine",
            descriptors=descriptors[:1],
        )
        errors = 0
        for start in range(1, 39, 2):
            chunk = frames[start : start + 2].transpose(0, 2).reshape(1, 1, 4, 3)
            try:
                memory.append(
                    start,
                    [(chunk, chunk * 2)],
                    descriptors=descriptors[start : start + 2],
                    selection_stream=stream,
                )
            except ValueError:
                # A non-finite chunk is rejected unchanged; replay it finite.
                errors += 1
                memory.append(
                    start,
                    [(chunk, chunk * 2)],
                    descriptors=descriptors[start : start + 2].nan_to_num(posinf=0.0),
                    selection_stream=stream,
                )
        return memory, errors

    expected, expected_errors = run(None)
    monkeypatch.setattr(torch.cuda, "stream", Recorder)
    actual, errors = run(sentinel)
    assert entered and errors == expected_errors == 1
    assert actual.entries == expected.entries
    assert torch.equal(actual._descriptors, expected._descriptors)
    for (k, v), (ek, ev) in zip(actual.layers(), expected.layers(), strict=True):
        assert torch.equal(k, ek) and torch.equal(v, ev)


def test_layers_with_different_shapes_match_separate_memories():
    """Mixed layer shapes skip the shared slab and still consolidate per layer."""
    generator = torch.Generator().manual_seed(5)
    wide = torch.randn(30, 1, 4, 2, 3, generator=generator)
    narrow = torch.randn(30, 1, 2, 2, 5, generator=generator)
    joint = FadeMemKVCache([(wide[0], wide[0]), (narrow[0], -narrow[0])], capacity=5)
    alone = [
        FadeMemKVCache([(wide[0], wide[0])], capacity=5),
        FadeMemKVCache([(narrow[0], -narrow[0])], capacity=5),
    ]
    for frame in range(1, 30):
        joint.append(
            frame, [(wide[frame], wide[frame]), (narrow[frame], -narrow[frame])]
        )
        alone[0].append(frame, [(wide[frame], wide[frame])])
        alone[1].append(frame, [(narrow[frame], -narrow[frame])])
    assert joint.entries == alone[0].entries
    for got, memory in zip(joint.layers(), alone, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(got, memory.layers()[0]))
