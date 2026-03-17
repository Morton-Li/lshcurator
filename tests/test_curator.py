import sys
from pathlib import Path

import numpy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lshcurator.curator as curator_module
from lshcurator import Curator, CuratorConfig
from lshcurator.utils.types import BucketKeyChunk


def _make_curator(*, similarity_threshold: float | None = 0.9) -> Curator:
	return Curator(CuratorConfig(
		shingle_k=5,
		shingle_step=1,
		bands=2,
		rows_per_band=2,
		similarity_threshold=similarity_threshold,
	))


def _patch_compute_bucket_keys(
	monkeypatch: pytest.MonkeyPatch,
	bucket_keys: numpy.ndarray,
	mapping: dict[Path, list[BucketKeyChunk]],
) -> None:
	def _fake_compute_bucket_keys(self, corpus_files_path, corpus_field_name, corpus_file_format='jsonl', **kwargs):
		return bucket_keys, mapping

	monkeypatch.setattr(Curator, 'compute_bucket_keys', _fake_compute_bucket_keys)


def _patch_iter_corpus_texts(
	monkeypatch: pytest.MonkeyPatch,
	rows: list[tuple[str, Path]],
	*,
	calls: list[dict] | None = None,
) -> None:
	def _fake_iter_corpus_texts(corpus_files_path, corpus_field_name, corpus_file_format='jsonl', **kwargs):
		if calls is not None:
			calls.append({
				'corpus_files_path': corpus_files_path,
				'corpus_field_name': corpus_field_name,
				'corpus_file_format': corpus_file_format,
				'kwargs': dict(kwargs),
			})
		return_file_path = kwargs.pop('return_file_path', False)
		for text, file_path in rows:
			yield (text, file_path) if return_file_path else text

	monkeypatch.setattr(curator_module, 'iter_corpus_texts', _fake_iter_corpus_texts)


def _patch_init_deduper(monkeypatch: pytest.MonkeyPatch, results: list[bool]):
	state: dict[str, object] = {}

	class FakeDeduper:
		def __init__(self):
			self.calls: list[str] = []
			self._results = iter(results)

		def __call__(self, text: str) -> bool:
			self.calls.append(text)
			return next(self._results)

	def _fake_init_deduper(self, bucket_keys: numpy.ndarray):
		state['bucket_keys'] = bucket_keys.copy()
		self.deduper = FakeDeduper()
		state['deduper'] = self.deduper
		return self.deduper

	monkeypatch.setattr(Curator, 'init_deduper', _fake_init_deduper)
	return state


def test_init_deduper_flattens_row_band_keys(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	captured: dict[str, object] = {}

	class FakeDeduper:
		def __init__(self, bucket_keys: numpy.ndarray, config):
			captured['bucket_keys'] = bucket_keys.copy()
			captured['config'] = config

	monkeypatch.setattr(curator_module, 'Deduper', FakeDeduper)

	deduper = curator.init_deduper(numpy.array([[11, 12], [13, 14]], dtype=numpy.uint64))  # type: ignore[arg-type]

	assert deduper is curator.deduper
	assert numpy.array_equal(captured['bucket_keys'], numpy.array([11, 12, 13, 14], dtype=numpy.uint64))


def test_init_deduper_rejects_non_1d_or_2d_bucket_keys():
	curator = _make_curator()

	with pytest.raises(ValueError, match='Expected bucket_keys to be either a 1D array'):
		curator.init_deduper(numpy.zeros((1, 1, 1), dtype=numpy.uint64))  # type: ignore[arg-type]


def test_process_corpus_requires_similarity_threshold():
	curator = _make_curator(similarity_threshold=None)

	with pytest.raises(ValueError, match='similarity_threshold must be set'):
		next(curator.process_corpus('dummy.jsonl', 'text', corpus_file_format='jsonl'))


@pytest.mark.parametrize('bad_value', ['1', -1])
def test_process_corpus_validates_filter_low_freq_bucket_keys(bad_value):
	curator = _make_curator()

	with pytest.raises(ValueError, match='filter_low_freq_bucket_keys must be a non-negative integer'):
		next(curator.process_corpus('dummy.jsonl', 'text', corpus_file_format='jsonl', filter_low_freq_bucket_keys=bad_value))


def test_process_corpus_requires_row_bands_bucket_keys(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_path = Path('shape.jsonl')

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([1, 2, 3], dtype=numpy.uint64),
		mapping={file_path: [BucketKeyChunk(start_position=0, size=3)]},
	)

	with pytest.raises(ValueError, match='Expected bucket_keys to be a 2D array'):
		list(curator.process_corpus(file_path, 'text', corpus_file_format='jsonl'))


def test_process_corpus_returns_empty_iterator_when_bucket_key_computation_is_empty(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_path = Path('empty.jsonl')
	iter_calls: list[dict] = []

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.empty((0, curator.config.bands), dtype=numpy.uint64),
		mapping={file_path: []},
	)
	_patch_iter_corpus_texts(monkeypatch, [('unused', file_path)], calls=iter_calls)

	assert list(curator.process_corpus(file_path, 'text', corpus_file_format='jsonl')) == []
	assert iter_calls == []


def test_process_corpus_returns_empty_iterator_when_no_selected_bucket_keys(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_path = Path('unique.jsonl')
	iter_calls: list[dict] = []

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([[1, 2], [3, 4]], dtype=numpy.uint64),
		mapping={file_path: [BucketKeyChunk(start_position=0, size=2)]},
	)
	_patch_iter_corpus_texts(monkeypatch, [('alpha', file_path), ('beta', file_path)], calls=iter_calls)

	def _fail_init_deduper(self, bucket_keys: numpy.ndarray):
		raise AssertionError('Deduper should not be initialized when no selected bucket keys remain')

	monkeypatch.setattr(Curator, 'init_deduper', _fail_init_deduper)

	assert list(curator.process_corpus(file_path, 'text', corpus_file_format='jsonl')) == []
	assert iter_calls == []


def test_process_corpus_filter_zero_behaves_like_one(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_path = Path('threshold-zero.jsonl')

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([
			[11, 12],
			[11, 13],
			[21, 22],
		], dtype=numpy.uint64),
		mapping={file_path: [BucketKeyChunk(start_position=0, size=3)]},
	)
	_patch_iter_corpus_texts(monkeypatch, [('a', file_path), ('b', file_path), ('c', file_path)])
	state = _patch_init_deduper(monkeypatch, [True, False])

	result = list(curator.process_corpus(
		file_path,
		'text',
		corpus_file_format='jsonl',
		filter_low_freq_bucket_keys=0,
	))

	assert result == [('a', True), ('b', False), ('c', True)]
	assert numpy.array_equal(state['bucket_keys'], numpy.array([11], dtype=numpy.uint64))
	assert getattr(state['deduper'], 'calls') == ['a', 'b']


def test_process_corpus_initializes_deduper_with_selected_keys_only_and_routes_mixed_rows(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_a = Path('part-a.jsonl')
	file_b = Path('part-b.jsonl')

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([
			[101, 201],
			[301, 201],
			[401, 501],
			[601, 701],
		], dtype=numpy.uint64),
		mapping={
			file_a: [BucketKeyChunk(start_position=0, size=1), BucketKeyChunk(start_position=2, size=1)],
			file_b: [BucketKeyChunk(start_position=1, size=1), BucketKeyChunk(start_position=3, size=1)],
		},
	)
	_patch_iter_corpus_texts(monkeypatch, [
		('dedupe-a1', file_a),
		('dedupe-b1', file_b),
		('skip-a2', file_a),
		('skip-b2', file_b),
	])
	state = _patch_init_deduper(monkeypatch, [True, False])

	result = list(curator.process_corpus([file_a, file_b], 'text', corpus_file_format='jsonl'))

	assert result == [
		('dedupe-a1', True),
		('dedupe-b1', False),
		('skip-a2', True),
		('skip-b2', True),
	]
	assert numpy.array_equal(state['bucket_keys'], numpy.array([201], dtype=numpy.uint64))
	assert getattr(state['deduper'], 'calls') == ['dedupe-a1', 'dedupe-b1']


def test_process_corpus_keeps_files_with_no_selected_rows_out_of_deduper(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_a = Path('selected.jsonl')
	file_b = Path('unselected.jsonl')

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([
			[11, 12],
			[11, 13],
			[21, 22],
			[23, 24],
		], dtype=numpy.uint64),
		mapping={
			file_a: [BucketKeyChunk(start_position=0, size=2)],
			file_b: [BucketKeyChunk(start_position=2, size=2)],
		},
	)
	_patch_iter_corpus_texts(monkeypatch, [
		('dedupe-1', file_a),
		('dedupe-2', file_a),
		('keep-1', file_b),
		('keep-2', file_b),
	])
	state = _patch_init_deduper(monkeypatch, [True, False])

	result = list(curator.process_corpus([file_a, file_b], 'text', corpus_file_format='jsonl'))

	assert result == [
		('dedupe-1', True),
		('dedupe-2', False),
		('keep-1', True),
		('keep-2', True),
	]
	assert getattr(state['deduper'], 'calls') == ['dedupe-1', 'dedupe-2']


def test_process_corpus_uses_fast_path_when_all_rows_need_deduplication(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_path = Path('all-dedupe.jsonl')
	iter_calls: list[dict] = []

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([
			[11, 12],
			[11, 12],
		], dtype=numpy.uint64),
		mapping={file_path: [BucketKeyChunk(start_position=0, size=2)]},
	)
	_patch_iter_corpus_texts(monkeypatch, [('row-1', file_path), ('row-2', file_path)], calls=iter_calls)
	state = _patch_init_deduper(monkeypatch, [True, False])

	result = list(curator.process_corpus(file_path, 'text', corpus_file_format='jsonl'))

	assert result == [('row-1', True), ('row-2', False)]
	assert getattr(state['deduper'], 'calls') == ['row-1', 'row-2']
	assert len(iter_calls) == 1
	assert iter_calls[0]['kwargs'].get('return_file_path', False) is False


def test_process_corpus_raises_when_reader_under_reads_selected_file_rows(monkeypatch: pytest.MonkeyPatch):
	curator = _make_curator()
	file_a = Path('under-read-a.jsonl')
	file_b = Path('under-read-b.jsonl')

	_patch_compute_bucket_keys(
		monkeypatch,
		bucket_keys=numpy.array([
			[11, 12],
			[31, 41],
			[11, 51],
		], dtype=numpy.uint64),
		mapping={
			file_a: [BucketKeyChunk(start_position=0, size=2)],
			file_b: [BucketKeyChunk(start_position=2, size=1)],
		},
	)
	_patch_iter_corpus_texts(monkeypatch, [('row-a1', file_a), ('row-b1', file_b)])
	_patch_init_deduper(monkeypatch, [True, True])

	with pytest.raises(ValueError, match='out of sync'):
		list(curator.process_corpus([file_a, file_b], 'text', corpus_file_format='jsonl'))

