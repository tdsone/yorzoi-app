import numpy as np
import pytest

from app.core import encode as E
from app.core.errors import InvalidAlphabet, SequenceTooLong, SequenceTooShort


class TestNormalize:
    def test_uppercases(self):
        assert E.normalize("acgt") == "ACGT"

    def test_strips_all_whitespace(self):
        assert E.normalize("AC GT\n\tAC\r\nGT") == "ACGTACGT"

    def test_drops_fasta_headers(self):
        assert E.normalize(">chr1 desc\nACGT\n>chr2\nTTTT") == "ACGTTTTT"

    def test_preserves_n(self):
        assert E.normalize("acgtn") == "ACGTN"


class TestValidate:
    def test_accepts_acgtn(self):
        E.validate("ACGTN", min_len=1, max_len=10, field="s")

    @pytest.mark.parametrize("ch", ["R", "Y", "W", "U", "-", "1", "*"])
    def test_rejects_non_acgtn(self, ch):
        with pytest.raises(InvalidAlphabet) as exc:
            E.validate("ACG" + ch, min_len=1, max_len=10, field="s")
        assert exc.value.details["character"] == ch
        assert exc.value.details["position"] == 3

    def test_rejects_non_ascii_with_position(self):
        # A paste from a word processor carries curly quotes and non-breaking spaces.
        with pytest.raises(InvalidAlphabet) as exc:
            E.validate("ACG“T", min_len=1, max_len=10, field="s")
        assert exc.value.details["position"] == 3

    def test_length_bounds(self):
        with pytest.raises(SequenceTooShort):
            E.validate("AC", min_len=16, max_len=100, field="s")
        with pytest.raises(SequenceTooLong):
            E.validate("A" * 101, min_len=1, max_len=100, field="s")

    def test_reports_first_offender_not_last(self):
        with pytest.raises(InvalidAlphabet) as exc:
            E.validate("ARCGY", min_len=1, max_len=10, field="s")
        assert exc.value.details["position"] == 1


class TestOneHot:
    def test_shape_dtype_and_order(self):
        oh = E.dna_1hot("ACGT")
        assert oh.shape == (4, 4)
        assert oh.dtype == np.float32
        assert np.array_equal(oh, np.eye(4, dtype=np.float32))

    def test_n_is_an_all_zero_column(self):
        oh = E.dna_1hot("ACNGT")
        assert oh[:, 2].sum() == 0.0
        assert oh.sum() == 4.0

    def test_contiguous(self):
        assert E.dna_1hot("ACGT" * 10).flags["C_CONTIGUOUS"]


class TestPrepare:
    def test_validates_before_encoding(self):
        # The LUT maps anything unknown to an all-zero column, i.e. silently to N.
        # prepare() must reject rather than encode garbage.
        with pytest.raises(InvalidAlphabet):
            E.prepare("ACGRT", min_len=1, max_len=100, field="s")

    def test_returns_normalized_sequence(self):
        seq, oh = E.prepare(">h\nac gt\n", min_len=1, max_len=100, field="s")
        assert seq == "ACGT"
        assert oh.shape == (4, 4)


class TestDecode:
    @pytest.mark.parametrize("seq", ["ACGT", "ACGTN", "N", "AAAA", "ACGTACGTNNNN"])
    def test_roundtrip(self, seq):
        assert E.decode_1hot(E.dna_1hot(seq)) == seq

    def test_accepts_position_major_layout(self):
        # Stored fixtures are (L, 4); the model wants (4, L). Both must decode.
        oh = E.dna_1hot("ACGTACGT")
        assert E.decode_1hot(oh.T) == "ACGTACGT"
