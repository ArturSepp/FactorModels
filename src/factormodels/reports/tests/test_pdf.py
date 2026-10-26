"""Shared PDF lifecycle preserves caller state and completed artifacts on failure."""
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from factormodels.reports import _pdf


@pytest.mark.parametrize('failure', ['none', 'export', 'replace'])
def test_atomic_export_closes_only_owned_figures_and_restores_style(tmp_path, monkeypatch, failure):
    """A partial export/replacement cannot damage the prior file or leak figures/style."""
    monkeypatch.setenv('FACTORMODELS_OUTPUT_PATH', str(tmp_path))
    target = tmp_path / 'report.pdf'
    target.write_bytes(b'previous report')
    caller_figure = plt.figure()
    owned = plt.figure()
    seen = []

    def export(figures, name, *, local_path, orientation, add_current_date):
        """Inspect export arguments and simulate success or a partial write."""
        assert figures == [owned]
        assert orientation == 'portrait' and add_current_date is False
        assert plt.rcParams['pdf.fonttype'] == 42
        assert plt.rcParams['font.family'] == ['DejaVu Sans']
        assert plt.rcParams['text.usetex'] is False
        temporary = Path(local_path) / (name + '.pdf')
        assert temporary.parent == target.parent and temporary != target
        seen.append(temporary)
        temporary.write_bytes(b'completed report')
        if failure == 'export':
            raise OSError('export failure')

    def fail_replace(self, destination):
        """Simulate failure of the atomic replacement after PDF creation."""
        assert self == seen[0] and destination == target
        raise OSError('replace failure')

    monkeypatch.setattr(_pdf.qis, 'save_figs_to_pdf', export)
    if failure == 'replace':
        monkeypatch.setattr(Path, 'replace', fail_replace)
    try:
        with plt.rc_context({'pdf.fonttype': 3, 'font.family': 'serif', 'text.usetex': True}):
            if failure == 'none':
                assert _pdf.write_figures_to_pdf([owned], 'report.pdf') == target
            else:
                with pytest.raises(OSError, match=failure + ' failure'):
                    _pdf.write_figures_to_pdf([owned], 'report.pdf')
            assert plt.rcParams['pdf.fonttype'] == 3
            assert plt.rcParams['font.family'] == ['serif']
            assert plt.rcParams['text.usetex'] is True
        assert target.read_bytes() == (b'completed report' if failure == 'none' else b'previous report')
        assert list(tmp_path.iterdir()) == [target]
        assert not plt.fignum_exists(owned.number)
        assert plt.fignum_exists(caller_figure.number)
    finally:
        plt.close(owned)
        plt.close(caller_figure)


def test_invalid_extension_closes_owned_figures_without_creating_folders(tmp_path):
    """Direct writer users receive cleanup even when path validation fails."""
    figure = plt.figure()
    target = tmp_path / 'future/report.txt'
    with pytest.raises(ValueError, match='extension'):
        _pdf.write_figures_to_pdf([figure], target)
    assert not target.parent.exists()
    assert not plt.fignum_exists(figure.number)
