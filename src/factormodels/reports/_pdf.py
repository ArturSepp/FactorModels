"""Write completed report figures atomically through QIS.

Purpose: Share PDF persistence and cleanup across FM report renderers.
Inputs: Completed Matplotlib figures and a caller-supplied output filename.
Outputs: Resolved PDF path after successful atomic replacement of the destination.
Units: Filesystem paths and page coordinates; no numerical analytics are computed.
Conditioning: Preserve existing output on failure, close supplied figures and remove temporary files.
Ownership: FM manages file lifecycle; QIS owns PDF export and local_path resolves caller settings.
"""
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import qis

from factormodels.utils.local_path import resolve_output_path

__all__ = ['write_figures_to_pdf']

PDF_RC = {'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
          'ps.fonttype': 42, 'text.usetex': False}


def write_figures_to_pdf(figures: Sequence[Figure], output_path: str | Path) -> Path:
    """Export and close completed figures, preserving any previous PDF on failure.

    Args:
        figures: Completed figures whose ownership passes to this writer.
        output_path: Absolute path or filename relative to the configured output root.

    Returns:
        Absolute destination path after successful replacement.

    Raises:
        ValueError: If the destination does not have a PDF extension.
        OSError: If directory creation, export or atomic replacement fails.
    """
    temporary: Path | None = None
    try:
        target = resolve_output_path(output_path)
        if target.suffix.lower() != '.pdf':
            raise ValueError('output_path must have the .pdf extension.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(PDF_RC):
            with NamedTemporaryFile(prefix='.fm-report-', suffix='.pdf',
                                    dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
            qis.save_figs_to_pdf(figures, temporary.stem, local_path=str(temporary.parent),
                                 orientation='portrait', add_current_date=False)
            temporary.replace(target)
    finally:
        for fig in figures:
            plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
