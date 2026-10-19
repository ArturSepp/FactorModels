# Source and dependency licensing

*Author: [Artur Sepp](https://github.com/ArturSepp)*

[TradableFactors](https://github.com/ArturSepp/tradablefactors) distributes its
first-party source under MIT, with original notices retained. See the
[software citation](../CITATION.cff), [LICENSE](../LICENSE), and
[supplemental third-party notices](https://github.com/ArturSepp/tradablefactors/blob/main/THIRD_PARTY_NOTICES.md).

## FactorLasso minimum

The base dependency is `factorlasso>=1.0.4.post1`. That
[published release](https://pypi.org/project/factorlasso/1.0.4.post1/) contains the
verified MIT licence and retained notices. The requirement excludes the earlier
GPL release series; those historical artifacts retain their original terms.
FactorLasso's estimation code and API are unchanged by that licence-only release.

## Reviewed Windows installation

The 19 October 2026 review covers a fresh base-wheel installation on Windows x64,
CPython 3.12: TradableFactors `0.1.0.dev3`, FactorLasso `1.0.4.post1`, QIS `5.38.1`,
and their recorded resolved dependencies. It excludes documentation/test tooling
and explicitly installed consumer packages. Open dependency ranges do not certify
every future version or every supported platform.

The inspected package graph uses permissive project licences, with the following
bundled-component qualifications and retained notices:

| Component in the reviewed wheels | Licence treatment |
|---|---|
| NumPy/SciPy GCC compiler runtimes | `GPL-3.0-or-later WITH GCC-exception-3.1`; retain the exception and its conditions, rather than describing the runtime as BSD or MIT. |
| Microsoft runtime DLLs bundled by numerical packages | Vendor redistributables under their Microsoft terms; these are not MIT-licensed OSS. |
| CUDA 11.2.2 half-precision headers shipped by Numba `0.68.0` | Separately licensed NVIDIA headers recorded in Numba's `LICENSES.third-party`. CPU factor-model use does not require a CUDA runtime or backend. |
| FreeType bundled by Pillow | The offered FreeType License alternative is retained; its alternative GPL wording does not make this choice a mandatory GPL dependency. |
| OpenBLAS in SCS `3.3.1` and SparseDiffPy `0.6.1` | BSD notices, with LAPACK notices where included. The exact identified binaries and complete supplemental texts ship in `THIRD_PARTY_NOTICES.md`. |

The [GCC runtime exception](https://www.gnu.org/licenses/gcc-exception-3.1.html)
grants additional permission under its stated conditions; it does not relicense
GCC. Microsoft separately documents the conditions for
[redistributing Visual C++ files](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170).
The reviewed installation is therefore not described as entirely MIT or entirely
permissively licensed across all bundled components.

The supplemental OpenBLAS/LAPACK notice accompanies the identified libraries
because the inspected upstream wheel licence directories omit those bundled
notices. It supplies attribution in TF's accompanying materials; it does not
modify the upstream wheels or certify redistribution of an entire environment.

## Explicit downstream composition

OptimalPortfolios is installed separately to consume prepared alphas and risk
inputs. Its current quadprog backend has its own GPL terms and remains outside
the TF base environment. An explicitly composed TF→OP installation must retain
that distinction. Vendor data, SDKs and private consumer environments also keep
their separate terms.
