# Changelog

## [0.5.1](https://github.com/gseg-ethz/GSEGUtils/compare/v0.5.0...v0.5.1) (2026-06-23)


### 🐛 Bug Fixes

* **deps:** numpydantic ~=1.10 + register publish-testpypi.yml on main ([#27](https://github.com/gseg-ethz/GSEGUtils/issues/27)) ([3602cb5](https://github.com/gseg-ethz/GSEGUtils/commit/3602cb5f43c332f8c9f69e39068c8b5ce205b92f))


### 🤖 Continuous Integration

* **11-05:** promote App-token release-please.yml to main (REL-04) ([#30](https://github.com/gseg-ethz/GSEGUtils/issues/30)) ([6896382](https://github.com/gseg-ethz/GSEGUtils/commit/6896382cef437bd38288ac11f7510724042e1885))
* **11-05:** register publish-pypi.yml on main (prod PyPI gate cleared) ([#29](https://github.com/gseg-ethz/GSEGUtils/issues/29)) ([522746b](https://github.com/gseg-ethz/GSEGUtils/commit/522746b993982efba7d38decc5abdc50aefc28c8))

## [0.5.0](https://github.com/gseg-ethz/GSEGUtils/compare/v0.4.4...v0.5.0) (2026-06-12)


### ✨ Features

* **01-04:** promote private angle helpers to public + deprecation shims (D-16) ([495f8f7](https://github.com/gseg-ethz/GSEGUtils/commit/495f8f725527270ee7049becae146c28157a1024))
* **02-01:** swap pickle codec for np.save/load + JSON sidecar in DiskBackedStore ([4ec69d2](https://github.com/gseg-ethz/GSEGUtils/commit/4ec69d256e43d2388b1177b9bd66bd2a4a7759f3))
* **04-04:** stream _convert_to_memmap with chunked writes (PERF-04) ([235b4df](https://github.com/gseg-ethz/GSEGUtils/commit/235b4df3eb0c75fc366d454b7da15d4c78452c2e))
* **04-05:** double-checked locking for SingletonMeta.__call__ (GREEN, PERF-05) ([bc3a544](https://github.com/gseg-ethz/GSEGUtils/commit/bc3a54488f4f11392aa50423f0928ecd8a8795e8))
* **04-06a:** COUPLE-05 clip-and-saturate normalize_uint*/linear_map_dtype ([c5d7425](https://github.com/gseg-ethz/GSEGUtils/commit/c5d742563c37f849baf93681e020f743e2d90c3a))


### 🐛 Bug Fixes

* **01-ci:** unblock pre-commit lint job on CI ([4bfc919](https://github.com/gseg-ethz/GSEGUtils/commit/4bfc919d975069dc6e8fbdedd9f7ccb0b2570366))
* **02-04:** re-register weakref finalizer in LazyDiskCache __setstate__ + extend to .meta.json sidecar ([0d03e71](https://github.com/gseg-ethz/GSEGUtils/commit/0d03e7165011028e7acccff5f7a805ab8709c78b))
* **03-01:** drop _data attribute on offload instead of writing None (BUG-02) ([d4173e7](https://github.com/gseg-ethz/GSEGUtils/commit/d4173e787ace741362e537e45cf4c60b771b80fb))
* **03-01:** honour NDArrayOperatorsMixin in DiskBackedNDArray (BUG-01) ([96f7c3e](https://github.com/gseg-ethz/GSEGUtils/commit/96f7c3ec61ee8bade7156d226d549d87d48d2ef9))


### 📚 Documentation

* **01-02a:** add NumPy-style docstrings for base_arrays ([df728a3](https://github.com/gseg-ethz/GSEGUtils/commit/df728a326608718beaf573adaef43d33915f0d89))
* **01-02a:** add NumPy-style docstrings for base_types + generate_init_stubs ([dfa780a](https://github.com/gseg-ethz/GSEGUtils/commit/dfa780a2ceaa4c9d6c9494ca7f31cbec3c3acc65))
* **01-02a:** add NumPy-style docstrings for lazy_disk_cache subpackage ([c36e314](https://github.com/gseg-ethz/GSEGUtils/commit/c36e314d84bbae5a507b64b674ccbfffd648a907))
* **01-02a:** add NumPy-style docstrings for small modules + utility leaves ([a482697](https://github.com/gseg-ethz/GSEGUtils/commit/a482697f1c59a8a5d183ad663f80e414eee2c695))
* **01-02a:** fix remaining D-rule violations in util + validators ([f10de42](https://github.com/gseg-ethz/GSEGUtils/commit/f10de420e2fd20d602977b279cc174130d27b418))
* **01-04:** clarify BaseArray._coerce_array contract docstring (D-14) ([9ee480b](https://github.com/gseg-ethz/GSEGUtils/commit/9ee480bc46875483e541718d6b150b40536f1361))
* **01-04:** clarify validate_in_range contract docstring (D-14) ([96dc8ed](https://github.com/gseg-ethz/GSEGUtils/commit/96dc8ed07ced69f94c1e888e6fab6eb88689ecb4))
* **02-03:** remove Private classifier; document Publication Policy (SEC-03, GSEGUtils side) ([b6dc56c](https://github.com/gseg-ethz/GSEGUtils/commit/b6dc56c652905ea21e387edc11bbc3aadeef07ba))
* **07-02:** author MIGRATION-v1.0.md for GSEGUtils v1.0 milestone ([5e0e95a](https://github.com/gseg-ethz/GSEGUtils/commit/5e0e95af0375e55a641e326219e184812d9fe242))


### 🎨 Styles

* **01-02a:** ruff check --fix + ruff format src/ tests/ ([7404d1c](https://github.com/gseg-ethz/GSEGUtils/commit/7404d1cbf23f058052cec149356d6986625019d0))
* **01-03a:** rephrase ci.yml pyright comment to avoid literal continue-on-error duplication ([2c7ccc6](https://github.com/gseg-ethz/GSEGUtils/commit/2c7ccc6caddeadee15c4086e4fdca44959778b2f))
* **lint:** reformat 5 files for ruff 0.15.12 (CI green-up) ([29383bb](https://github.com/gseg-ethz/GSEGUtils/commit/29383bb6019b5ca12fda1d5c5679e99fc29ca435))


### 🧹 Miscellaneous Chores

* **01-02a:** resolve non-D ruff findings in src/ ([25efc69](https://github.com/gseg-ethz/GSEGUtils/commit/25efc69e14735c49f046cb338196af7b9312601c))
* **01-02a:** swap black/isort/flake8 for ruff (config + dev-dep + pre-commit) ([99fdc32](https://github.com/gseg-ethz/GSEGUtils/commit/99fdc328d08e9bd85ec44e654da913c2f4e6a5e2))
* **01-03a:** add pyright~=1.1 dev dep + expand tests/** ruff per-file-ignores (D-12) ([39ac873](https://github.com/gseg-ethz/GSEGUtils/commit/39ac87324592703ddcae6f53cdd288cfd424463b))
* **01-03a:** promote pyrightconfig.json to strict mode (D-12) ([f8c2572](https://github.com/gseg-ethz/GSEGUtils/commit/f8c257275bdb141bbb02502f4bdb9b023fddd033))
* **01-03a:** re-add mypy pre-commit hook + add pyright warn-only pre-commit + CI hooks (D-12) ([60c5d21](https://github.com/gseg-ethz/GSEGUtils/commit/60c5d21cf6dc73e6883a3ffce25b8459e51539bc))
* **01-03a:** unblock mypy — drop scripts/, drop dead self-ref, enable pydantic.mypy plugin (D-13, D-26) ([8fd3799](https://github.com/gseg-ethz/GSEGUtils/commit/8fd3799c4f87baee7204b94a6145071650ff82b6))
* **01-05:** unpin sphinx from git commit, switch to ~= 8.2 (D-18, D-19) ([367aa73](https://github.com/gseg-ethz/GSEGUtils/commit/367aa73d9cb509003ebcb1997d533d891470aa08))
* **04-04:** add psutil runtime dep, pytest-benchmark dev dep, pytest marker registration ([6e659a8](https://github.com/gseg-ethz/GSEGUtils/commit/6e659a85806675cd436c18986f2982a9eafb5a41))
* **04-04:** exclude benchmark marker from CI pytest invocation ([5f63660](https://github.com/gseg-ethz/GSEGUtils/commit/5f63660c06cbbc6d5d9cdbc9a36305e39a423aa1))
* **06-02:** DEBT-09 delete commented test_freeze scaffold (re-anchored) ([e413d2a](https://github.com/gseg-ethz/GSEGUtils/commit/e413d2ad8e8afc521ebefa87b18e569906cdc031))
* **deps:** pin numpy &gt;= 2.0, &lt; 2.4 and tighten requires-python to &gt;=3.12,&lt;3.13 ([bc075ff](https://github.com/gseg-ethz/GSEGUtils/commit/bc075ffa08623bbe1a6e0da59dc8c24480d8f9f3))
* signal release-please target for v0.5.0 ([501f225](https://github.com/gseg-ethz/GSEGUtils/commit/501f2251ea7a42eee58fc3d72ad563582c89eed7))


### ✅ Tests

* **01-04:** add GSEGUtils-only validator contract tests (D-15) ([017d996](https://github.com/gseg-ethz/GSEGUtils/commit/017d996c99cd3eced982f66c5dd440af8573392f))
* **01-04:** assert DeprecationWarning for legacy angle helper aliases (D-16) ([87ff201](https://github.com/gseg-ethz/GSEGUtils/commit/87ff201b2932cd1db053068a7c0c5319e4170974))
* **02-01:** add lazy_disk_cache regression tests (codec, legacy refusal, W-5) ([16ca149](https://github.com/gseg-ethz/GSEGUtils/commit/16ca1492417e7e63e5ca325eb4d50173bea7a01e))
* **02-04:** add FRAG-03 finalizer re-registration regression tests + W-1 sidecar coverage ([86d62ec](https://github.com/gseg-ethz/GSEGUtils/commit/86d62ec2d747e056fc93305615c49c3fe70c2f85))
* **02-05:** add FRAG-04 atomicity regression tests for DiskBackedStore codec ([c91033c](https://github.com/gseg-ethz/GSEGUtils/commit/c91033cb2e4a8fba9759f60fda66c7efa42f55ab))
* **03-01:** add BUG-01 ufunc + BUG-02 drop_buffer regressions ([f623e7a](https://github.com/gseg-ethz/GSEGUtils/commit/f623e7ad1003b18df13e222c65cc1f906ff49ecf))
* **04-04:** add PERF-04 unit tests + benchmark scaffolding + conftest fixtures ([957288c](https://github.com/gseg-ethz/GSEGUtils/commit/957288ce78a1bff1d8d6d41f4b570ca649ceeba4))
* **04-05:** add PERF-05 microbenchmark + canonical benchmarks/ dir ([29f5b35](https://github.com/gseg-ethz/GSEGUtils/commit/29f5b3545cf0dd41ffb15831f0494d96b47207dd))
* **04-05:** add PERF-05 SingletonMeta DCL test suite (RED) ([b0a13ec](https://github.com/gseg-ethz/GSEGUtils/commit/b0a13ecee9d66e8bde88ea65bb15a4a950e1aa8d))
* **04-06a:** COUPLE-05 GSEGUtils-side regression suite ([84126c7](https://github.com/gseg-ethz/GSEGUtils/commit/84126c7d24c2a5f71e24c76f4162cc0bb8390754))
* **04-06a:** RED — assert NaN/Inf raises + clip-and-saturate canary ([c9154c5](https://github.com/gseg-ethz/GSEGUtils/commit/c9154c5420f056ca008540dfe93cf6f5c84a1277))
* **06-01:** TEST-04 expand lazy_disk_cache pickle/finalizer coverage ([34440b2](https://github.com/gseg-ethz/GSEGUtils/commit/34440b207d92fef89f0debfd9cb651ad0b850284))
* **06-02:** TEST-06 add generate_init_stubs round-trip regression suite ([2bdac2f](https://github.com/gseg-ethz/GSEGUtils/commit/2bdac2fe9cdef3818128b3d9bb905bbc3f87e6ee))


### 🤖 Continuous Integration

* trigger workflow on push to refactor/gsd (in addition to main) ([0a54ffe](https://github.com/gseg-ethz/GSEGUtils/commit/0a54ffe4fc45c602d70277a1d2b6669e2e5fc647))

## [0.4.4](https://github.com/gseg-ethz/GSEGUtils/compare/v0.4.3...v0.4.4) (2026-01-14)


### 🧹 Miscellaneous Chores

* Adapt license language to be in line with ETH convention ([8d8ba06](https://github.com/gseg-ethz/GSEGUtils/commit/8d8ba06dcacf6fb9fb98c4cf846d7081a6612bd8))
* Lint ([e1c7da6](https://github.com/gseg-ethz/GSEGUtils/commit/e1c7da640e109417206e906ccedf5355e891554c))


### 🤖 Continuous Integration

* Add pytest-cov to dev dependencies ([af83441](https://github.com/gseg-ethz/GSEGUtils/commit/af83441b4de1d6f729652f145c4b6c28b9f59d23))
* Fix coverage error ([b2c7faa](https://github.com/gseg-ethz/GSEGUtils/commit/b2c7faa992ef7b12bac666dd61721c832ba0464c))
* Modify release-please to only run after other ci activities ([f09144f](https://github.com/gseg-ethz/GSEGUtils/commit/f09144f88537622eef9889ddb78004121390f845))

## [0.4.3](https://github.com/gseg-ethz/GSEGUtils/compare/v0.4.2...v0.4.3) (2026-01-09)


### 📚 Documentation

* Robust LICENSE copy ([cf499bc](https://github.com/gseg-ethz/GSEGUtils/commit/cf499bcf9dbf691ba2565b5565a2cc1663eec3f4))


### 🧹 Miscellaneous Chores

* Update `LICENSE` to reflect ETHZ guidelines ([29a46ae](https://github.com/gseg-ethz/GSEGUtils/commit/29a46ae3e521498ec59aa4597ba14e73da842dc0))

## [0.4.2](https://github.com/gseg-ethz/GSEGUtils/compare/v0.4.1...v0.4.2) (2026-01-09)


### ✨ Features

* extend normalise function to handle floating point in [-1, 1] ([9a3529b](https://github.com/gseg-ethz/GSEGUtils/commit/9a3529b02046599247224d2b8fc99980a7353512))
* **lazy_disk_cache:** `DiskCacheStore.offload` behavior change ([45a388d](https://github.com/gseg-ethz/GSEGUtils/commit/45a388d761694041516ecde3902325e7f2c018df))
* **lazy_disk_cache:** add generic `DiskBackedStore` ([53afeaa](https://github.com/gseg-ethz/GSEGUtils/commit/53afeaac2ff95401731446b4216eee4fffbc82a9))
* Logging-level selection added to 'setup_logging()' ([ffe91ea](https://github.com/gseg-ethz/GSEGUtils/commit/ffe91eab144e74119b8ecd8ea43816d48e6c57dd))


### 🐛 Bug Fixes

* include all submodules in `__init__` ([014c859](https://github.com/gseg-ethz/GSEGUtils/commit/014c8592456c5bf775153d89f2c5b84b8d3ba5b2))
* reorder log messages in `lazy_disk_cache` ([82f0873](https://github.com/gseg-ethz/GSEGUtils/commit/82f0873d7ea5adc28e3096717528749f7f96edbb))


### 📚 Documentation

* Merge branch 'doc' into dev ([8616f49](https://github.com/gseg-ethz/GSEGUtils/commit/8616f4956b84ee5fe956200c0b5920ac57d9f148))


### 🧹 Miscellaneous Chores

* adapt `.gitignore` and add `pyrightconfig.json` ([0f1207a](https://github.com/gseg-ethz/GSEGUtils/commit/0f1207a0503dfe666ca5fa759a97cf3df05b078a))
* merge remote-tracking branch 'origin/dev' into dev ([28e49d1](https://github.com/gseg-ethz/GSEGUtils/commit/28e49d1db9fd7fc344c28d17624e0301428fc246))
* merge remote-tracking branch 'origin/doc' into doc ([818ba6a](https://github.com/gseg-ethz/GSEGUtils/commit/818ba6a9533103e1d526baae8cf1a77dc95b7172))


### ♻️ Code Refactoring

* include stub files ([7ce5c3a](https://github.com/gseg-ethz/GSEGUtils/commit/7ce5c3a05c352b5b496c52c12295ce42b7b4ab92))


### 🔨 Build System

* adapt `pyproject.toml` and add `LICENSE` ([b4f5e05](https://github.com/gseg-ethz/GSEGUtils/commit/b4f5e0531eccac0f5884d9ec927a7d345fd7362e))
* adapt `pyproject.toml` and add `LICENSE` ([370381c](https://github.com/gseg-ethz/GSEGUtils/commit/370381c2707d9a4646e5abcc464fa3bb496ff1f8))
* add `pyrightconfig.json` ([6dacd5e](https://github.com/gseg-ethz/GSEGUtils/commit/6dacd5ed5298435c4078718dcde3ab1ec4ed7202))

## [0.4.1](https://github.com/gseg-ethz/GSEGUtils/compare/v0.4.0...v0.4.1) (2025-08-06)


### 🧹 Miscellaneous Chores

* Merge branch 'dev' into 'main' ([e3afd87](https://github.com/gseg-ethz/GSEGUtils/commit/e3afd878b0f1f2dad6794268a1a0d61ef116d751))
* Merge remote-tracking branch 'origin/dev' into dev ([e3f8705](https://github.com/gseg-ethz/GSEGUtils/commit/e3f870563e9786c3cb93eb89b959cb3971b334ad))

## [0.4.0](https://github.com/gseg-ethz/GSEGUtils/compare/v0.3.4...v0.4.0) (2025-08-06)


### ✨ Features

* add extra base Array_Nx2_T for Floating and Uint8 ([dabf33d](https://github.com/gseg-ethz/GSEGUtils/commit/dabf33d41503af407f89c3163f8acd7993fd7e32))
* Extend array types to cover most constrained shape and dtype combinations ([af85fc8](https://github.com/gseg-ethz/GSEGUtils/commit/af85fc80759e736798bb70aefd39977c960ce8f6))


### 🐛 Bug Fixes

* copying to 0D arrays failed in angle unit conversion ([f91ab62](https://github.com/gseg-ethz/GSEGUtils/commit/f91ab62c3633a12e09fbf8d12c56eb1784897c3b))
* update BaseArray code and ensure test coverage ([b7fda2b](https://github.com/gseg-ethz/GSEGUtils/commit/b7fda2b106420a7b1a6bfe7aa1382c1bfaac6f43))


### 🧹 Miscellaneous Chores

* add .idea directory to gitignore ([1e5339c](https://github.com/gseg-ethz/GSEGUtils/commit/1e5339c19d247a17468d1cc3496bac6e1bc82947))
* Merge of dev onto main for release 0.4.0 ([53b9d66](https://github.com/gseg-ethz/GSEGUtils/commit/53b9d66c63479a0cca1e7e2a892ce225edf023d8))


### ✅ Tests

* shift from PCHandler general pydantic tests to show expected behaviour ([c435897](https://github.com/gseg-ethz/GSEGUtils/commit/c435897803cf7d7a5e85249e79ca3a9191025ec5))

## [0.3.4](https://github.com/gseg-ethz/GSEGUtils/compare/v0.3.3...v0.3.4) (2025-08-05)


### 🤖 Continuous Integration

* Change fetch depth in release-please ([adb0041](https://github.com/gseg-ethz/GSEGUtils/commit/adb0041a3a323253dad13bbe1b89b5886049d0a6))
* Reverted release-type ([7cf2c8a](https://github.com/gseg-ethz/GSEGUtils/commit/7cf2c8a046b9a5e5f215d46deb15bf1a9cfe460d))

## [0.3.3](https://github.com/gseg-ethz/GSEGUtils/compare/v0.3.2...v0.3.3) (2025-07-31)


### 🐛 Bug Fixes

* **lazy_disk_cache:** Force deactivate `purge_disk_on_gc` ([6e27999](https://github.com/gseg-ethz/GSEGUtils/commit/6e27999fc1bd838b57e8b387e4f13b6f0aa49fda))

## [0.3.2](https://github.com/gseg-ethz/GSEGUtils/compare/v0.3.1...v0.3.2) (2025-07-29)


### ✨ Features

* Added DiskBackedNDArray ([ae2771a](https://github.com/gseg-ethz/GSEGUtils/commit/ae2771ae98feb66d17916c08e26dc2e7c83aaa26))

## [0.3.1](https://github.com/gseg-ethz/GSEGUtils/compare/v0.3.0...v0.3.1) (2025-07-29)


### ✨ Features

* Encapsuled setting for lazy disk caching into Config dataclass ([3aef2cb](https://github.com/gseg-ethz/GSEGUtils/commit/3aef2cbdda56e026ce125dbec69147f79a9d64f4))


### 🧹 Miscellaneous Chores

* Merge remote-tracking branch 'origin/main' ([a3c9b73](https://github.com/gseg-ethz/GSEGUtils/commit/a3c9b73dc07181c8923c79f355a5c8e9c096525f))

## [0.3.0](https://github.com/gseg-ethz/GSEGUtils/compare/v0.2.0...v0.3.0) (2025-07-23)


### ✨ Features

* Added logging setup ([ecaebc7](https://github.com/gseg-ethz/GSEGUtils/commit/ecaebc746ac4a6887072d1919f0ffb8b73adb752))


### 🔨 Build System

* Loosened numpy requirements ([e8f208f](https://github.com/gseg-ethz/GSEGUtils/commit/e8f208f91c27668c2cf0edf868ef8852f6372e37))
* Loosened numpy requirements further ([b21a14f](https://github.com/gseg-ethz/GSEGUtils/commit/b21a14fc876f9851df43a84d413e53cc317d1fab))

## [0.2.0](https://github.com/gseg-ethz/GSEGUtils/compare/v0.1.3...v0.2.0) (2025-07-23)


### 🧹 Miscellaneous Chores

* force version ([76a8242](https://github.com/gseg-ethz/GSEGUtils/commit/76a82427dbd583a752c940d04753bce2b1535a75))

## [0.1.3](https://github.com/gseg-ethz/GSEGUtils/compare/v0.1.2...v0.1.3) (2025-07-23)


### ✨ Features

* Added 'lazy_disk_cache' and 'singleton' meta ([bbb7af6](https://github.com/gseg-ethz/GSEGUtils/commit/bbb7af602e9930fc2481a1f0f2349dbbb03097ca))

## [0.1.2](https://github.com/gseg-ethz/GSEGUtils/compare/v0.1.1...v0.1.2) (2025-07-23)


### 🔨 Build System

* Changed version naming to ignore coarse tags ([3ada5e5](https://github.com/gseg-ethz/GSEGUtils/commit/3ada5e58a69e1d8a17ed67bcedc5728651aa95dd))

## [0.1.1](https://github.com/gseg-ethz/GSEGUtils/compare/v0.1.0...v0.1.1) (2025-07-23)


### 🧹 Miscellaneous Chores

* Added \'.gitignore\' ([d5e3245](https://github.com/gseg-ethz/GSEGUtils/commit/d5e32451269859b22399aa3f8a414588e5a312ce))
* Created baseline from 'pchandler' ([f7d475a](https://github.com/gseg-ethz/GSEGUtils/commit/f7d475a0c9fe1ceebfb8256b1331a9c2a41a9290))


### 🤖 Continuous Integration

* Changed permissions structure ([b4bfec9](https://github.com/gseg-ethz/GSEGUtils/commit/b4bfec90e363ca570f82015e2a70bba8209164aa))
* Updated release-please workflow to new token ([f916a36](https://github.com/gseg-ethz/GSEGUtils/commit/f916a36faa56aae4ee6915f52ee4f5216a65a4b7))
