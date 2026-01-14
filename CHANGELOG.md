# Changelog

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
