# Changelog

## [0.0.11](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.10...v0.0.11) (2026-03-23)


### Features

* **operators:** add fourier physics operators ([3eb4138](https://github.com/jejjohnson/spectraldiffx/commit/3eb41382642c7a90f2c1590c144387af677f284e))
* **operators:** add Fourier physics operators ([381c4c2](https://github.com/jejjohnson/spectraldiffx/commit/381c4c2418e998b2a155e34011bf53860d5ae26b)), closes [#25](https://github.com/jejjohnson/spectraldiffx/issues/25)


### Bug Fixes

* address PR [#60](https://github.com/jejjohnson/spectraldiffx/issues/60) review comments ([2a37f07](https://github.com/jejjohnson/spectraldiffx/commit/2a37f07b6e8ed0b7998c680f72ed51f686214f1f))

## [0.0.10](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.9...v0.0.10) (2026-03-23)


### Features

* **solvers:** add inhomogeneous boundary condition support ([31b5569](https://github.com/jejjohnson/spectraldiffx/commit/31b5569391018034429f9036a39f895fb8f8667d))
* **solvers:** add inhomogeneous boundary condition support ([038580b](https://github.com/jejjohnson/spectraldiffx/commit/038580b799a958bd866ebdf747e8f5ae36fe0e88)), closes [#56](https://github.com/jejjohnson/spectraldiffx/issues/56)
* **solvers:** add mixed per-axis BC solver for 3D ([eabbc46](https://github.com/jejjohnson/spectraldiffx/commit/eabbc464d9acd676f964fb660895835502bbf37f))
* **solvers:** add mixed per-axis boundary condition solver for 3D ([f102410](https://github.com/jejjohnson/spectraldiffx/commit/f10241097ce5fbcef83c6b8d1a09a440fcae4055)), closes [#55](https://github.com/jejjohnson/spectraldiffx/issues/55)


### Bug Fixes

* **docs:** clarify inhomogeneous BC flowchart and fix matplotlib backend order ([3f48ed2](https://github.com/jejjohnson/spectraldiffx/commit/3f48ed2b2c813a76106e60d9331b69f85fc3c416))
* raise ValueError for periodic + inhomogeneous BC values ([f8fe687](https://github.com/jejjohnson/spectraldiffx/commit/f8fe68738db292f9e68695b7d88a1f979f081b84))

## [0.0.9](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.8...v0.0.9) (2026-03-22)


### Features

* add pseudo-spectral eigenvalues and `approximation` parameter to solvers ([61f3846](https://github.com/jejjohnson/spectraldiffx/commit/61f3846e6924ddf94455421dd5de8c11bbbc075f)), closes [#40](https://github.com/jejjohnson/spectraldiffx/issues/40)
* add pseudo-spectral eigenvalues and approximation parameter ([b2e57a4](https://github.com/jejjohnson/spectraldiffx/commit/b2e57a406539817799347f26a78e4fa3532fe683))
* **solvers:** add mixed per-axis boundary condition solver (2D) ([0ebaa25](https://github.com/jejjohnson/spectraldiffx/commit/0ebaa25b870bfaaff56833a2db142ecf6a6b6cf7))
* **solvers:** add mixed per-axis boundary condition solver for 2D ([c4a08be](https://github.com/jejjohnson/spectraldiffx/commit/c4a08bec1f2b94adf64d7b3313cf9a8db1448e9a)), closes [#39](https://github.com/jejjohnson/spectraldiffx/issues/39)


### Bug Fixes

* address PR [#52](https://github.com/jejjohnson/spectraldiffx/issues/52) review comments ([6224e74](https://github.com/jejjohnson/spectraldiffx/commit/6224e745531c58ba1b550bd8b9f15127af7cc08a))
* address PR [#53](https://github.com/jejjohnson/spectraldiffx/issues/53) review comments ([0b37d66](https://github.com/jejjohnson/spectraldiffx/commit/0b37d6663da6633fbbd16b57f260f610f545cd57))

## [0.0.8](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.7...v0.0.8) (2026-03-20)


### Bug Fixes

* **docs:** add MathJax config for $-delimiter support and SPA navigation ([4cd3c4d](https://github.com/jejjohnson/spectraldiffx/commit/4cd3c4d0390bf85755caf06db2277a7a4047a82f))
* **docs:** add placeholder images for OOM notebooks (navier_stokes_2d, qg_model) ([6298476](https://github.com/jejjohnson/spectraldiffx/commit/62984764be1e1df862a807584509df08af80e299))
* **docs:** correct image paths in notebook markdown embeds ([f2135a4](https://github.com/jejjohnson/spectraldiffx/commit/f2135a425ebb0b5295fdf2a3a8abde174719e976))

## [0.0.7](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.6...v0.0.7) (2026-03-20)


### Bug Fixes

* address PR [#48](https://github.com/jejjohnson/spectraldiffx/issues/48) review comments (7 items) ([7a487c5](https://github.com/jejjohnson/spectraldiffx/commit/7a487c57198522fb65d73be459a6e79092d69628))
* correct axis mismatch in demo_2d analytical derivatives ([85118fb](https://github.com/jejjohnson/spectraldiffx/commit/85118fbb47a654f0799088464a2ca296cc55f856))
* correct hyperviscosity sign and tune parameters in pseudospectral_part2 ([1f8ed19](https://github.com/jejjohnson/spectraldiffx/commit/1f8ed1967b63af776ba5e6f1ea2b01691b2c725e))
* improve orthogonality Gram matrix figure layout ([6ae1f3a](https://github.com/jejjohnson/spectraldiffx/commit/6ae1f3a545efec812ac09ab5d069d5db1d52052f))
* resolve cyclopts.Option → cyclopts.Parameter in CLI notebooks ([f6387b8](https://github.com/jejjohnson/spectraldiffx/commit/f6387b8b0e2b29204da9bf7e711db22159556b1c))
* stabilize demo_qg simulation and fix hyperviscosity sign ([6367c17](https://github.com/jejjohnson/spectraldiffx/commit/6367c17c4dc9e28d883f7d8f93c789e41fab8195))
* use independent colour scales for Poisson vs Helmholtz comparison ([81cefe0](https://github.com/jejjohnson/spectraldiffx/commit/81cefe00f0a6c17b251710134fe6418a187591db))

## [0.0.6](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.5...v0.0.6) (2026-03-19)


### Features

* add eigenvalue functions for all BC/grid-type combinations ([492d8b1](https://github.com/jejjohnson/spectraldiffx/commit/492d8b130fb294347230c5dd058ea1bc700a5552)), closes [#38](https://github.com/jejjohnson/spectraldiffx/issues/38)
* add solvers for all BC/grid-type combinations (1D, 2D, 3D) ([cafa3a6](https://github.com/jejjohnson/spectraldiffx/commit/cafa3a6f1b92f64fe17308e74d65f9f5c4c0946f)), closes [#38](https://github.com/jejjohnson/spectraldiffx/issues/38)
* staggered-grid eigenvalues and solvers for all BC/grid-type combinations ([4c72fad](https://github.com/jejjohnson/spectraldiffx/commit/4c72fadb7eb058b9129c667ec2433c58a7092d82))


### Bug Fixes

* address PR review comments ([be9fdaf](https://github.com/jejjohnson/spectraldiffx/commit/be9fdaf327cd663171e342d2ace52dc643f4d207))

## [0.0.5](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.4...v0.0.5) (2026-03-19)


### Features

* add spectral transforms, elliptic solvers, and capacitance to fourier/ ([1f34f3b](https://github.com/jejjohnson/spectraldiffx/commit/1f34f3b69ed4568dcd466579a7ded9d3a4894594))
* migrate spectral transforms, solvers, and capacitance from finitevolX ([f9759ac](https://github.com/jejjohnson/spectraldiffx/commit/f9759ac3e61a669f436e6e79039fe8abdc5eac31))


### Bug Fixes

* address PR review comments from Copilot ([1c29c8f](https://github.com/jejjohnson/spectraldiffx/commit/1c29c8fd12c2aa9c0df19c099f3724e053d91c88))
* resolve lint, format, and type-check errors ([615cb13](https://github.com/jejjohnson/spectraldiffx/commit/615cb13f920dbc24f65399c7f189f78032cfb108))

## [0.0.4](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.3...v0.0.4) (2026-03-10)


### Bug Fixes

* correct MkDocs configuration errors ([0533388](https://github.com/jejjohnson/spectraldiffx/commit/05333888721c4d55bc45702d74057883568bfeef))
* correct MkDocs configuration errors blocking docs deployment ([d5eaad3](https://github.com/jejjohnson/spectraldiffx/commit/d5eaad3411ae178d84ba330b1be74d86e8b2e26a))

## [0.0.3](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.2...v0.0.3) (2026-03-10)


### Features

* add jupytext notebooks, mkdocs.yml, and docs/ directory structure ([6f9d942](https://github.com/jejjohnson/spectraldiffx/commit/6f9d942477b1cb822c9f327298df18a91a297de4))


### Bug Fixes

* address review comments on documentation revamp ([3c944f1](https://github.com/jejjohnson/spectraldiffx/commit/3c944f1b302375db412adfd56c6165b1cad0ae5f))

## [0.0.2](https://github.com/jejjohnson/spectraldiffx/compare/v0.0.1...v0.0.2) (2026-03-10)


### Bug Fixes

* pages fetch-depth, labeler github-token, labeler.yml indentation ([14adc4f](https://github.com/jejjohnson/spectraldiffx/commit/14adc4f53116cba6373eaf274dfe07bc77996cb5))

## 0.0.1 (2026-02-28)


### Documentation

* overhaul README to reflect current package state ([9e6ffef](https://github.com/jejjohnson/spectraldiffx/commit/9e6ffef914339ed346039e8c5e39320b0cdc9d7b))
